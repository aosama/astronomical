//! Bounded concurrent scheduling of payload segments across one download journey.
//!
//! Per-stream endpoint throughput, not HTTP protocol version, bounds a single payload stream, so
//! the transfer window keeps several ranged requests in flight at once. Each pending file is
//! decomposed into fixed absolute segments (see `download_payload_transfer_segment`), and the
//! largest files contribute their first segments before any file's second segment launches, so
//! the longest pole starts early and the tail is never one lone multi-gigabyte shard.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{Arc, Mutex},
};

use futures_util::stream::{FuturesUnordered, StreamExt};

use super::{
    DownloadJob, DownloadJobError, DownloadJobFile, DownloadPayloadTransfer,
    DownloadPayloadTransferError, DownloadProgressSnapshot,
    download_payload_transfer_segment::{
        PayloadSegment, SegmentedFileAssembler, plan_payload_segments,
    },
};
use crate::{SupervisorPerformanceMeasurement, SupervisorPerformanceOperation};

/// Concurrent in-flight ranged transfers per journey. The bound is deliberately network-scoped:
/// stream count does not scale with model size or machine hardware, so one value serves every
/// laptop without configuration. Measured endpoint behavior shows aggregate throughput scales
/// with stream count, so the window keeps filling as earlier segments complete.
pub const MAXIMUM_CONCURRENT_PAYLOAD_TRANSFERS: usize = 8;

/// Why the bounded window stopped launching and streaming pending files.
#[derive(Debug)]
pub enum TransferWindowStop {
    Paused,
    TransferFailed(DownloadPayloadTransferError),
}

/// Completion state of one ranged segment after its bytes reached the assembler.
#[derive(Debug)]
pub(super) struct SegmentOutcome {
    pub(super) contiguous_end_bytes: u64,
    pub(super) file_completed: bool,
}

/// Process-local live progress shared by every in-flight transfer of one window.
///
/// Restart recovery derives authoritative progress from staged file lengths, so live refreshes
/// must not force synchronized metadata transactions into the network hot path; this owner keeps
/// the quarter-second user-interface cadence purely in memory.
pub(super) struct ObservedTransferProgress {
    live_job: Mutex<DownloadJob>,
    snapshot: DownloadProgressSnapshot,
}

impl ObservedTransferProgress {
    pub(super) fn new(download_job: DownloadJob, snapshot: DownloadProgressSnapshot) -> Self {
        Self {
            live_job: Mutex::new(download_job),
            snapshot,
        }
    }

    pub(super) fn select_file(
        &self,
        relative_path: &str,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        let mut live_job = self.lock_live_job();
        live_job.select_download_file(relative_path, updated_at_unix_millis)?;
        self.snapshot.publish(live_job.clone());
        Ok(())
    }

    pub(super) fn record_received_bytes(
        &self,
        relative_path: &str,
        bytes_on_disk: u64,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        let mut live_job = self.lock_live_job();
        live_job.record_file_progress(relative_path, bytes_on_disk, updated_at_unix_millis)?;
        self.snapshot.publish(live_job.clone());
        Ok(())
    }

    fn lock_live_job(&self) -> std::sync::MutexGuard<'_, DownloadJob> {
        self.live_job
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
    }
}

/// Orders pending files so the largest remaining file starts first, and interleaves their
/// segments round-wise so no file's later segments starve a smaller file's first segment.
fn pending_segment_queue(
    download_job: &DownloadJob,
) -> VecDeque<(DownloadJobFile, PayloadSegment)> {
    let mut pending_files: Vec<DownloadJobFile> = download_job
        .files()
        .iter()
        .filter(|download_file| download_file.bytes_on_disk() < download_file.expected_bytes())
        .cloned()
        .collect();
    // Ascending remaining so pop_front() pulls the largest remaining file first; the stable sort
    // keeps manifest order for equal sizes.
    pending_files.sort_by(|left_file, right_file| {
        left_file
            .expected_bytes()
            .checked_sub(left_file.bytes_on_disk())
            .cmp(
                &right_file
                    .expected_bytes()
                    .checked_sub(right_file.bytes_on_disk()),
            )
    });
    let planned_segments = pending_files
        .iter()
        .map(|pending_file| {
            (
                pending_file,
                plan_payload_segments(pending_file.bytes_on_disk(), pending_file.expected_bytes()),
            )
        })
        .collect::<Vec<_>>();
    let maximum_segment_count = planned_segments
        .iter()
        .map(|(_, planned_segments)| planned_segments.len())
        .max()
        .unwrap_or_default();
    let mut segment_queue = VecDeque::new();
    for segment_index in 0..maximum_segment_count {
        for (pending_file, planned_file_segments) in &planned_segments {
            if let Some(segment) = planned_file_segments.get(segment_index) {
                segment_queue.push_back(((*pending_file).clone(), segment.clone()));
            }
        }
    }
    segment_queue
}

impl DownloadPayloadTransfer {
    pub(super) async fn download_pending_files_within_bounded_window(
        &self,
        download_job: &mut DownloadJob,
        updated_at_unix_millis: u64,
    ) -> Result<(), TransferWindowStop> {
        let observed_transfer_progress =
            ObservedTransferProgress::new(download_job.clone(), self.progress_snapshot().clone());
        // One shared-reference binding keeps the boxed per-segment futures self-contained.
        let transfer = self;
        let observed_transfer_progress = &observed_transfer_progress;
        let mut segment_queue = pending_segment_queue(download_job);
        let mut in_flight_transfers = FuturesUnordered::new();
        let mut deferred_transfer_error: Option<DownloadPayloadTransferError> = None;
        let mut selected_relative_paths = BTreeSet::new();
        let mut completed_relative_paths = BTreeSet::new();
        let mut open_assemblers = BTreeMap::<String, Arc<SegmentedFileAssembler>>::new();

        loop {
            while in_flight_transfers.len() < MAXIMUM_CONCURRENT_PAYLOAD_TRANSFERS
                && deferred_transfer_error.is_none()
                && !self.is_pause_requested()
            {
                let Some((pending_file, payload_segment)) = segment_queue.pop_front() else {
                    break;
                };
                let relative_path = pending_file.relative_path();
                if !selected_relative_paths.contains(relative_path) {
                    self.select_pending_file_durably(
                        download_job,
                        &pending_file,
                        updated_at_unix_millis,
                        &observed_transfer_progress,
                    )
                    .await
                    .map_err(TransferWindowStop::TransferFailed)?;
                    selected_relative_paths.insert(relative_path.to_owned());
                    let job_store = self.job_store();
                    let assembler = SegmentedFileAssembler::open(
                        job_store.models_directory(),
                        &download_job
                            .staging_directory(job_store.models_directory())
                            .join(relative_path),
                        pending_file.bytes_on_disk(),
                    )
                    .await
                    .map_err(TransferWindowStop::TransferFailed)?;
                    open_assemblers.insert(relative_path.to_owned(), Arc::new(assembler));
                }
                let assembler = open_assemblers
                    .get(relative_path)
                    .cloned()
                    .expect("assembler should exist for every launched file");
                let transfer_job = download_job.clone();
                let attribution_log = self.attribution_log();
                let huggingface_id = download_job.huggingface_id().to_owned();
                let revision = download_job.revision().to_owned();
                in_flight_transfers.push(Box::pin(async move {
                    let segment_start_bytes = payload_segment.start_offset_bytes();
                    let outcome = attribution_log
                        .measure_async_operation(
                            SupervisorPerformanceOperation::FileTransfer,
                            || {
                                transfer.transfer_payload_segment(
                                    &transfer_job,
                                    &pending_file,
                                    &payload_segment,
                                    assembler,
                                    updated_at_unix_millis,
                                    &observed_transfer_progress,
                                )
                            },
                            |transfer_outcome| {
                                let measurement = if transfer_outcome.is_ok() {
                                    SupervisorPerformanceMeasurement::success()
                                } else {
                                    SupervisorPerformanceMeasurement::failure()
                                };
                                measurement
                                    .with_file_transfer(
                                        &huggingface_id,
                                        &revision,
                                        pending_file.relative_path(),
                                        segment_start_bytes,
                                        transfer_outcome.as_ref().map_or(0, |segment_outcome| {
                                            segment_outcome.contiguous_end_bytes
                                        }),
                                    )
                                    .unwrap_or_else(|_| SupervisorPerformanceMeasurement::failure())
                            },
                        )
                        .await
                        .map_err(DownloadPayloadTransferError::Attribution)
                        .and_then(std::convert::identity);
                    (pending_file, outcome)
                }));
            }

            let Some((pending_file, transfer_result)) = in_flight_transfers.next().await else {
                break;
            };
            match transfer_result {
                Ok(segment_outcome) => {
                    if segment_outcome.file_completed
                        && completed_relative_paths.insert(pending_file.relative_path().to_owned())
                    {
                        self.record_file_progress_durably(
                            download_job,
                            &pending_file,
                            segment_outcome.contiguous_end_bytes,
                            updated_at_unix_millis,
                        )
                        .await
                        .map_err(TransferWindowStop::TransferFailed)?;
                    }
                }
                Err(transfer_error) => {
                    deferred_transfer_error.get_or_insert(transfer_error);
                    // Stop launching new segments; the in-flight ones drain so their received
                    // bytes synchronize before the failure becomes durable.
                    self.request_pause();
                }
            }
        }

        self.finalize_open_assemblers(&open_assemblers)
            .await
            .map_err(TransferWindowStop::TransferFailed)?;
        match deferred_transfer_error {
            Some(transfer_error) => Err(TransferWindowStop::TransferFailed(transfer_error)),
            None if self.is_pause_requested() => Err(TransferWindowStop::Paused),
            None => Ok(()),
        }
    }

    async fn finalize_open_assemblers(
        &self,
        open_assemblers: &BTreeMap<String, Arc<SegmentedFileAssembler>>,
    ) -> Result<(), DownloadPayloadTransferError> {
        let finalize_outcomes = open_assemblers
            .values()
            .map(|assembler| assembler.finalize())
            .collect::<Vec<_>>();
        let mut deferred_error = None;
        for finalize_outcome in futures_util::future::join_all(finalize_outcomes).await {
            if let Err(finalize_error) = finalize_outcome {
                deferred_error.get_or_insert(finalize_error);
            }
        }
        deferred_error.map_or(Ok(()), Err)
    }
}
