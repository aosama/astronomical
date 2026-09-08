//! Bounded concurrent scheduling of pending payload files across one download journey.
//!
//! Per-stream endpoint throughput, not HTTP protocol version, bounds a single payload stream, so
//! the transfer window keeps several manifest files in flight at once. Each file still appends
//! contiguously from its recorded offset, which preserves the durability contract: a staged file's
//! synchronized length always equals its received bytes, so pause and restart recovery keep
//! reconciling durable progress from synchronized lengths with no durable-schema change.

use std::sync::Mutex;

use futures_util::stream::{FuturesUnordered, StreamExt};

use super::{
    DownloadJob, DownloadJobError, DownloadJobFile, DownloadPayloadTransfer,
    DownloadPayloadTransferError, DownloadProgressSnapshot,
};
use crate::{SupervisorPerformanceMeasurement, SupervisorPerformanceOperation};

/// Concurrent in-flight file transfers per journey. The bound is deliberately network-scoped:
/// stream count does not scale with model size or machine hardware, so one value serves every
/// laptop without configuration.
pub const MAXIMUM_CONCURRENT_PAYLOAD_FILE_TRANSFERS: usize = 4;

/// Why the bounded window stopped launching and streaming pending files.
#[derive(Debug)]
pub enum TransferWindowStop {
    Paused,
    TransferFailed(DownloadPayloadTransferError),
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

fn pending_files_largest_remaining_first(download_job: &DownloadJob) -> Vec<DownloadJobFile> {
    let mut pending_files: Vec<DownloadJobFile> = download_job
        .files()
        .iter()
        .filter(|download_file| download_file.bytes_on_disk() < download_file.expected_bytes())
        .cloned()
        .collect();
    // Largest remaining first: the longest pole starts before shorter files can delay it, and the
    // stable sort keeps manifest order for equal sizes.
    pending_files.sort_by(|left_file, right_file| {
        right_file
            .expected_bytes()
            .checked_sub(right_file.bytes_on_disk())
            .cmp(
                &left_file
                    .expected_bytes()
                    .checked_sub(left_file.bytes_on_disk()),
            )
    });
    pending_files
}

impl DownloadPayloadTransfer {
    pub(super) async fn download_pending_files_within_bounded_window(
        &self,
        download_job: &mut DownloadJob,
        updated_at_unix_millis: u64,
    ) -> Result<(), TransferWindowStop> {
        let observed_transfer_progress =
            ObservedTransferProgress::new(download_job.clone(), self.progress_snapshot().clone());
        // One shared-reference binding keeps the boxed per-file futures self-contained.
        let transfer = self;
        let observed_transfer_progress = &observed_transfer_progress;
        let mut pending_files = pending_files_largest_remaining_first(download_job);
        let mut in_flight_transfers = FuturesUnordered::new();
        let mut deferred_transfer_error: Option<DownloadPayloadTransferError> = None;

        loop {
            while in_flight_transfers.len() < MAXIMUM_CONCURRENT_PAYLOAD_FILE_TRANSFERS
                && deferred_transfer_error.is_none()
                && !self.is_pause_requested()
            {
                let Some(pending_file) = pending_files.pop() else {
                    break;
                };
                self.select_pending_file_durably(
                    download_job,
                    &pending_file,
                    updated_at_unix_millis,
                    &observed_transfer_progress,
                )
                .await
                .map_err(TransferWindowStop::TransferFailed)?;
                let transfer_job = download_job.clone();
                let attribution_log = self.attribution_log();
                let huggingface_id = download_job.huggingface_id().to_owned();
                let revision = download_job.revision().to_owned();
                let relative_path = pending_file.relative_path().to_owned();
                let resume_offset_bytes = pending_file.bytes_on_disk();
                in_flight_transfers.push(Box::pin(async move {
                    let transferred_bytes = attribution_log
                        .measure_async_operation(
                            SupervisorPerformanceOperation::FileTransfer,
                            || {
                                transfer.transfer_file(
                                    &transfer_job,
                                    &pending_file,
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
                                        &relative_path,
                                        resume_offset_bytes,
                                        transfer_outcome.as_ref().copied().unwrap_or(0),
                                    )
                                    .unwrap_or_else(|_| SupervisorPerformanceMeasurement::failure())
                            },
                        )
                        .await
                        .map_err(DownloadPayloadTransferError::Attribution)
                        .and_then(std::convert::identity);
                    (pending_file, transferred_bytes)
                }));
            }

            let Some((pending_file, transfer_result)) = in_flight_transfers.next().await else {
                break;
            };
            match transfer_result {
                Ok(transferred_bytes) => {
                    self.record_file_progress_durably(
                        download_job,
                        &pending_file,
                        transferred_bytes,
                        updated_at_unix_millis,
                    )
                    .await
                    .map_err(TransferWindowStop::TransferFailed)?;
                }
                Err(transfer_error) => {
                    deferred_transfer_error.get_or_insert(transfer_error);
                    // Stop the remaining streams at their next chunk boundary so their received
                    // bytes synchronize before the failure becomes durable.
                    self.request_pause();
                }
            }
        }

        match deferred_transfer_error {
            Some(transfer_error) => Err(TransferWindowStop::TransferFailed(transfer_error)),
            None if self.is_pause_requested() => Err(TransferWindowStop::Paused),
            None => Ok(()),
        }
    }
}
