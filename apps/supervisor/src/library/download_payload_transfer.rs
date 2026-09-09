//! One-file-at-a-time ranged transfer and provider-digest verification.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use thiserror::Error;

use super::{
    DownloadJob, DownloadJobPublicErrorCode, DownloadJobState, DownloadJobStore,
    DownloadJobStoreError, DownloadProgressSnapshot, HubPayloadRequest, HubPayloadTransport,
    HubTransportError,
    download_payload_response::{payload_url, validate_payload_response},
    download_payload_transfer_segment::{PayloadSegment, SegmentedFileAssembler},
    download_payload_transfer_window::{
        ObservedTransferProgress, SegmentOutcome, TransferWindowStop,
    },
    download_payload_verification::verify_download_job,
    download_publication_reconciliation::reconcile_publication_intent,
    download_staged_file::open_staged_file_for_append,
};
use crate::{
    SupervisorPerformanceAttributionLog, SupervisorPerformanceMeasurement,
    SupervisorPerformanceOperation,
};

const LIVE_PROGRESS_PUBLICATION_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Clone, Default)]
pub struct DownloadTransferControl {
    pause_requested: Arc<AtomicBool>,
}

#[derive(Debug)]
pub enum DownloadPayloadTransferOutcome {
    Paused(DownloadJob),
    ReadyToPublish(DownloadJob),
}

pub struct DownloadPayloadTransfer {
    job_store: DownloadJobStore,
    transport: Arc<dyn HubPayloadTransport>,
    attribution_log: SupervisorPerformanceAttributionLog,
    transfer_control: DownloadTransferControl,
    progress_snapshot: DownloadProgressSnapshot,
}

#[derive(Debug, Error)]
pub enum DownloadPayloadTransferError {
    #[error("durable download state failed: {0}")]
    JobStore(#[from] DownloadJobStoreError),
    #[error("download job is not ready for payload transfer")]
    InvalidJobState,
    #[error("payload transport failed: {0}")]
    Transport(#[from] HubTransportError),
    #[error("payload response status or range framing is invalid")]
    InvalidRangeResponse,
    #[error("the repository requires authentication or gated access")]
    DownloadGated,
    #[error("payload response exceeds or does not reach its manifest size")]
    InvalidPayloadLength,
    #[error("failed to write staged payload: {0}")]
    WritePayload(#[source] std::io::Error),
    #[error("payload digest for {relative_path} does not match provider evidence")]
    ChecksumMismatch { relative_path: String },
    #[error("supervisor performance attribution failed: {0}")]
    Attribution(#[source] std::io::Error),
    #[error("download task failed: {0}")]
    Task(#[source] tokio::task::JoinError),
}

impl DownloadTransferControl {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn request_pause(&self) {
        self.pause_requested.store(true, Ordering::Release);
    }

    fn is_pause_requested(&self) -> bool {
        self.pause_requested.load(Ordering::Acquire)
    }
}

impl DownloadPayloadTransfer {
    #[must_use]
    pub fn new(
        job_store: DownloadJobStore,
        transport: Arc<dyn HubPayloadTransport>,
        attribution_log: SupervisorPerformanceAttributionLog,
        transfer_control: DownloadTransferControl,
    ) -> Self {
        Self::with_progress_snapshot(
            job_store,
            transport,
            attribution_log,
            transfer_control,
            DownloadProgressSnapshot::new(),
        )
    }

    #[must_use]
    pub fn with_progress_snapshot(
        job_store: DownloadJobStore,
        transport: Arc<dyn HubPayloadTransport>,
        attribution_log: SupervisorPerformanceAttributionLog,
        transfer_control: DownloadTransferControl,
        progress_snapshot: DownloadProgressSnapshot,
    ) -> Self {
        Self {
            job_store,
            transport,
            attribution_log,
            transfer_control,
            progress_snapshot,
        }
    }

    pub async fn resume(
        &self,
        updated_at_unix_millis: u64,
    ) -> Result<DownloadPayloadTransferOutcome, DownloadPayloadTransferError> {
        let mut download_job = self.load_recovered_job(updated_at_unix_millis).await?;
        if !download_job.has_exact_manifest()
            || !matches!(
                download_job.state(),
                DownloadJobState::Paused | DownloadJobState::Failed
            )
        {
            return Err(DownloadPayloadTransferError::InvalidJobState);
        }
        download_job
            .mark_downloading(updated_at_unix_millis)
            .map_err(DownloadJobStoreError::from)?;
        self.replace_job(download_job.clone()).await?;
        self.progress_snapshot.publish(download_job.clone());

        self.replace_job(download_job.clone()).await?;
        self.progress_snapshot.publish(download_job.clone());

        match self
            .download_pending_files_within_bounded_window(&mut download_job, updated_at_unix_millis)
            .await
        {
            Ok(()) => {}
            Err(TransferWindowStop::Paused) => {
                return self.pause(updated_at_unix_millis).await;
            }
            Err(TransferWindowStop::TransferFailed(transfer_error)) => {
                let public_error_code =
                    if matches!(transfer_error, DownloadPayloadTransferError::DownloadGated) {
                        DownloadJobPublicErrorCode::DownloadGated
                    } else {
                        DownloadJobPublicErrorCode::DownloadFailed
                    };
                self.persist_failure(public_error_code, updated_at_unix_millis)
                    .await?;
                return Err(transfer_error);
            }
        }

        download_job
            .mark_verifying(updated_at_unix_millis)
            .map_err(DownloadJobStoreError::from)?;
        self.replace_job(download_job.clone()).await?;
        self.progress_snapshot.publish(download_job.clone());
        let verification_job = download_job.clone();
        let models_directory = self.job_store.models_directory().to_path_buf();
        let verification_outcome = self
            .attribution_log
            .measure_blocking_operation(
                SupervisorPerformanceOperation::Verification,
                move || verify_download_job(&models_directory, &verification_job),
                |verification_outcome| {
                    let measurement = if verification_outcome.is_ok() {
                        SupervisorPerformanceMeasurement::success()
                    } else {
                        SupervisorPerformanceMeasurement::failure()
                    };
                    measurement
                        .with_verification(
                            download_job.huggingface_id(),
                            download_job.revision(),
                            download_job.files().len(),
                            download_job.bytes_total(),
                        )
                        .unwrap_or_else(|_| SupervisorPerformanceMeasurement::failure())
                },
            )
            .await
            .map_err(DownloadPayloadTransferError::Attribution)?;
        if let Err(verification_error) = verification_outcome {
            if let DownloadPayloadTransferError::ChecksumMismatch { relative_path } =
                &verification_error
            {
                self.prepare_checksum_retry(
                    &mut download_job,
                    relative_path,
                    updated_at_unix_millis,
                )
                .await?;
            } else {
                self.persist_failure(
                    DownloadJobPublicErrorCode::ChecksumMismatch,
                    updated_at_unix_millis,
                )
                .await?;
            }
            return Err(verification_error);
        }
        download_job
            .mark_publishing(updated_at_unix_millis)
            .map_err(DownloadJobStoreError::from)?;
        let publication_job = download_job.clone();
        let publication_store = self.job_store.clone();
        let publication_reconciliation_outcome =
            reconcile_publication_intent(publication_store, &self.attribution_log, publication_job)
                .await
                .map_err(DownloadPayloadTransferError::Attribution)?;
        if let Err(publication_error) = publication_reconciliation_outcome {
            let public_error_code = if matches!(
                publication_error,
                DownloadJobStoreError::PublishedModelAlreadyExists
            ) {
                DownloadJobPublicErrorCode::ModelAlreadyPresent
            } else {
                DownloadJobPublicErrorCode::DownloadFailed
            };
            self.persist_failure(public_error_code, updated_at_unix_millis)
                .await?;
            return Err(DownloadPayloadTransferError::JobStore(publication_error));
        }
        Ok(DownloadPayloadTransferOutcome::ReadyToPublish(download_job))
    }

    /// Streams one segment's ranged response through the file's ordered assembler while
    /// publishing live progress. The assembler appends only at the contiguous prefix, so the
    /// staged file's synchronized length keeps equaling its received bytes for pause and restart
    /// reconciliation no matter which segment finishes first.
    pub(super) async fn transfer_payload_segment(
        &self,
        download_job: &DownloadJob,
        download_file: &super::DownloadJobFile,
        payload_segment: &PayloadSegment,
        assembler: std::sync::Arc<SegmentedFileAssembler>,
        updated_at_unix_millis: u64,
        observed_transfer_progress: &ObservedTransferProgress,
    ) -> Result<SegmentOutcome, DownloadPayloadTransferError> {
        let segment_start_bytes = payload_segment.start_offset_bytes();
        let expected_segment_bytes = payload_segment.end_offset_bytes() - segment_start_bytes;
        let payload_url = payload_url(
            download_job.huggingface_id(),
            download_job.revision(),
            download_file.relative_path(),
        )?;
        let payload_response = self
            .transport
            .execute_payload(HubPayloadRequest::get(payload_url, segment_start_bytes))
            .await?;
        if matches!(payload_response.status(), 401 | 403) {
            return Err(DownloadPayloadTransferError::DownloadGated);
        }
        validate_payload_response(
            payload_response.status(),
            payload_response.content_range(),
            payload_response.content_length(),
            segment_start_bytes,
            download_file.expected_bytes(),
        )?;
        let mut transferred_bytes = 0_u64;
        let mut last_progress_publication_at = None;
        let mut deferred_transfer_error = None;
        let mut payload_stream = payload_response.into_byte_stream();
        while let Some(payload_chunk) = payload_stream.next().await {
            let payload_chunk = match payload_chunk {
                Ok(payload_chunk) => payload_chunk,
                Err(transport_error) => {
                    deferred_transfer_error =
                        Some(DownloadPayloadTransferError::Transport(transport_error));
                    break;
                }
            };
            let remaining_segment_bytes = expected_segment_bytes - transferred_bytes;
            // Open-ended responses carry the whole remaining tail, so a single chunk may span
            // the segment boundary; consume only this segment's share and drop the rest.
            let consumed_chunk = if payload_chunk.len() as u64 > remaining_segment_bytes {
                payload_chunk.slice(0..remaining_segment_bytes as usize)
            } else {
                payload_chunk
            };
            let Some(updated_transferred_bytes) =
                transferred_bytes.checked_add(consumed_chunk.len() as u64)
            else {
                deferred_transfer_error = Some(DownloadPayloadTransferError::InvalidPayloadLength);
                break;
            };
            transferred_bytes = updated_transferred_bytes;
            assembler
                .write_chunk(
                    segment_start_bytes + transferred_bytes - consumed_chunk.len() as u64,
                    consumed_chunk,
                )
                .await?;
            let should_publish_progress =
                last_progress_publication_at.is_none_or(|last_publication_at: Instant| {
                    last_publication_at.elapsed() >= LIVE_PROGRESS_PUBLICATION_INTERVAL
                });
            if should_publish_progress {
                observed_transfer_progress
                    .record_received_bytes(
                        download_file.relative_path(),
                        assembler.contiguous_end_bytes().await,
                        updated_at_unix_millis,
                    )
                    .map_err(DownloadJobStoreError::from)?;
                last_progress_publication_at = Some(Instant::now());
            }
            if transferred_bytes == expected_segment_bytes {
                break;
            }
            if self.transfer_control.is_pause_requested() {
                break;
            }
        }
        if let Some(transfer_error) = deferred_transfer_error {
            return Err(transfer_error);
        }
        if !self.transfer_control.is_pause_requested()
            && transferred_bytes != expected_segment_bytes
        {
            return Err(DownloadPayloadTransferError::InvalidPayloadLength);
        }
        let contiguous_end_bytes = assembler.contiguous_end_bytes().await;
        // The file completes when the assembler's contiguous prefix reaches the expected size;
        // whichever segment's write advances the prefix past the file end observes it first.
        let file_completed = !self.transfer_control.is_pause_requested()
            && contiguous_end_bytes == download_file.expected_bytes();
        if file_completed {
            assembler.finalize().await?;
        }
        Ok(SegmentOutcome {
            contiguous_end_bytes,
            file_completed,
        })
    }

    async fn load_recovered_job(
        &self,
        updated_at_unix_millis: u64,
    ) -> Result<DownloadJob, DownloadPayloadTransferError> {
        let job_store = self.job_store.clone();
        tokio::task::spawn_blocking(move || {
            job_store.load_recovering_interrupted_job(updated_at_unix_millis)
        })
        .await
        .map_err(DownloadPayloadTransferError::Task)??
        .ok_or(DownloadPayloadTransferError::InvalidJobState)
    }

    async fn replace_job(
        &self,
        download_job: DownloadJob,
    ) -> Result<(), DownloadPayloadTransferError> {
        let job_store = self.job_store.clone();
        tokio::task::spawn_blocking(move || job_store.replace_current(&download_job))
            .await
            .map_err(DownloadPayloadTransferError::Task)??;
        Ok(())
    }

    pub(super) fn attribution_log(&self) -> &SupervisorPerformanceAttributionLog {
        &self.attribution_log
    }

    pub(super) fn job_store(&self) -> &DownloadJobStore {
        &self.job_store
    }

    pub(super) fn progress_snapshot(&self) -> &DownloadProgressSnapshot {
        &self.progress_snapshot
    }

    pub(super) fn is_pause_requested(&self) -> bool {
        self.transfer_control.is_pause_requested()
    }

    pub(super) fn request_pause(&self) {
        self.transfer_control.request_pause();
    }

    pub(super) async fn select_pending_file_durably(
        &self,
        download_job: &mut DownloadJob,
        pending_file: &super::DownloadJobFile,
        updated_at_unix_millis: u64,
        observed_transfer_progress: &ObservedTransferProgress,
    ) -> Result<(), DownloadPayloadTransferError> {
        download_job
            .select_download_file(pending_file.relative_path(), updated_at_unix_millis)
            .map_err(DownloadJobStoreError::from)?;
        self.replace_job(download_job.clone()).await?;
        observed_transfer_progress
            .select_file(pending_file.relative_path(), updated_at_unix_millis)
            .map_err(|_| DownloadPayloadTransferError::InvalidJobState)
    }

    pub(super) async fn record_file_progress_durably(
        &self,
        download_job: &mut DownloadJob,
        pending_file: &super::DownloadJobFile,
        bytes_on_disk: u64,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadPayloadTransferError> {
        download_job
            .record_file_progress(
                pending_file.relative_path(),
                bytes_on_disk,
                updated_at_unix_millis,
            )
            .map_err(DownloadJobStoreError::from)?;
        self.replace_job(download_job.clone()).await?;
        self.progress_snapshot.publish(download_job.clone());
        Ok(())
    }

    async fn prepare_checksum_retry(
        &self,
        download_job: &mut DownloadJob,
        relative_path: &str,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadPayloadTransferError> {
        let staged_file_path = download_job
            .staging_directory(self.job_store.models_directory())
            .join(relative_path);
        let expected_file_bytes = download_job
            .files()
            .iter()
            .find(|download_file| download_file.relative_path() == relative_path)
            .map(super::DownloadJobFile::expected_bytes)
            .ok_or(DownloadPayloadTransferError::InvalidJobState)?;
        let models_directory = self.job_store.models_directory().to_path_buf();
        tokio::task::spawn_blocking(move || -> Result<(), DownloadPayloadTransferError> {
            let staged_file = open_staged_file_for_append(
                &models_directory,
                &staged_file_path,
                expected_file_bytes,
            )?;
            staged_file
                .set_len(0)
                .map_err(DownloadPayloadTransferError::WritePayload)?;
            staged_file
                .sync_all()
                .map_err(DownloadPayloadTransferError::WritePayload)
        })
        .await
        .map_err(DownloadPayloadTransferError::Task)??;
        download_job
            .mark_checksum_failed_for_retry(relative_path, updated_at_unix_millis)
            .map_err(DownloadJobStoreError::from)?;
        self.replace_job(download_job.clone()).await?;
        self.progress_snapshot.publish(download_job.clone());
        Ok(())
    }

    async fn pause(
        &self,
        updated_at_unix_millis: u64,
    ) -> Result<DownloadPayloadTransferOutcome, DownloadPayloadTransferError> {
        let job_store = self.job_store.clone();
        let paused_job = tokio::task::spawn_blocking(move || {
            job_store.pause_current_job(updated_at_unix_millis)
        })
        .await
        .map_err(DownloadPayloadTransferError::Task)??
        .ok_or(DownloadPayloadTransferError::InvalidJobState)?;
        self.progress_snapshot.publish(paused_job.clone());
        Ok(DownloadPayloadTransferOutcome::Paused(paused_job))
    }

    async fn persist_failure(
        &self,
        error_code: DownloadJobPublicErrorCode,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadPayloadTransferError> {
        let job_store = self.job_store.clone();
        let failed_job = tokio::task::spawn_blocking(move || {
            let mut failed_job = job_store
                .pause_current_job(updated_at_unix_millis)?
                .ok_or(DownloadJobStoreError::JobNotFound)?;
            failed_job.mark_failed(error_code, updated_at_unix_millis)?;
            job_store.replace_current(&failed_job)?;
            Ok::<DownloadJob, DownloadJobStoreError>(failed_job)
        })
        .await
        .map_err(DownloadPayloadTransferError::Task)??;
        self.progress_snapshot.publish(failed_job);
        Ok(())
    }
}
