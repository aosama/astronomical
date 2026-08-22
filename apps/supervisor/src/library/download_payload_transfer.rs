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
use tokio::io::AsyncWriteExt;

use super::{
    DownloadJob, DownloadJobPublicErrorCode, DownloadJobState, DownloadJobStore,
    DownloadJobStoreError, DownloadProgressSnapshot, HubPayloadRequest, HubPayloadTransport,
    HubTransportError,
    download_payload_response::{payload_url, validate_payload_response},
    download_payload_verification::verify_download_job,
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

        for file_index in 0..download_job.files().len() {
            let download_file = download_job.files()[file_index].clone();
            if download_file.bytes_on_disk() == download_file.expected_bytes() {
                continue;
            }
            if self.transfer_control.is_pause_requested() {
                return self.pause(updated_at_unix_millis).await;
            }
            download_job
                .select_download_file(download_file.relative_path(), updated_at_unix_millis)
                .map_err(DownloadJobStoreError::from)?;
            self.replace_job(download_job.clone()).await?;
            self.progress_snapshot.publish(download_job.clone());
            let resume_offset_bytes = download_file.bytes_on_disk();
            let huggingface_id = download_job.huggingface_id().to_owned();
            let revision = download_job.revision().to_owned();
            let relative_path = download_file.relative_path().to_owned();
            let transfer_outcome = self
                .attribution_log
                .measure_async_operation(
                    SupervisorPerformanceOperation::FileTransfer,
                    || self.transfer_file(&download_job, &download_file, updated_at_unix_millis),
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
                .map_err(DownloadPayloadTransferError::Attribution);
            let transferred_bytes = match transfer_outcome {
                Ok(Ok(transferred_bytes)) => transferred_bytes,
                Ok(Err(transfer_error)) => {
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
                Err(attribution_error) => return Err(attribution_error),
            };
            let completed_file_bytes = resume_offset_bytes
                .checked_add(transferred_bytes)
                .ok_or(DownloadPayloadTransferError::InvalidPayloadLength)?;
            download_job
                .record_file_progress(
                    download_file.relative_path(),
                    completed_file_bytes,
                    updated_at_unix_millis,
                )
                .map_err(DownloadJobStoreError::from)?;
            self.replace_job(download_job.clone()).await?;
            self.progress_snapshot.publish(download_job.clone());
            if self.transfer_control.is_pause_requested() {
                return self.pause(updated_at_unix_millis).await;
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
        if let Err(publication_error) = tokio::task::spawn_blocking(move || {
            publication_store.replace_current_for_publication(&publication_job)
        })
        .await
        .map_err(DownloadPayloadTransferError::Task)?
        {
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

    async fn transfer_file(
        &self,
        download_job: &DownloadJob,
        download_file: &super::DownloadJobFile,
        updated_at_unix_millis: u64,
    ) -> Result<u64, DownloadPayloadTransferError> {
        let resume_offset_bytes = download_file.bytes_on_disk();
        let payload_url = payload_url(
            download_job.huggingface_id(),
            download_job.revision(),
            download_file.relative_path(),
        )?;
        let payload_response = self
            .transport
            .execute_payload(HubPayloadRequest::get(payload_url, resume_offset_bytes))
            .await?;
        if matches!(payload_response.status(), 401 | 403) {
            return Err(DownloadPayloadTransferError::DownloadGated);
        }
        validate_payload_response(
            payload_response.status(),
            payload_response.content_range(),
            payload_response.content_length(),
            resume_offset_bytes,
            download_file.expected_bytes(),
        )?;
        let staged_file_path = download_job
            .staging_directory(self.job_store.models_directory())
            .join(download_file.relative_path());
        let models_directory = self.job_store.models_directory().to_path_buf();
        let open_path = staged_file_path.clone();
        let staged_file = tokio::task::spawn_blocking(move || {
            open_staged_file_for_append(&models_directory, &open_path, resume_offset_bytes)
        })
        .await
        .map_err(DownloadPayloadTransferError::Task)??;
        let mut staged_file = tokio::fs::File::from_std(staged_file);
        let expected_transfer_bytes = download_file.expected_bytes() - resume_offset_bytes;
        let mut transferred_bytes = 0_u64;
        let mut last_progress_publication_at = None;
        let mut observed_download_job = download_job.clone();
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
            let Some(updated_transferred_bytes) =
                transferred_bytes.checked_add(payload_chunk.len() as u64)
            else {
                deferred_transfer_error = Some(DownloadPayloadTransferError::InvalidPayloadLength);
                break;
            };
            if updated_transferred_bytes > expected_transfer_bytes {
                deferred_transfer_error = Some(DownloadPayloadTransferError::InvalidPayloadLength);
                break;
            }
            transferred_bytes = updated_transferred_bytes;
            staged_file
                .write_all(&payload_chunk)
                .await
                .map_err(DownloadPayloadTransferError::WritePayload)?;
            let should_publish_progress =
                last_progress_publication_at.is_none_or(|last_publication_at: Instant| {
                    last_publication_at.elapsed() >= LIVE_PROGRESS_PUBLICATION_INTERVAL
                });
            if should_publish_progress {
                observed_download_job
                    .record_file_progress(
                        download_file.relative_path(),
                        resume_offset_bytes + transferred_bytes,
                        updated_at_unix_millis,
                    )
                    .map_err(DownloadJobStoreError::from)?;
                self.progress_snapshot
                    .publish(observed_download_job.clone());
                last_progress_publication_at = Some(Instant::now());
            }
            if self.transfer_control.is_pause_requested() {
                break;
            }
        }
        staged_file
            .sync_all()
            .await
            .map_err(DownloadPayloadTransferError::WritePayload)?;
        if let Some(transfer_error) = deferred_transfer_error {
            return Err(transfer_error);
        }
        if !self.transfer_control.is_pause_requested()
            && transferred_bytes != expected_transfer_bytes
        {
            return Err(DownloadPayloadTransferError::InvalidPayloadLength);
        }
        Ok(transferred_bytes)
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
