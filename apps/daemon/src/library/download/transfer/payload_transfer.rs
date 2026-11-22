//! Resume orchestration for one download: ranged transfer of payload files, creation of the
//! zero-byte files the transfer window never schedules, and provider-digest verification.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use thiserror::Error;

use super::payload_transfer_window::{ObservedTransferProgress, TransferWindowStop};
use crate::library::download::publication::reconciliation::reconcile_publication_intent;
use crate::library::download::transfer::payload_verification::verify_download_job;
use crate::library::download::transfer::staged_file::{
    EmptyFileCreation, create_empty_staged_files, open_staged_file_for_append,
};
use crate::library::{
    DownloadJob, DownloadJobPublicErrorCode, DownloadJobState, DownloadJobStore,
    DownloadJobStoreError, DownloadProgressSnapshot, HubPayloadTransport, HubTransportError,
};
use crate::{
    DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement, DaemonPerformanceOperation,
};

/// How one resumed download window ended for the resume orchestration:
/// download completed (continue to verification), an operator pause stopped
/// it (return the pause transition's own outcome), or the transfer failed
/// (persist the mapped public failure code, then re-raise the error).
enum ResumeDownloadOutcome {
    Completed,
    Paused(Result<DownloadPayloadTransferOutcome, DownloadPayloadTransferError>),
    Failed {
        public_error_code: DownloadJobPublicErrorCode,
        transfer_error: DownloadPayloadTransferError,
    },
}

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
    attribution_log: DaemonPerformanceAttributionLog,
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
    #[error("daemon performance attribution failed: {0}")]
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

    /// The shared flag itself, for blocking work that must stop between steps on a pause or cancel.
    fn pause_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.pause_requested)
    }
}

impl DownloadPayloadTransfer {
    #[must_use]
    pub fn new(
        job_store: DownloadJobStore,
        transport: Arc<dyn HubPayloadTransport>,
        attribution_log: DaemonPerformanceAttributionLog,
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
        attribution_log: DaemonPerformanceAttributionLog,
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

        match self
            .resume_download_window(&mut download_job, updated_at_unix_millis)
            .await?
        {
            ResumeDownloadOutcome::Completed => {}
            ResumeDownloadOutcome::Paused(pause_outcome) => return pause_outcome,
            ResumeDownloadOutcome::Failed {
                public_error_code,
                transfer_error,
            } => {
                self.persist_failure(public_error_code, updated_at_unix_millis)
                    .await?;
                return Err(transfer_error);
            }
        }

        self.verify_and_recover(&mut download_job, updated_at_unix_millis)
            .await?;
        let published_download_job = self
            .publish_reconciled_download_job(download_job, updated_at_unix_millis)
            .await?;
        Ok(DownloadPayloadTransferOutcome::ReadyToPublish(
            published_download_job,
        ))
    }

    /// Fetches every file the job still needs. Zero-byte files are created directly, because
    /// `pending_segment_queue` only queues files with bytes left, and the rest go through the
    /// bounded transfer window. A pause or cancel requested during the creation stops here as a
    /// pause, before any request is made.
    async fn download_all_pending_files(
        &self,
        download_job: &mut DownloadJob,
        updated_at_unix_millis: u64,
    ) -> Result<(), TransferWindowStop> {
        let empty_file_creation = self
            .create_empty_staged_files_off_runtime(download_job)
            .await
            .map_err(TransferWindowStop::TransferFailed)?;
        if empty_file_creation == EmptyFileCreation::Interrupted || self.is_pause_requested() {
            return Err(TransferWindowStop::Paused);
        }
        self.download_pending_files_within_bounded_window(download_job, updated_at_unix_millis)
            .await
    }

    /// Creates zero-byte files off the async runtime, like every other staged-file write. The
    /// thread cannot be aborted by a cancel, so it is handed the pause flag to stop between files.
    async fn create_empty_staged_files_off_runtime(
        &self,
        download_job: &DownloadJob,
    ) -> Result<EmptyFileCreation, DownloadPayloadTransferError> {
        let models_directory = self.job_store.models_directory().to_path_buf();
        let download_job_with_empty_files = download_job.clone();
        let pause_requested = self.transfer_control.pause_flag();
        match tokio::task::spawn_blocking(move || {
            create_empty_staged_files(
                &models_directory,
                &download_job_with_empty_files,
                &pause_requested,
            )
        })
        .await
        {
            Ok(creation_outcome) => Ok(creation_outcome?),
            Err(join_error) => Err(DownloadPayloadTransferError::Task(join_error)),
        }
    }

    /// Creates the job's empty files, then runs one bounded download window, and classifies how
    /// it ended. Paused
    /// carries the pause transition's own outcome for the caller to return;
    /// Failed carries both the public failure code (gated vs failed, so the
    /// operator sees why bytes stopped) and the transfer error to re-raise
    /// after the failure is persisted.
    async fn resume_download_window(
        &self,
        download_job: &mut DownloadJob,
        updated_at_unix_millis: u64,
    ) -> Result<ResumeDownloadOutcome, DownloadPayloadTransferError> {
        match self
            .download_all_pending_files(download_job, updated_at_unix_millis)
            .await
        {
            Ok(()) => Ok(ResumeDownloadOutcome::Completed),
            Err(TransferWindowStop::Paused) => Ok(ResumeDownloadOutcome::Paused(
                self.pause(updated_at_unix_millis).await,
            )),
            Err(TransferWindowStop::TransferFailed(transfer_error)) => {
                let public_error_code =
                    if matches!(transfer_error, DownloadPayloadTransferError::DownloadGated) {
                        DownloadJobPublicErrorCode::DownloadGated
                    } else {
                        DownloadJobPublicErrorCode::DownloadFailed
                    };
                Ok(ResumeDownloadOutcome::Failed {
                    public_error_code,
                    transfer_error,
                })
            }
        }
    }

    /// Marks the job verifying, re-publishes it, runs the measured checksum
    /// verification, and routes the failure: a checksum mismatch primes the
    /// retry path (the mismatched file alone is re-downloaded), any other
    /// verification failure persists a download failure, which a resume can
    /// retry. The verification error itself always propagates to the caller.
    async fn verify_and_recover(
        &self,
        download_job: &mut DownloadJob,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadPayloadTransferError> {
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
                DaemonPerformanceOperation::Verification,
                move || verify_download_job(&models_directory, &verification_job),
                |verification_outcome| {
                    let measurement = if verification_outcome.is_ok() {
                        DaemonPerformanceMeasurement::success()
                    } else {
                        DaemonPerformanceMeasurement::failure()
                    };
                    measurement
                        .with_verification(
                            download_job.huggingface_id(),
                            download_job.revision(),
                            download_job.files().len(),
                            download_job.bytes_total(),
                        )
                        .unwrap_or_else(|_| DaemonPerformanceMeasurement::failure())
                },
            )
            .await
            .map_err(DownloadPayloadTransferError::Attribution)?;
        if let Err(verification_error) = verification_outcome {
            if let DownloadPayloadTransferError::ChecksumMismatch { relative_path } =
                &verification_error
            {
                self.prepare_checksum_retry(download_job, relative_path, updated_at_unix_millis)
                    .await?;
            } else {
                self.persist_failure(
                    DownloadJobPublicErrorCode::DownloadFailed,
                    updated_at_unix_millis,
                )
                .await?;
            }
            return Err(verification_error);
        }
        Ok(())
    }

    /// Marks the job publishing and reconciles the publication intent. An
    /// already-published model maps to the dedicated public code; any other
    /// store failure persists a download failure. On success the marked job
    /// returns as the caller's ready-to-publish payload.
    async fn publish_reconciled_download_job(
        &self,
        download_job: DownloadJob,
        updated_at_unix_millis: u64,
    ) -> Result<DownloadJob, DownloadPayloadTransferError> {
        let mut download_job = download_job;
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
        Ok(download_job)
    }

    async fn load_recovered_job(
        &self,
        updated_at_unix_millis: u64,
    ) -> Result<DownloadJob, DownloadPayloadTransferError> {
        let job_store = self.job_store.clone();
        // Two failure layers: the join fails only when the blocking task
        // panicked or was aborted; the inner Result is the recovery outcome.
        let recover_join = tokio::task::spawn_blocking(move || {
            job_store.load_recovering_interrupted_job(updated_at_unix_millis)
        })
        .await;
        let recover_result = match recover_join {
            Ok(recover_result) => recover_result,
            Err(join_error) => return Err(DownloadPayloadTransferError::Task(join_error)),
        };
        recover_result?.ok_or(DownloadPayloadTransferError::InvalidJobState)
    }

    async fn replace_job(
        &self,
        download_job: DownloadJob,
    ) -> Result<(), DownloadPayloadTransferError> {
        let job_store = self.job_store.clone();
        let replace_join =
            tokio::task::spawn_blocking(move || job_store.replace_current(&download_job)).await;
        let replace_result = match replace_join {
            Ok(replace_result) => replace_result,
            Err(join_error) => return Err(DownloadPayloadTransferError::Task(join_error)),
        };
        replace_result?;
        Ok(())
    }

    pub(super) fn transport(&self) -> &dyn HubPayloadTransport {
        self.transport.as_ref()
    }

    pub(super) fn attribution_log(&self) -> &DaemonPerformanceAttributionLog {
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
        pending_file: &crate::library::DownloadJobFile,
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
        pending_file: &crate::library::DownloadJobFile,
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
            .map(crate::library::DownloadJobFile::expected_bytes)
            .ok_or(DownloadPayloadTransferError::InvalidJobState)?;
        let models_directory = self.job_store.models_directory().to_path_buf();
        let truncate_join =
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
            .await;
        let truncate_result = match truncate_join {
            Ok(truncate_result) => truncate_result,
            Err(join_error) => return Err(DownloadPayloadTransferError::Task(join_error)),
        };
        truncate_result?;
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
        let pause_join = tokio::task::spawn_blocking(move || {
            job_store.pause_current_job(updated_at_unix_millis)
        })
        .await;
        let pause_result = match pause_join {
            Ok(pause_result) => pause_result,
            Err(join_error) => return Err(DownloadPayloadTransferError::Task(join_error)),
        };
        let paused_job = pause_result?.ok_or(DownloadPayloadTransferError::InvalidJobState)?;
        self.progress_snapshot.publish(paused_job.clone());
        Ok(DownloadPayloadTransferOutcome::Paused(paused_job))
    }

    async fn persist_failure(
        &self,
        error_code: DownloadJobPublicErrorCode,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadPayloadTransferError> {
        let job_store = self.job_store.clone();
        let persist_join = tokio::task::spawn_blocking(move || {
            let mut failed_job = job_store
                .pause_current_job(updated_at_unix_millis)?
                .ok_or(DownloadJobStoreError::JobNotFound)?;
            failed_job.mark_failed(error_code, updated_at_unix_millis)?;
            job_store.replace_current(&failed_job)?;
            Ok::<DownloadJob, DownloadJobStoreError>(failed_job)
        })
        .await;
        let persist_result = match persist_join {
            Ok(persist_result) => persist_result,
            Err(join_error) => return Err(DownloadPayloadTransferError::Task(join_error)),
        };
        let failed_job = persist_result?;
        crate::problem_reports::report_download_failure(error_code, failed_job.huggingface_id());
        self.progress_snapshot.publish(failed_job);
        Ok(())
    }
}
