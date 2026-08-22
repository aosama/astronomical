//! Ordered, attributed journey from one catalog entry to exact durable transfer state.

use std::io;

use thiserror::Error;

use crate::{
    SupervisorPerformanceAttributionLog, SupervisorPerformanceMeasurement,
    SupervisorPerformanceOperation,
};

use super::{
    DiskCapacityQuery, DownloadCatalogEntry, DownloadDiskCapacityCheck, DownloadDiskPreflight,
    DownloadDiskPreflightError, DownloadJob, DownloadJobError, DownloadJobStore,
    DownloadJobStoreError, HuggingFaceHub, HuggingFaceHubError,
};

/// Owns the required disk-before-network ordering and durable phase transitions.
pub struct DownloadManifestPreflight<CapacityQuery> {
    job_store: DownloadJobStore,
    disk_preflight: DownloadDiskPreflight<CapacityQuery>,
    hub: HuggingFaceHub,
    attribution_log: SupervisorPerformanceAttributionLog,
}

/// Typed failure from the ordered manifest-preflight journey.
#[derive(Debug, Error)]
pub enum DownloadManifestPreflightError {
    #[error("download job state is invalid: {0}")]
    Job(#[from] DownloadJobError),
    #[error("download job persistence failed: {0}")]
    Store(#[from] DownloadJobStoreError),
    #[error("download disk preflight failed: {0}")]
    Disk(#[from] DownloadDiskPreflightError),
    #[error("Hugging Face manifest retrieval failed: {0}")]
    Hub(#[from] HuggingFaceHubError),
    #[error("download performance attribution failed: {0}")]
    Attribution(#[from] io::Error),
    #[error("download blocking task failed: {0}")]
    BackgroundTask(#[from] tokio::task::JoinError),
}

impl<CapacityQuery> DownloadManifestPreflight<CapacityQuery>
where
    CapacityQuery: DiskCapacityQuery + Clone + 'static,
{
    #[must_use]
    pub fn new(
        job_store: DownloadJobStore,
        disk_preflight: DownloadDiskPreflight<CapacityQuery>,
        hub: HuggingFaceHub,
        attribution_log: SupervisorPerformanceAttributionLog,
    ) -> Self {
        Self {
            job_store,
            disk_preflight,
            hub,
            attribution_log,
        }
    }

    /// Persists each restart boundary and never performs Hub I/O before initial disk admission.
    pub async fn prepare(
        &self,
        catalog_entry: &DownloadCatalogEntry,
        checking_disk_at_unix_millis: u64,
        fetching_manifest_at_unix_millis: u64,
        manifest_ready_at_unix_millis: u64,
    ) -> Result<DownloadJob, DownloadManifestPreflightError> {
        let mut premanifest_job = match self.recover_job(checking_disk_at_unix_millis).await? {
            Some(existing_job) => {
                if existing_job.huggingface_id() != catalog_entry.huggingface_id()
                    || existing_job.revision() != catalog_entry.revision()
                {
                    return Err(DownloadJobStoreError::JobIdentityMismatch.into());
                }
                existing_job
            }
            None => {
                let new_job = DownloadJob::new_checking_disk(
                    catalog_entry.huggingface_id(),
                    catalog_entry.revision(),
                    catalog_entry.approximate_size_bytes(),
                    checking_disk_at_unix_millis,
                )?;
                self.create_job(&new_job).await?;
                new_job
            }
        };
        let models_directory = self.job_store.models_directory();

        if premanifest_job.has_exact_manifest() {
            self.check_exact_capacity(catalog_entry, &premanifest_job)
                .await?;
            return Ok(premanifest_job);
        }

        let _initial_capacity = self
            .attribution_log
            .measure_blocking_operation(
                SupervisorPerformanceOperation::DiskPreflight,
                {
                    let disk_preflight = self.disk_preflight.clone();
                    let models_directory = models_directory.to_path_buf();
                    let approximate_size_bytes = catalog_entry.approximate_size_bytes();
                    move || {
                        disk_preflight
                            .check_initial_download(&models_directory, approximate_size_bytes)
                    }
                },
                |capacity_outcome| disk_measurement(catalog_entry, capacity_outcome),
            )
            .await??;
        premanifest_job.mark_fetching_manifest(fetching_manifest_at_unix_millis)?;
        self.replace_job(&premanifest_job).await?;

        let manifest_outcome = self
            .attribution_log
            .measure_async_operation(
                SupervisorPerformanceOperation::ManifestFetch,
                || {
                    self.hub.fetch_selected_manifest(
                        catalog_entry.huggingface_id(),
                        catalog_entry.revision(),
                        catalog_entry.download_path_selection(),
                    )
                },
                |manifest_outcome| match manifest_outcome {
                    Ok(manifest) => SupervisorPerformanceMeasurement::validated_manifest_fetch(
                        true,
                        catalog_entry.huggingface_id(),
                        catalog_entry.revision(),
                        manifest.files().len(),
                        manifest.total_bytes(),
                    ),
                    Err(_) => SupervisorPerformanceMeasurement::validated_manifest_fetch(
                        false,
                        catalog_entry.huggingface_id(),
                        catalog_entry.revision(),
                        0,
                        0,
                    ),
                },
            )
            .await??;
        let exact_job =
            DownloadJob::from_manifest(&manifest_outcome, manifest_ready_at_unix_millis)?;
        self.replace_job(&exact_job).await?;
        let reconciled_exact_job = self
            .recover_job(manifest_ready_at_unix_millis)
            .await?
            .ok_or(DownloadJobStoreError::JobNotFound)?;
        self.check_exact_capacity(catalog_entry, &reconciled_exact_job)
            .await?;
        Ok(reconciled_exact_job)
    }

    async fn check_exact_capacity(
        &self,
        catalog_entry: &DownloadCatalogEntry,
        exact_job: &DownloadJob,
    ) -> Result<(), DownloadManifestPreflightError> {
        let _exact_capacity = self
            .attribution_log
            .measure_blocking_operation(
                SupervisorPerformanceOperation::DiskPreflight,
                {
                    let disk_preflight = self.disk_preflight.clone();
                    let models_directory = self.job_store.models_directory().to_path_buf();
                    let exact_job = exact_job.clone();
                    move || disk_preflight.check_job_remaining_bytes(&models_directory, &exact_job)
                },
                |capacity_outcome| disk_measurement(catalog_entry, capacity_outcome),
            )
            .await??;
        Ok(())
    }

    async fn recover_job(
        &self,
        updated_at_unix_millis: u64,
    ) -> Result<Option<DownloadJob>, DownloadManifestPreflightError> {
        let job_store = self.job_store.clone();
        Ok(tokio::task::spawn_blocking(move || {
            job_store.load_recovering_interrupted_job(updated_at_unix_millis)
        })
        .await??)
    }

    async fn create_job(
        &self,
        download_job: &DownloadJob,
    ) -> Result<(), DownloadManifestPreflightError> {
        let job_store = self.job_store.clone();
        let download_job = download_job.clone();
        Ok(tokio::task::spawn_blocking(move || job_store.create(&download_job)).await??)
    }

    async fn replace_job(
        &self,
        replacement_job: &DownloadJob,
    ) -> Result<(), DownloadManifestPreflightError> {
        let job_store = self.job_store.clone();
        let replacement_job = replacement_job.clone();
        Ok(
            tokio::task::spawn_blocking(move || job_store.replace_current(&replacement_job))
                .await??,
        )
    }
}

fn disk_measurement(
    catalog_entry: &DownloadCatalogEntry,
    capacity_outcome: &Result<DownloadDiskCapacityCheck, DownloadDiskPreflightError>,
) -> SupervisorPerformanceMeasurement {
    let (is_success, required_bytes, available_bytes) = match capacity_outcome {
        Ok(capacity_check) => (
            true,
            capacity_check.required_bytes(),
            capacity_check.available_bytes(),
        ),
        Err(DownloadDiskPreflightError::InsufficientSpace {
            required_bytes,
            available_bytes,
        }) => (false, *required_bytes, *available_bytes),
        Err(DownloadDiskPreflightError::QueryCapacity { required_bytes, .. }) => {
            (false, *required_bytes, 0)
        }
        Err(DownloadDiskPreflightError::RequiredBytesOverflow {
            catalog_approximate_bytes,
            margin_bytes,
        }) => (
            false,
            catalog_approximate_bytes.saturating_add(*margin_bytes),
            0,
        ),
        Err(DownloadDiskPreflightError::ExactManifestRequired) => (false, 0, 0),
    };
    SupervisorPerformanceMeasurement::validated_disk_preflight(
        is_success,
        catalog_entry.huggingface_id(),
        catalog_entry.revision(),
        required_bytes,
        available_bytes,
    )
}
