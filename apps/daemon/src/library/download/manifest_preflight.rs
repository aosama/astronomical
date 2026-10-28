//! Ordered, attributed journey from one catalog entry to exact durable transfer state.

use std::io;

use thiserror::Error;

use crate::{
    DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement, DaemonPerformanceOperation,
};

use crate::library::{
    DiskCapacityQuery, DownloadCatalogEntry, DownloadDiskCapacityCheck, DownloadDiskPreflight,
    DownloadDiskPreflightError, DownloadExecutablePreflight, DownloadExecutablePreflightError,
    DownloadJob, DownloadJobError, DownloadJobStore, DownloadJobStoreError, HuggingFaceHub,
    HuggingFaceHubError,
};

/// Owns the required disk-before-network ordering and durable phase transitions.
pub struct DownloadManifestPreflight<CapacityQuery> {
    job_store: DownloadJobStore,
    disk_preflight: DownloadDiskPreflight<CapacityQuery>,
    hub: HuggingFaceHub,
    attribution_log: DaemonPerformanceAttributionLog,
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
    #[error("the manifest does not describe an executable artifact: {0}")]
    ExecutableIdentity(#[from] DownloadExecutablePreflightError),
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
        attribution_log: DaemonPerformanceAttributionLog,
    ) -> Self {
        Self {
            job_store,
            disk_preflight,
            hub,
            attribution_log,
        }
    }

    /// Persists each restart boundary and never performs Hub I/O before initial disk admission.
    #[expect(clippy::too_many_lines)]
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

        // Two failure layers: the outer wait fails when the attribution
        // machinery itself errors; the inner Result is the measured operation.
        let initial_capacity_join = self
            .attribution_log
            .measure_blocking_operation(
                DaemonPerformanceOperation::DiskPreflight,
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
            .await;
        let initial_capacity_result = match initial_capacity_join {
            Ok(capacity_result) => capacity_result,
            Err(attribution_error) => {
                return Err(DownloadManifestPreflightError::Attribution(
                    attribution_error,
                ));
            }
        };
        let _initial_capacity = initial_capacity_result?;
        premanifest_job.mark_fetching_manifest(fetching_manifest_at_unix_millis)?;
        self.replace_job(&premanifest_job).await?;

        let manifest_fetch_join = self
            .attribution_log
            .measure_async_operation(
                DaemonPerformanceOperation::ManifestFetch,
                || {
                    self.hub.fetch_selected_manifest(
                        catalog_entry.huggingface_id(),
                        catalog_entry.revision(),
                        catalog_entry.download_path_selection(),
                    )
                },
                |manifest_outcome| match manifest_outcome {
                    Ok(manifest) => DaemonPerformanceMeasurement::validated_manifest_fetch(
                        true,
                        catalog_entry.huggingface_id(),
                        catalog_entry.revision(),
                        manifest.files().len(),
                        manifest.total_bytes(),
                    ),
                    Err(_) => DaemonPerformanceMeasurement::validated_manifest_fetch(
                        false,
                        catalog_entry.huggingface_id(),
                        catalog_entry.revision(),
                        0,
                        0,
                    ),
                },
            )
            .await;
        let manifest_fetch_result = match manifest_fetch_join {
            Ok(fetch_result) => fetch_result,
            Err(attribution_error) => {
                return Err(DownloadManifestPreflightError::Attribution(
                    attribution_error,
                ));
            }
        };
        let manifest_outcome = manifest_fetch_result?;
        self.validate_executable_identity(catalog_entry, &manifest_outcome)
            .await?;
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

    async fn validate_executable_identity(
        &self,
        catalog_entry: &DownloadCatalogEntry,
        manifest: &crate::library::HuggingFaceManifest,
    ) -> Result<(), DownloadManifestPreflightError> {
        let executable_preflight = DownloadExecutablePreflight::new(self.hub.clone());
        self.attribution_log
            .measure_async_operation(
                DaemonPerformanceOperation::ExecutablePreflight,
                || executable_preflight.validate(manifest),
                |validation_outcome| {
                    DaemonPerformanceMeasurement::validated_executable_preflight(
                        validation_outcome.is_ok(),
                        catalog_entry.huggingface_id(),
                        catalog_entry.revision(),
                        manifest.files().len(),
                    )
                },
            )
            .await
            .map_err(DownloadManifestPreflightError::Attribution)
            .and_then(|executable_identity_outcome| {
                executable_identity_outcome
                    .map_err(DownloadManifestPreflightError::ExecutableIdentity)
            })
    }

    async fn check_exact_capacity(
        &self,
        catalog_entry: &DownloadCatalogEntry,
        exact_job: &DownloadJob,
    ) -> Result<(), DownloadManifestPreflightError> {
        let exact_capacity_join = self
            .attribution_log
            .measure_blocking_operation(
                DaemonPerformanceOperation::DiskPreflight,
                {
                    let disk_preflight = self.disk_preflight.clone();
                    let models_directory = self.job_store.models_directory().to_path_buf();
                    let exact_job = exact_job.clone();
                    move || disk_preflight.check_job_remaining_bytes(&models_directory, &exact_job)
                },
                |capacity_outcome| disk_measurement(catalog_entry, capacity_outcome),
            )
            .await;
        let exact_capacity_result = match exact_capacity_join {
            Ok(capacity_result) => capacity_result,
            Err(attribution_error) => {
                return Err(DownloadManifestPreflightError::Attribution(
                    attribution_error,
                ));
            }
        };
        let _exact_capacity = exact_capacity_result?;
        Ok(())
    }

    async fn recover_job(
        &self,
        updated_at_unix_millis: u64,
    ) -> Result<Option<DownloadJob>, DownloadManifestPreflightError> {
        let job_store = self.job_store.clone();
        let recover_join = tokio::task::spawn_blocking(move || {
            job_store.load_recovering_interrupted_job(updated_at_unix_millis)
        })
        .await;
        let recover_result = match recover_join {
            Ok(recover_result) => recover_result,
            Err(join_error) => {
                return Err(DownloadManifestPreflightError::BackgroundTask(join_error));
            }
        };
        Ok(recover_result?)
    }

    async fn create_job(
        &self,
        download_job: &DownloadJob,
    ) -> Result<(), DownloadManifestPreflightError> {
        let job_store = self.job_store.clone();
        let download_job = download_job.clone();
        let create_join =
            tokio::task::spawn_blocking(move || job_store.create(&download_job)).await;
        let create_result = match create_join {
            Ok(create_result) => create_result,
            Err(join_error) => {
                return Err(DownloadManifestPreflightError::BackgroundTask(join_error));
            }
        };
        Ok(create_result?)
    }

    async fn replace_job(
        &self,
        replacement_job: &DownloadJob,
    ) -> Result<(), DownloadManifestPreflightError> {
        let job_store = self.job_store.clone();
        let replacement_job = replacement_job.clone();
        let replace_join =
            tokio::task::spawn_blocking(move || job_store.replace_current(&replacement_job)).await;
        let replace_result = match replace_join {
            Ok(replace_result) => replace_result,
            Err(join_error) => {
                return Err(DownloadManifestPreflightError::BackgroundTask(join_error));
            }
        };
        Ok(replace_result?)
    }
}

fn disk_measurement(
    catalog_entry: &DownloadCatalogEntry,
    capacity_outcome: &Result<DownloadDiskCapacityCheck, DownloadDiskPreflightError>,
) -> DaemonPerformanceMeasurement {
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
    DaemonPerformanceMeasurement::validated_disk_preflight(
        is_success,
        catalog_entry.huggingface_id(),
        catalog_entry.revision(),
        required_bytes,
        available_bytes,
    )
}
