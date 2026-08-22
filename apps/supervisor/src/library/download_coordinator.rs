//! Background ownership for one daemon-wide durable Library download.

use std::{
    collections::BTreeSet,
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use thiserror::Error;
use tokio::{sync::Mutex, task::JoinHandle};

use super::{
    DiskCapacityQuery, DownloadCatalog, DownloadCatalogEntry, DownloadDiskPreflight, DownloadJob,
    DownloadJobPublicErrorCode, DownloadJobState, DownloadJobStore, DownloadJobStoreError,
    DownloadManifestPreflight, DownloadManifestPreflightError, DownloadPayloadTransfer,
    DownloadPayloadTransferOutcome, DownloadProgressSnapshot, DownloadPublication,
    DownloadPublicationRefresh, DownloadTransferControl, HubPayloadTransport, HubTransport,
    HuggingFaceHub,
};
use crate::SupervisorPerformanceAttributionLog;

#[derive(Clone)]
struct SharedDiskCapacityQuery(Arc<dyn DiskCapacityQuery>);

impl DiskCapacityQuery for SharedDiskCapacityQuery {
    fn available_space_bytes(&self, existing_same_volume_path: &Path) -> io::Result<u64> {
        self.0.available_space_bytes(existing_same_volume_path)
    }
}

struct ProgressPublishingRefresh {
    discovery_refresh: Arc<dyn DownloadPublicationRefresh>,
    progress_snapshot: DownloadProgressSnapshot,
    publishing_job: DownloadJob,
}

impl DownloadPublicationRefresh for ProgressPublishingRefresh {
    fn refresh(
        &self,
        published_directory: &Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // The destination exists at this boundary, so user interfaces may now report publishing
        // without racing ahead of the atomic staging rename.
        self.progress_snapshot.publish(self.publishing_job.clone());
        self.discovery_refresh.refresh(published_directory)
    }
}

/// Coordinates REST controls with one restart-safe background journey.
pub struct LibraryDownloadCoordinator {
    download_catalog: Arc<DownloadCatalog>,
    job_store: DownloadJobStore,
    capacity_query: SharedDiskCapacityQuery,
    metadata_transport: Arc<dyn HubTransport>,
    payload_transport: Arc<dyn HubPayloadTransport>,
    discovery_refresh: Arc<dyn DownloadPublicationRefresh>,
    attribution_log: SupervisorPerformanceAttributionLog,
    active_task: Mutex<Option<JoinHandle<()>>>,
    active_control: Mutex<Option<DownloadTransferControl>>,
    command_lock: Mutex<()>,
    validated_publications: Arc<Mutex<BTreeSet<String>>>,
    progress_snapshot: DownloadProgressSnapshot,
}

#[derive(Debug, Error)]
pub enum LibraryDownloadCoordinatorError {
    #[error("another Library download is active")]
    LibraryBusy,
    #[error("the requested model is not in the release catalog")]
    CatalogEntryNotFound,
    #[error("no resumable Library download exists")]
    JobNotFound,
    #[error("durable Library state failed: {0}")]
    JobStore(#[from] DownloadJobStoreError),
    #[error("Library background task failed: {0}")]
    BackgroundTask(#[from] tokio::task::JoinError),
}

impl LibraryDownloadCoordinator {
    #[must_use]
    pub fn new(
        download_catalog: Arc<DownloadCatalog>,
        models_directory: PathBuf,
        capacity_query: Arc<dyn DiskCapacityQuery>,
        metadata_transport: Arc<dyn HubTransport>,
        payload_transport: Arc<dyn HubPayloadTransport>,
        discovery_refresh: Arc<dyn DownloadPublicationRefresh>,
        attribution_log: SupervisorPerformanceAttributionLog,
    ) -> Self {
        Self {
            download_catalog,
            job_store: DownloadJobStore::new(models_directory),
            capacity_query: SharedDiskCapacityQuery(capacity_query),
            metadata_transport,
            payload_transport,
            discovery_refresh,
            attribution_log,
            active_task: Mutex::new(None),
            active_control: Mutex::new(None),
            command_lock: Mutex::new(()),
            validated_publications: Arc::new(Mutex::new(BTreeSet::new())),
            progress_snapshot: DownloadProgressSnapshot::new(),
        }
    }

    #[must_use]
    pub fn models_directory(&self) -> &Path {
        self.job_store.models_directory()
    }

    pub async fn recover_startup_state(&self) -> Result<(), LibraryDownloadCoordinatorError> {
        let _command_guard = self.command_lock.lock().await;
        let job_store = self.job_store.clone();
        let recovered_job = tokio::task::spawn_blocking(move || match job_store.load() {
            Ok(Some(download_job)) if download_job.state() == DownloadJobState::Publishing => {
                Ok(Some(download_job))
            }
            Ok(_) => job_store.load_recovering_interrupted_job(current_unix_millis()),
            Err(error) => Err(error),
        })
        .await??;
        if recovered_job.is_some_and(|job| job.state() == DownloadJobState::Publishing) {
            self.spawn_publication_task().await?;
        }
        Ok(())
    }

    pub async fn current_job(
        &self,
    ) -> Result<Option<DownloadJob>, LibraryDownloadCoordinatorError> {
        if self.has_active_task().await
            && let Some(download_job) = self.progress_snapshot.current()
        {
            return Ok(Some(download_job));
        }
        let job_store = self.job_store.clone();
        Ok(tokio::task::spawn_blocking(move || job_store.load()).await??)
    }

    pub async fn validated_publications_snapshot(&self) -> BTreeSet<String> {
        self.validated_publications.lock().await.clone()
    }

    #[must_use]
    pub fn destination_directory(&self, huggingface_id: &str) -> PathBuf {
        self.job_store.models_directory().join(huggingface_id)
    }

    pub async fn start(&self, huggingface_id: &str) -> Result<(), LibraryDownloadCoordinatorError> {
        let _command_guard = self.command_lock.lock().await;
        self.clear_finished_task().await;
        if self.has_active_task().await || self.current_job().await?.is_some() {
            return Err(LibraryDownloadCoordinatorError::LibraryBusy);
        }
        let catalog_entry = self
            .download_catalog
            .entries()
            .iter()
            .find(|catalog_entry| catalog_entry.huggingface_id() == huggingface_id)
            .cloned()
            .ok_or(LibraryDownloadCoordinatorError::CatalogEntryNotFound)?;
        self.spawn_download_task(catalog_entry).await;
        Ok(())
    }

    pub async fn resume(&self) -> Result<(), LibraryDownloadCoordinatorError> {
        let _command_guard = self.command_lock.lock().await;
        self.clear_finished_task().await;
        if self.has_active_task().await {
            return Err(LibraryDownloadCoordinatorError::LibraryBusy);
        }
        let download_job = self
            .current_job()
            .await?
            .ok_or(LibraryDownloadCoordinatorError::JobNotFound)?;
        if download_job.state() == DownloadJobState::Publishing {
            return self.spawn_publication_task().await;
        }
        if !matches!(
            download_job.state(),
            DownloadJobState::Paused | DownloadJobState::Failed
        ) {
            return Err(LibraryDownloadCoordinatorError::LibraryBusy);
        }
        let catalog_entry = self
            .download_catalog
            .entries()
            .iter()
            .find(|catalog_entry| {
                catalog_entry.huggingface_id() == download_job.huggingface_id()
                    && catalog_entry.revision() == download_job.revision()
            })
            .cloned()
            .ok_or(LibraryDownloadCoordinatorError::CatalogEntryNotFound)?;
        self.spawn_download_task(catalog_entry).await;
        Ok(())
    }

    pub async fn pause(&self) -> Result<DownloadJob, LibraryDownloadCoordinatorError> {
        let _command_guard = self.command_lock.lock().await;
        self.abort_active_task().await;
        let job_store = self.job_store.clone();
        tokio::task::spawn_blocking(move || job_store.pause_current_job(current_unix_millis()))
            .await??
            .ok_or(LibraryDownloadCoordinatorError::JobNotFound)
    }

    pub async fn cancel(&self) -> Result<bool, LibraryDownloadCoordinatorError> {
        let _command_guard = self.command_lock.lock().await;
        self.abort_active_task().await;
        let job_store = self.job_store.clone();
        Ok(tokio::task::spawn_blocking(move || job_store.cancel_current_job()).await??)
    }

    async fn spawn_download_task(&self, catalog_entry: DownloadCatalogEntry) {
        self.progress_snapshot.clear();
        let transfer_control = DownloadTransferControl::new();
        *self.active_control.lock().await = Some(transfer_control.clone());
        let task_dependencies = DownloadTaskDependencies {
            job_store: self.job_store.clone(),
            capacity_query: self.capacity_query.clone(),
            metadata_transport: Arc::clone(&self.metadata_transport),
            payload_transport: Arc::clone(&self.payload_transport),
            discovery_refresh: Arc::clone(&self.discovery_refresh),
            attribution_log: self.attribution_log.clone(),
            transfer_control,
            validated_publications: Arc::clone(&self.validated_publications),
            progress_snapshot: self.progress_snapshot.clone(),
        };
        *self.active_task.lock().await = Some(tokio::spawn(async move {
            if let Err(download_error) = run_download(catalog_entry, task_dependencies).await {
                tracing::warn!(error = %download_error, "Library download stopped");
            }
        }));
    }

    async fn spawn_publication_task(&self) -> Result<(), LibraryDownloadCoordinatorError> {
        let publishing_job = self
            .current_job()
            .await?
            .ok_or(LibraryDownloadCoordinatorError::JobNotFound)?;
        let huggingface_id = publishing_job.huggingface_id().to_owned();
        let publication =
            DownloadPublication::new(self.job_store.clone(), self.attribution_log.clone());
        let discovery_refresh = Arc::new(ProgressPublishingRefresh {
            discovery_refresh: Arc::clone(&self.discovery_refresh),
            progress_snapshot: self.progress_snapshot.clone(),
            publishing_job,
        });
        let validated_publications = Arc::clone(&self.validated_publications);
        *self.active_task.lock().await = Some(tokio::spawn(async move {
            match publication.publish(discovery_refresh).await {
                Ok(_) => {
                    validated_publications.lock().await.insert(huggingface_id);
                }
                Err(publication_error) => {
                    tracing::warn!(error = %publication_error, "Library publication recovery stopped");
                }
            }
        }));
        Ok(())
    }

    async fn abort_active_task(&self) {
        if let Some(transfer_control) = self.active_control.lock().await.take() {
            transfer_control.request_pause();
        }
        if let Some(mut active_task) = self.active_task.lock().await.take() {
            // Give the transfer loop a chance to synchronize its open file before cancellation;
            // the fallback remains bounded so an unresponsive transport cannot block controls.
            if tokio::time::timeout(Duration::from_secs(1), &mut active_task)
                .await
                .is_err()
            {
                active_task.abort();
                let _task_outcome = active_task.await;
            }
        }
    }

    async fn clear_finished_task(&self) {
        let mut active_task = self.active_task.lock().await;
        if active_task.as_ref().is_some_and(JoinHandle::is_finished) {
            let finished_task = active_task.take();
            drop(active_task);
            if let Some(finished_task) = finished_task {
                let _task_outcome = finished_task.await;
            }
            *self.active_control.lock().await = None;
        }
    }

    async fn has_active_task(&self) -> bool {
        self.active_task
            .lock()
            .await
            .as_ref()
            .is_some_and(|active_task| !active_task.is_finished())
    }
}

struct DownloadTaskDependencies {
    job_store: DownloadJobStore,
    capacity_query: SharedDiskCapacityQuery,
    metadata_transport: Arc<dyn HubTransport>,
    payload_transport: Arc<dyn HubPayloadTransport>,
    discovery_refresh: Arc<dyn DownloadPublicationRefresh>,
    attribution_log: SupervisorPerformanceAttributionLog,
    transfer_control: DownloadTransferControl,
    validated_publications: Arc<Mutex<BTreeSet<String>>>,
    progress_snapshot: DownloadProgressSnapshot,
}

async fn run_download(
    catalog_entry: DownloadCatalogEntry,
    dependencies: DownloadTaskDependencies,
) -> Result<(), String> {
    let timestamp = current_unix_millis();
    let preflight = DownloadManifestPreflight::new(
        dependencies.job_store.clone(),
        DownloadDiskPreflight::new(dependencies.capacity_query),
        HuggingFaceHub::new(dependencies.metadata_transport),
        dependencies.attribution_log.clone(),
    );
    if let Err(preflight_error) = preflight
        .prepare(&catalog_entry, timestamp, timestamp, timestamp)
        .await
    {
        persist_preflight_failure(
            &dependencies.job_store,
            preflight_public_error(&preflight_error),
        )
        .await;
        return Err(preflight_error.to_string());
    }
    let transfer = DownloadPayloadTransfer::with_progress_snapshot(
        dependencies.job_store.clone(),
        dependencies.payload_transport,
        dependencies.attribution_log.clone(),
        dependencies.transfer_control,
        dependencies.progress_snapshot.clone(),
    );
    match transfer.resume(current_unix_millis()).await {
        Ok(DownloadPayloadTransferOutcome::Paused(_)) => Ok(()),
        Ok(DownloadPayloadTransferOutcome::ReadyToPublish(publishing_job)) => {
            let discovery_refresh = Arc::new(ProgressPublishingRefresh {
                discovery_refresh: dependencies.discovery_refresh,
                progress_snapshot: dependencies.progress_snapshot,
                publishing_job,
            });
            DownloadPublication::new(dependencies.job_store, dependencies.attribution_log)
                .publish(discovery_refresh)
                .await
                .map_err(|error| error.to_string())?;
            dependencies
                .validated_publications
                .lock()
                .await
                .insert(catalog_entry.huggingface_id().to_owned());
            Ok(())
        }
        Err(transfer_error) => Err(transfer_error.to_string()),
    }
}

fn preflight_public_error(
    preflight_error: &DownloadManifestPreflightError,
) -> DownloadJobPublicErrorCode {
    match preflight_error {
        DownloadManifestPreflightError::Disk(_) => DownloadJobPublicErrorCode::InsufficientDisk,
        DownloadManifestPreflightError::Hub(super::HuggingFaceHubError::DownloadGated) => {
            DownloadJobPublicErrorCode::DownloadGated
        }
        _ => DownloadJobPublicErrorCode::DownloadFailed,
    }
}

async fn persist_preflight_failure(
    job_store: &DownloadJobStore,
    error_code: DownloadJobPublicErrorCode,
) {
    let job_store = job_store.clone();
    let failure_outcome = tokio::task::spawn_blocking(move || {
        let timestamp = current_unix_millis();
        let Some(mut failed_job) = job_store.pause_current_job(timestamp)? else {
            return Ok::<(), DownloadJobStoreError>(());
        };
        failed_job.mark_failed(error_code, timestamp)?;
        job_store.replace_current(&failed_job)
    })
    .await;
    if let Ok(Err(failure_error)) = failure_outcome {
        tracing::warn!(error = %failure_error, "Library failure state could not be persisted");
    }
}

fn current_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}
