//! Application builders that install the daemon-owned Library download boundary.

use std::sync::{Arc, RwLock, atomic::AtomicU64};

use axum::Router;
use tokio::sync::Mutex as AsyncMutex;

use crate::{
    ImageGenerationExecutor, ResolvedRuntimeConfig, ResolvedRuntimeConfigResolver, WorkerHandle,
    application::{ApplicationState, application_router},
    library::{DownloadCatalog, LibraryDownloadCoordinator},
};

pub fn build_application_with_full_control_and_library_download(
    worker_handle: WorkerHandle,
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    runtime_config_resolver: ResolvedRuntimeConfigResolver,
    shutdown_controller: crate::shutdown_control::ShutdownController,
    download_catalog: Arc<DownloadCatalog>,
    library_download_coordinator: Arc<LibraryDownloadCoordinator>,
) -> Router {
    let initial_models = reloadable_config
        .read()
        .ok()
        .map(|resolved| resolved.discovered_models.clone())
        .unwrap_or_default();
    let configured_config_snapshot = reloadable_config
        .read()
        .ok()
        .map(|resolved| Arc::new(RwLock::new(resolved.clone())));
    application_router(ApplicationState {
        completion_id_namespace: Arc::from("library-production"),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(worker_handle.clone()),
        worker_control: Some(worker_handle),
        download_catalog,
        library_download_coordinator: Some(library_download_coordinator),
        discovered_models: initial_models,
        reloadable_config: Some(reloadable_config),
        configured_config_snapshot,
        configuration_validation_error: Arc::new(RwLock::new(None)),
        runtime_config_resolver: Some(runtime_config_resolver),
        configuration_transition_lock: Arc::new(AsyncMutex::new(())),
        pending_memory_config_generation: Arc::new(AsyncMutex::new(None)),
        shutdown_controller: Some(shutdown_controller),
    })
}

pub fn build_application_with_library_download(
    generation_executor: impl ImageGenerationExecutor,
    download_catalog: Arc<DownloadCatalog>,
    library_download_coordinator: Arc<LibraryDownloadCoordinator>,
) -> Router {
    application_router(ApplicationState {
        completion_id_namespace: Arc::from("library-test"),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(generation_executor),
        worker_control: None,
        download_catalog,
        library_download_coordinator: Some(library_download_coordinator),
        discovered_models: Vec::new(),
        reloadable_config: None,
        configured_config_snapshot: None,
        configuration_validation_error: Arc::new(RwLock::new(None)),
        runtime_config_resolver: None,
        configuration_transition_lock: Arc::new(AsyncMutex::new(())),
        pending_memory_config_generation: Arc::new(AsyncMutex::new(None)),
        shutdown_controller: None,
    })
}
