use crate::config_reload::{ResolvedRuntimeConfig, ResolvedRuntimeConfigResolver};
use crate::library::{DownloadCatalog, library_catalog_routes};
use crate::status_endpoint::status_check;
use crate::{
    ImageGenerationExecutor, WorkerHandle,
    console_assets::console_routes,
    openai_chat_endpoint,
    openai_models_endpoint::{list_models, retrieve_model},
    openai_responses_endpoint,
    system_telemetry::system_telemetry_routes,
};
use astronomical_config::DiscoveredModel;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::Mutex as AsyncMutex;
/// Maximum JSON body accepted by the bounded OpenAI-compatible chat endpoint.
pub const MAX_OPENAI_CHAT_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;
// Request counters restart with each Router. A process/time/instance namespace
// prevents an OpenCode conversation from confusing tool-call IDs after either
// a daemon restart or multiple in-process Router constructions in tests.
static NEXT_APPLICATION_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct ApplicationState {
    pub(crate) completion_id_namespace: Arc<str>,
    pub(crate) next_chat_request_id: Arc<AtomicU64>,
    pub(crate) generation_executor: Arc<dyn ImageGenerationExecutor>,
    pub(crate) worker_control: Option<WorkerHandle>,
    /// Immutable release-bundled model catalog validated before production startup.
    pub(crate) download_catalog: Arc<DownloadCatalog>,
    /// Executable models discovered from configured directories at startup.
    pub(crate) discovered_models: Vec<DiscoveredModel>,
    /// Reloadable runtime config shared by the config-reload endpoint and the
    /// status/models endpoints. Present only when the application was built with
    /// Development reload support (`build_development_application_with_reload`).
    pub(crate) reloadable_config: Option<Arc<RwLock<ResolvedRuntimeConfig>>>,
    /// Most recently accepted persisted snapshot, which may intentionally lead
    /// the live serving snapshot while a restart is pending or failed.
    pub(crate) configured_config_snapshot: Option<Arc<RwLock<ResolvedRuntimeConfig>>>,
    /// Bounded public error retained while the durable document cannot be resolved.
    pub(crate) configuration_validation_error: Arc<RwLock<Option<String>>>,
    /// Startup-equivalent resolver used for each config reload.
    pub(crate) runtime_config_resolver: Option<ResolvedRuntimeConfigResolver>,
    /// Serializes request policy admission with configuration transitions and acknowledgements.
    pub(crate) configuration_transition_lock: Arc<AsyncMutex<()>>,
    /// Keeps a persisted live-memory update transactional until the worker
    /// either applies or rejects a value queued behind active generation.
    pub(crate) pending_memory_config_generation: Arc<AsyncMutex<Option<String>>>,
    /// Internal shutdown controller for `POST /v1/control/shutdown`.
    pub(crate) shutdown_controller: Option<crate::shutdown_control::ShutdownController>,
}

impl Clone for ApplicationState {
    fn clone(&self) -> Self {
        Self {
            completion_id_namespace: Arc::clone(&self.completion_id_namespace),
            next_chat_request_id: Arc::clone(&self.next_chat_request_id),
            generation_executor: Arc::clone(&self.generation_executor),
            worker_control: self.worker_control.clone(),
            download_catalog: Arc::clone(&self.download_catalog),
            discovered_models: self.discovered_models.clone(),
            reloadable_config: self.reloadable_config.clone(),
            configured_config_snapshot: self.configured_config_snapshot.clone(),
            configuration_validation_error: Arc::clone(&self.configuration_validation_error),
            runtime_config_resolver: self.runtime_config_resolver.clone(),
            configuration_transition_lock: Arc::clone(&self.configuration_transition_lock),
            pending_memory_config_generation: Arc::clone(&self.pending_memory_config_generation),
            shutdown_controller: self.shutdown_controller.clone(),
        }
    }
}

impl ApplicationState {
    /// Returns one consistent model-discovery snapshot for listing and routing.
    pub(crate) fn discovered_models_snapshot(&self) -> Vec<DiscoveredModel> {
        self.reloadable_config
            .as_ref()
            .and_then(|reloadable_config| {
                reloadable_config
                    .read()
                    .ok()
                    .map(|resolved_runtime_config| {
                        resolved_runtime_config.discovered_models.clone()
                    })
            })
            .unwrap_or_else(|| self.discovered_models.clone())
    }

    pub(crate) fn configured_speculative_prefill_target_model_id(&self) -> Option<String> {
        let ready_model_id = self
            .generation_executor
            .worker_health_snapshot()
            .ready_model_id?;
        self.reloadable_config
            .as_ref()
            .and_then(|reloadable_config| reloadable_config.read().ok())
            .and_then(|resolved_runtime_config| {
                let _configured_policy = resolved_runtime_config
                    .model_policy_catalog
                    .get(&ready_model_id)?
                    .acceleration_availability
                    .configured_speculative_prefill
                    .as_ref()?;
                Some(ready_model_id)
            })
    }

    pub(crate) fn resolve_available_generation_model_id(
        &self,
        requested_model_id: &str,
        ready_model_id: Option<&str>,
    ) -> Option<String> {
        let discovered_models = self.discovered_models_snapshot();
        let known_model_ids: Vec<&str> = discovered_models
            .iter()
            .map(|discovered_model| discovered_model.model_id.as_str())
            .collect();
        let resolved_model_id =
            astronomical_config::resolve_model_id(requested_model_id, &known_model_ids);
        let is_ready_model = ready_model_id == Some(resolved_model_id);
        let is_discovered_model = discovered_models
            .iter()
            .any(|discovered_model| discovered_model.model_id == resolved_model_id);
        (is_ready_model || is_discovered_model).then(|| resolved_model_id.to_owned())
    }
}

/// Builds the bounded HTTP API using the supplied generation executor.
pub fn build_application(generation_executor: impl ImageGenerationExecutor) -> Router {
    build_application_with_discovered_models(generation_executor, Vec::new())
}

/// Builds the bounded HTTP API with a discovered-model listing.
pub fn build_application_with_discovered_models(
    generation_executor: impl ImageGenerationExecutor,
    discovered_models: Vec<DiscoveredModel>,
) -> Router {
    let application_state = ApplicationState {
        completion_id_namespace: completion_id_namespace(),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(generation_executor),
        worker_control: None,
        download_catalog: Arc::new(DownloadCatalog::empty_v1()),
        discovered_models,
        reloadable_config: None,
        configured_config_snapshot: None,
        configuration_validation_error: Arc::new(RwLock::new(None)),
        runtime_config_resolver: None,
        configuration_transition_lock: Arc::new(AsyncMutex::new(())),
        pending_memory_config_generation: Arc::new(AsyncMutex::new(None)),
        shutdown_controller: None,
    };

    application_router(application_state)
}

/// Builds the bounded HTTP API with an explicitly validated download catalog.
pub fn build_application_with_download_catalog(
    generation_executor: impl ImageGenerationExecutor,
    download_catalog: DownloadCatalog,
) -> Router {
    let application_state = ApplicationState {
        completion_id_namespace: completion_id_namespace(),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(generation_executor),
        worker_control: None,
        download_catalog: Arc::new(download_catalog),
        discovered_models: Vec::new(),
        reloadable_config: None,
        configured_config_snapshot: None,
        configuration_validation_error: Arc::new(RwLock::new(None)),
        runtime_config_resolver: None,
        configuration_transition_lock: Arc::new(AsyncMutex::new(())),
        pending_memory_config_generation: Arc::new(AsyncMutex::new(None)),
        shutdown_controller: None,
    };

    application_router(application_state)
}

/// Builds the bounded HTTP API with an internal shutdown control endpoint.
/// The menu bar app calls `POST /v1/control/shutdown` to trigger a graceful
/// daemon exit without relying on OS signals.
pub fn build_application_with_shutdown(
    generation_executor: impl ImageGenerationExecutor,
    shutdown_controller: crate::shutdown_control::ShutdownController,
) -> Router {
    let application_state = ApplicationState {
        completion_id_namespace: completion_id_namespace(),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(generation_executor),
        worker_control: None,
        download_catalog: Arc::new(DownloadCatalog::empty_v1()),
        discovered_models: Vec::new(),
        reloadable_config: None,
        configured_config_snapshot: None,
        configuration_validation_error: Arc::new(RwLock::new(None)),
        runtime_config_resolver: None,
        configuration_transition_lock: Arc::new(AsyncMutex::new(())),
        pending_memory_config_generation: Arc::new(AsyncMutex::new(None)),
        shutdown_controller: Some(shutdown_controller),
    };

    application_router(application_state)
}

/// Builds the bounded HTTP API with config-reload support. The supplied
/// `Arc<RwLock<ResolvedRuntimeConfig>>` is the live, reloadable runtime state.
/// The `development_home_directory` supplies isolated Development config reloads.
pub fn build_development_application_with_reload(
    generation_executor: impl ImageGenerationExecutor,
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    development_home_directory: PathBuf,
) -> Router {
    let fallback_worker_executable_path = reloadable_config
        .read()
        .ok()
        .map(|resolved_config| resolved_config.worker_executable_path.clone())
        .unwrap_or_default();
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        development_home_directory,
        fallback_worker_executable_path,
    );
    let initial_models = reloadable_config
        .read()
        .ok()
        .map(|resolved| resolved.discovered_models.clone())
        .unwrap_or_default();
    let configured_config_snapshot = reloadable_config
        .read()
        .ok()
        .map(|resolved| Arc::new(RwLock::new(resolved.clone())));
    let application_state = ApplicationState {
        completion_id_namespace: completion_id_namespace(),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(generation_executor),
        worker_control: None,
        download_catalog: Arc::new(DownloadCatalog::empty_v1()),
        discovered_models: initial_models,
        reloadable_config: Some(reloadable_config),
        configured_config_snapshot,
        configuration_validation_error: Arc::new(RwLock::new(None)),
        runtime_config_resolver: Some(runtime_config_resolver),
        configuration_transition_lock: Arc::new(AsyncMutex::new(())),
        pending_memory_config_generation: Arc::new(AsyncMutex::new(None)),
        shutdown_controller: None,
    };

    application_router(application_state)
}

async fn health_check() -> &'static str {
    "ok"
}

async fn readiness_check(State(application_state): State<ApplicationState>) -> Response {
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    let readiness_body = worker_health_snapshot.status.as_str();

    if worker_health_snapshot.status.is_ready() {
        return (StatusCode::OK, readiness_body).into_response();
    }

    (StatusCode::SERVICE_UNAVAILABLE, readiness_body).into_response()
}

async fn cache_stats(State(application_state): State<ApplicationState>) -> Response {
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    let persistent_prompt_cache_summary = crate::PersistentPromptCacheSummary::from_worker_event(
        worker_health_snapshot
            .persistent_prompt_cache_stats
            .as_ref(),
    );
    let serving_session = &worker_health_snapshot.serving_session;
    let pending_cache_clear = worker_health_snapshot
        .pending_prompt_cache_clear
        .as_ref()
        .map(|pending_clear| {
            serde_json::json!({
                "model_id": pending_clear.model_id,
            })
        });
    Json(serde_json::json!({
        "persistent_prompt_cache_hits": persistent_prompt_cache_summary.hits,
        "persistent_prompt_cache_misses": persistent_prompt_cache_summary.misses,
        "persistent_prompt_cache_tokens_saved": persistent_prompt_cache_summary.tokens_saved,
        "persistent_prompt_cache_block_token_count": persistent_prompt_cache_summary.block_token_count,
        "persistent_prompt_cache_sequence_state_block_count": persistent_prompt_cache_summary.sequence_state_block_count,
        "persistent_prompt_cache_boundary_state_snapshot_count": persistent_prompt_cache_summary.boundary_state_snapshot_count,
        "persistent_prompt_cache_visual_embedding_count": persistent_prompt_cache_summary.visual_embedding_count,
        "persistent_prompt_cache_total_size_bytes": persistent_prompt_cache_summary.total_size_bytes,
        "persistent_prompt_cache_visual_embedding_total_size_bytes": persistent_prompt_cache_summary.visual_embedding_total_size_bytes,
        "persistent_prompt_cache_maximum_size_bytes": persistent_prompt_cache_summary.maximum_size_bytes,
        "persistent_prompt_cache_hit_rate": persistent_prompt_cache_summary.hit_rate(),
        "persistent_prompt_cache_visual_embedding_hits": persistent_prompt_cache_summary.visual_embedding_hits,
        "persistent_prompt_cache_visual_embedding_misses": persistent_prompt_cache_summary.visual_embedding_misses,
        "persistent_prompt_cache_visual_embedding_rows_loaded": persistent_prompt_cache_summary.visual_embedding_rows_loaded,
        "pending_cache_clear": pending_cache_clear,
        "speculative_prefill_cache_efficacy": {
            "target": {
                "eligible_token_count": serving_session.target_prompt_work_token_count,
                "restored_token_count": serving_session.target_reused_prompt_work_token_count,
                "reuse_rate": prompt_work_reuse_rate(
                    serving_session.target_reused_prompt_work_token_count,
                    serving_session.target_prompt_work_token_count,
                ),
            },
            "drafter": {
                "eligible_token_count": serving_session.drafter_prompt_work_token_count,
                "restored_token_count": serving_session.drafter_reused_prompt_work_token_count,
                "reuse_rate": prompt_work_reuse_rate(
                    serving_session.drafter_reused_prompt_work_token_count,
                    serving_session.drafter_prompt_work_token_count,
                ),
            },
            "combined": {
                "eligible_token_count": serving_session.target_prompt_work_token_count
                    .saturating_add(serving_session.drafter_prompt_work_token_count),
                "restored_token_count": serving_session.target_reused_prompt_work_token_count
                    .saturating_add(serving_session.drafter_reused_prompt_work_token_count),
                "reuse_rate": prompt_work_reuse_rate(
                    serving_session.target_reused_prompt_work_token_count
                        .saturating_add(serving_session.drafter_reused_prompt_work_token_count),
                    serving_session.target_prompt_work_token_count
                        .saturating_add(serving_session.drafter_prompt_work_token_count),
                ),
            },
        },
    }))
    .into_response()
}

fn prompt_work_reuse_rate(restored_token_count: u64, eligible_token_count: u64) -> f64 {
    if eligible_token_count == 0 {
        return 0.0;
    }
    restored_token_count.min(eligible_token_count) as f64 / eligible_token_count as f64
}

fn completion_id_namespace() -> Arc<str> {
    let started_at_unix_nanoseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration_since_epoch| duration_since_epoch.as_nanos())
        .unwrap_or_default();
    let process_id = std::process::id();
    let application_instance_id = NEXT_APPLICATION_INSTANCE_ID.fetch_add(1, Ordering::Relaxed);
    Arc::from(format!(
        "{started_at_unix_nanoseconds:x}-{process_id:x}-{application_instance_id:x}"
    ))
}

pub(crate) fn allocate_chat_request_id(next_chat_request_id: &AtomicU64) -> Option<u64> {
    next_chat_request_id
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current_request_id| {
            current_request_id.checked_add(1)
        })
        .ok()
}

/// Builds control-focused HTTP API variants with an empty test catalog.
pub fn build_application_with_full_control(
    worker_handle: WorkerHandle,
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    runtime_config_resolver: ResolvedRuntimeConfigResolver,
    shutdown_controller: crate::shutdown_control::ShutdownController,
) -> Router {
    build_application_with_full_control_and_download_catalog(
        worker_handle,
        reloadable_config,
        runtime_config_resolver,
        shutdown_controller,
        DownloadCatalog::empty_v1(),
    )
}

/// Builds the production HTTP API with its startup-validated catalog.
pub fn build_application_with_full_control_and_download_catalog(
    worker_handle: WorkerHandle,
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    runtime_config_resolver: ResolvedRuntimeConfigResolver,
    shutdown_controller: crate::shutdown_control::ShutdownController,
    download_catalog: DownloadCatalog,
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
    let application_state = ApplicationState {
        completion_id_namespace: completion_id_namespace(),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(worker_handle.clone()),
        worker_control: Some(worker_handle),
        download_catalog: Arc::new(download_catalog),
        discovered_models: initial_models,
        reloadable_config: Some(reloadable_config),
        configured_config_snapshot,
        configuration_validation_error: Arc::new(RwLock::new(None)),
        runtime_config_resolver: Some(runtime_config_resolver),
        configuration_transition_lock: Arc::new(AsyncMutex::new(())),
        pending_memory_config_generation: Arc::new(AsyncMutex::new(None)),
        shutdown_controller: Some(shutdown_controller),
    };

    application_router(application_state)
}

fn application_router(application_state: ApplicationState) -> Router {
    let supports_config_reload = application_state.reloadable_config.is_some()
        && application_state.runtime_config_resolver.is_some();
    let supports_shutdown = application_state.shutdown_controller.is_some();
    let supports_cache_clear = application_state.worker_control.is_some();
    let supports_live_mlx_memory_control =
        supports_config_reload && application_state.worker_control.is_some();
    let router = Router::new()
        .merge(console_routes())
        .merge(library_catalog_routes())
        .merge(system_telemetry_routes())
        .route("/health", get(health_check))
        .route("/ready", get(readiness_check))
        .route("/v1/status", get(status_check))
        .route("/v1/cache/stats", get(cache_stats))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{*model}", get(retrieve_model))
        .route(
            "/v1/chat/completions",
            post(openai_chat_endpoint::create_chat_completion)
                .layer(DefaultBodyLimit::max(MAX_OPENAI_CHAT_REQUEST_BODY_BYTES)),
        )
        .route(
            "/v1/responses",
            post(openai_responses_endpoint::create_response)
                .layer(DefaultBodyLimit::max(MAX_OPENAI_CHAT_REQUEST_BODY_BYTES)),
        );
    let router = router.route(
        "/v1/images/generations",
        post(crate::openai_image_generation_endpoint::create_image_generation)
            .layer(DefaultBodyLimit::max(MAX_OPENAI_CHAT_REQUEST_BODY_BYTES)),
    );
    let router = if supports_config_reload {
        router.route(
            "/v1/config/reload",
            post(crate::config_reload_endpoint::reload_config),
        )
    } else {
        router
    };
    let router = if supports_cache_clear {
        router.route(
            "/v1/cache",
            delete(crate::cache_clear_endpoint::clear_cache),
        )
    } else {
        router
    };
    let router = if supports_live_mlx_memory_control {
        router.route(
            "/v1/config/maximum-mlx-memory",
            put(crate::maximum_mlx_memory_endpoint::update_maximum_mlx_memory)
                .layer(DefaultBodyLimit::max(4 * 1024)),
        )
    } else {
        router
    };
    let router = if supports_shutdown {
        router.route(
            "/v1/control/shutdown",
            post(crate::shutdown_control::request_supervisor_shutdown),
        )
    } else {
        router
    };
    router.with_state(application_state)
}
