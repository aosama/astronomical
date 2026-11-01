use crate::config::config_reload::{ResolvedRuntimeConfig, ResolvedRuntimeConfigResolver};
use crate::library::{
    DownloadCatalog, LibraryDownloadCoordinator, library_catalog_routes, library_download_routes,
};
use crate::rest::status::status_check;
use crate::{
    ImageGenerationExecutor, WorkerHandle,
    console::console_assets::console_routes,
    rest::openai::chat_endpoint,
    rest::openai::models_endpoint::{list_models, retrieve_model},
    rest::openai::responses_endpoint,
    system_telemetry::system_telemetry_routes,
};
use astronomical_config::DiscoveredModel;
use astronomical_rest_contract::MAX_OPENAI_REQUEST_BODY_BYTES;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::{Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Mutex as AsyncMutex;
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::rest::openai::request_body_budget::enforce_openai_request_body_budget;
// Request counters restart with each Router. A process/time/instance namespace
// prevents an OpenCode conversation from confusing tool-call IDs after either
// a daemon restart or multiple in-process Router constructions in tests.
static NEXT_APPLICATION_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub(crate) struct ApplicationState {
    pub(crate) completion_id_namespace: Arc<str>,
    pub(crate) next_chat_request_id: Arc<AtomicU64>,
    pub(crate) generation_executor: Arc<dyn ImageGenerationExecutor>,
    pub(crate) worker_control: Option<WorkerHandle>,
    /// Immutable release-bundled model catalog validated before production startup.
    pub(crate) download_catalog: Arc<DownloadCatalog>,
    pub(crate) library_download_coordinator: Option<Arc<LibraryDownloadCoordinator>>,
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
        download_catalog: Arc::new(DownloadCatalog::empty()),
        library_download_coordinator: None,
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
        library_download_coordinator: None,
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
        download_catalog: Arc::new(DownloadCatalog::empty()),
        library_download_coordinator: None,
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
        download_catalog: Arc::new(DownloadCatalog::empty()),
        library_download_coordinator: None,
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
        "persistent_prompt_cache_partial_tail_hits": persistent_prompt_cache_summary.partial_tail_hits,
        "pending_cache_clear": pending_cache_clear,
    }))
    .into_response()
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
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |current_request_id| {
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
        DownloadCatalog::empty(),
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
        library_download_coordinator: None,
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

pub(crate) fn application_router(application_state: ApplicationState) -> Router {
    let supports_config_reload = application_state.reloadable_config.is_some()
        && application_state.runtime_config_resolver.is_some();
    let supports_shutdown = application_state.shutdown_controller.is_some();
    let supports_cache_clear = application_state.worker_control.is_some();
    let supports_live_mlx_memory_control =
        supports_config_reload && application_state.worker_control.is_some();
    let router = core_api_routes();
    let router = openai_generation_routes(router);
    let router = capability_conditional_routes(
        router,
        supports_config_reload,
        supports_cache_clear,
        supports_live_mlx_memory_control,
        supports_shutdown,
    );
    router
        .layer(canvas_cors_layer())
        .with_state(application_state)
}

/// The always-present routes: console/catalog/telemetry merges plus the
/// health, readiness, status, cache-stats, model listing, and the two
/// body-budgeted OpenAI text endpoints (chat completions and responses).
fn core_api_routes() -> Router<ApplicationState> {
    Router::new()
        .merge(console_routes())
        .merge(library_catalog_routes())
        .merge(library_download_routes())
        .merge(crate::rest::config_reveal::config_reveal_routes())
        .merge(system_telemetry_routes())
        .route("/health", get(health_check))
        .route("/ready", get(readiness_check))
        .route("/v1/status", get(status_check))
        .route("/v1/cache/stats", get(cache_stats))
        .route("/v1/experts/stats", get(crate::rest::experts::expert_stats))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{*model}", get(retrieve_model))
        .route(
            "/v1/chat/completions",
            post(chat_endpoint::create_chat_completion)
                .layer(DefaultBodyLimit::max(MAX_OPENAI_REQUEST_BODY_BYTES))
                .layer(axum::middleware::from_fn(
                    enforce_openai_request_body_budget,
                )),
        )
        .route(
            "/v1/responses",
            post(responses_endpoint::create_response)
                .layer(DefaultBodyLimit::max(MAX_OPENAI_REQUEST_BODY_BYTES))
                .layer(axum::middleware::from_fn(
                    enforce_openai_request_body_budget,
                )),
        )
}

/// The image-generation and embeddings endpoints, each under the same
/// OpenAI request body budget as the text endpoints.
fn openai_generation_routes(router: Router<ApplicationState>) -> Router<ApplicationState> {
    let router = router.route(
        "/v1/images/generations",
        post(crate::rest::openai::image_generation_endpoint::create_image_generation)
            .layer(DefaultBodyLimit::max(MAX_OPENAI_REQUEST_BODY_BYTES))
            .layer(axum::middleware::from_fn(
                enforce_openai_request_body_budget,
            )),
    );
    router.route(
        "/v1/embeddings",
        post(crate::rest::openai::embeddings_endpoint::create_embeddings)
            .layer(DefaultBodyLimit::max(MAX_OPENAI_REQUEST_BODY_BYTES))
            .layer(axum::middleware::from_fn(
                enforce_openai_request_body_budget,
            )),
    )
}

/// Adds each control-plane route only when the daemon was built with the
/// capability it exercises: config reload needs the reloadable config and
/// resolver, cache clear needs a worker, the live MLX memory ceiling needs
/// both reload and worker control, and shutdown needs the shutdown
/// controller.
fn capability_conditional_routes(
    router: Router<ApplicationState>,
    supports_config_reload: bool,
    supports_cache_clear: bool,
    supports_live_mlx_memory_control: bool,
    supports_shutdown: bool,
) -> Router<ApplicationState> {
    let router = if supports_config_reload {
        router.route(
            "/v1/config/reload",
            post(crate::rest::config_reload::reload_config),
        )
    } else {
        router
    };
    let router = if supports_cache_clear {
        router.route("/v1/cache", delete(crate::rest::cache_clear::clear_cache))
    } else {
        router
    };
    let router = if supports_live_mlx_memory_control {
        router.route(
            "/v1/config/maximum-mlx-memory",
            put(crate::rest::maximum_mlx_memory::update_maximum_mlx_memory)
                .layer(DefaultBodyLimit::max(4 * 1024)),
        )
    } else {
        router
    };
    if supports_shutdown {
        router.route(
            "/v1/control/shutdown",
            post(crate::shutdown_control::request_daemon_shutdown),
        )
    } else {
        router
    }
}

/// The CORS policy for the Thin Talk canvas: it runs inside a WKWebView
/// served from the thintalk-asset:// shell scheme and calls this REST API
/// directly with fetch, so the browser enforces CORS against that exact
/// origin. Only the shell origin, the loopback origins the page may also
/// target, and the methods/headers the client uses are allowed.
fn canvas_cors_layer() -> CorsLayer {
    let canvas_origins = [
        "thintalk-asset://shell".to_string(),
        "http://127.0.0.1".to_string(),
        "http://localhost".to_string(),
    ];
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, _| {
            canvas_origins.iter().any(|allowed| {
                origin
                    .to_str()
                    .map(|origin_value| {
                        origin_value == allowed
                            || (allowed.starts_with("http://")
                                && origin_value.starts_with(&format!("{allowed}:")))
                    })
                    .unwrap_or(false)
            })
        }))
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE])
        .max_age(Duration::from_secs(3600))
}
