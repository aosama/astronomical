use crate::config_reload::{ResolvedRuntimeConfig, ResolvedRuntimeConfigResolver};
use crate::status_endpoint::status_check;
use crate::{
    ChatGenerationExecutor, WorkerHandle,
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
    routing::{get, post, put},
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
    pub(crate) generation_executor: Arc<dyn ChatGenerationExecutor>,
    pub(crate) worker_control: Option<WorkerHandle>,
    /// Static config warning surfaced once at daemon startup so the menu bar app
    /// can flash a callout (e.g. ignored `fixed_prefill_chunck_tokens`).
    pub(crate) config_warning: Option<Arc<str>>,
    /// All Qwen3.5-MoE-family models discovered from config directories at startup.
    pub(crate) discovered_models: Vec<DiscoveredModel>,
    /// Reloadable runtime config shared by the config-reload endpoint and the
    /// status/models endpoints. Present only when the application was built with
    /// reload support (`build_application_with_reload`).
    pub(crate) reloadable_config: Option<Arc<RwLock<ResolvedRuntimeConfig>>>,
    /// Startup-equivalent resolver used for each config reload.
    pub(crate) runtime_config_resolver: Option<ResolvedRuntimeConfigResolver>,
    /// Serializes config-file mutations and their worker acknowledgements.
    pub(crate) config_mutation_lock: Arc<AsyncMutex<()>>,
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
            config_warning: self.config_warning.clone(),
            discovered_models: self.discovered_models.clone(),
            reloadable_config: self.reloadable_config.clone(),
            runtime_config_resolver: self.runtime_config_resolver.clone(),
            config_mutation_lock: Arc::clone(&self.config_mutation_lock),
            shutdown_controller: self.shutdown_controller.clone(),
        }
    }
}

impl ApplicationState {
    pub(crate) fn configured_speculative_prefill_target_model_id(&self) -> Option<String> {
        self.reloadable_config
            .as_ref()
            .and_then(|reloadable_config| reloadable_config.read().ok())
            .and_then(|resolved_runtime_config| {
                if !resolved_runtime_config.speculative_prefill.is_enabled() {
                    return None;
                }
                resolved_runtime_config
                    .speculative_prefill
                    .target_model_id()
                    .map(str::to_owned)
            })
    }

    pub(crate) fn resolve_available_generation_model_id(
        &self,
        requested_model_id: &str,
        ready_model_id: Option<&str>,
    ) -> Option<String> {
        let known_model_ids: Vec<&str> = self
            .discovered_models
            .iter()
            .map(|discovered_model| discovered_model.model_id.as_str())
            .collect();
        let resolved_model_id =
            astronomical_config::resolve_model_id(requested_model_id, &known_model_ids);
        if self
            .configured_speculative_prefill_target_model_id()
            .is_some_and(|configured_target_model_id| {
                configured_target_model_id != resolved_model_id
            })
        {
            return None;
        }
        let is_ready_model = ready_model_id == Some(resolved_model_id);
        let is_discovered_model = self
            .discovered_models
            .iter()
            .any(|discovered_model| discovered_model.model_id == resolved_model_id);
        (is_ready_model || is_discovered_model).then(|| resolved_model_id.to_owned())
    }
}

/// Builds the bounded HTTP API using the supplied generation executor and no
/// startup config warning.
pub fn build_application(generation_executor: impl ChatGenerationExecutor) -> Router {
    build_application_with_config_warning_and_discovered_models(
        generation_executor,
        None,
        Vec::new(),
    )
}

/// Builds the bounded HTTP API and surfaces one static config warning through
/// `/v1/status` so the menu bar app can flash a callout to the user.
pub fn build_application_with_config_warning(
    generation_executor: impl ChatGenerationExecutor,
    config_warning: Option<String>,
) -> Router {
    build_application_with_config_warning_and_discovered_models(
        generation_executor,
        config_warning,
        Vec::new(),
    )
}

/// Builds the bounded HTTP API with config warning and discovered model listing.
pub fn build_application_with_config_warning_and_discovered_models(
    generation_executor: impl ChatGenerationExecutor,
    config_warning: Option<String>,
    discovered_models: Vec<DiscoveredModel>,
) -> Router {
    let application_state = ApplicationState {
        completion_id_namespace: completion_id_namespace(),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(generation_executor),
        worker_control: None,
        config_warning: config_warning.map(Arc::<str>::from),
        discovered_models,
        reloadable_config: None,
        runtime_config_resolver: None,
        config_mutation_lock: Arc::new(AsyncMutex::new(())),
        shutdown_controller: None,
    };

    application_router(application_state)
}

/// Builds the bounded HTTP API with an internal shutdown control endpoint.
/// The menu bar app calls `POST /v1/control/shutdown` to trigger a graceful
/// daemon exit without relying on OS signals.
pub fn build_application_with_shutdown(
    generation_executor: impl ChatGenerationExecutor,
    shutdown_controller: crate::shutdown_control::ShutdownController,
) -> Router {
    let application_state = ApplicationState {
        completion_id_namespace: completion_id_namespace(),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(generation_executor),
        worker_control: None,
        config_warning: None,
        discovered_models: Vec::new(),
        reloadable_config: None,
        runtime_config_resolver: None,
        config_mutation_lock: Arc::new(AsyncMutex::new(())),
        shutdown_controller: Some(shutdown_controller),
    };

    application_router(application_state)
}

/// Builds the bounded HTTP API with config-reload support. The supplied
/// `Arc<RwLock<ResolvedRuntimeConfig>>` is the live, reloadable runtime state.
/// The `config_home_directory` is used to reload `config.json` at runtime.
pub fn build_application_with_reload(
    generation_executor: impl ChatGenerationExecutor,
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    config_home_directory: PathBuf,
) -> Router {
    let fallback_worker_executable_path = reloadable_config
        .read()
        .ok()
        .map(|resolved_config| resolved_config.worker_executable_path.clone())
        .unwrap_or_default();
    let runtime_config_resolver =
        ResolvedRuntimeConfigResolver::new(config_home_directory, fallback_worker_executable_path);
    let initial_warning = reloadable_config
        .read()
        .ok()
        .and_then(|resolved| resolved.config_warning.clone());
    let initial_models = reloadable_config
        .read()
        .ok()
        .map(|resolved| resolved.discovered_models.clone())
        .unwrap_or_default();
    let application_state = ApplicationState {
        completion_id_namespace: completion_id_namespace(),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(generation_executor),
        worker_control: None,
        config_warning: initial_warning.map(Arc::<str>::from),
        discovered_models: initial_models,
        reloadable_config: Some(reloadable_config),
        runtime_config_resolver: Some(runtime_config_resolver),
        config_mutation_lock: Arc::new(AsyncMutex::new(())),
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
    Json(serde_json::json!({
        "persistent_prompt_cache_hits": persistent_prompt_cache_summary.hits,
        "persistent_prompt_cache_misses": persistent_prompt_cache_summary.misses,
        "persistent_prompt_cache_tokens_saved": persistent_prompt_cache_summary.tokens_saved,
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

/// Builds the bounded HTTP API with both config-reload and shutdown control.
/// This is the builder used by the production daemon so the menu bar app can
/// both reload config and request a graceful daemon restart.
pub fn build_application_with_full_control(
    worker_handle: WorkerHandle,
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    runtime_config_resolver: ResolvedRuntimeConfigResolver,
    shutdown_controller: crate::shutdown_control::ShutdownController,
) -> Router {
    let initial_warning = reloadable_config
        .read()
        .ok()
        .and_then(|resolved| resolved.config_warning.clone());
    let initial_models = reloadable_config
        .read()
        .ok()
        .map(|resolved| resolved.discovered_models.clone())
        .unwrap_or_default();
    let application_state = ApplicationState {
        completion_id_namespace: completion_id_namespace(),
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        generation_executor: Arc::new(worker_handle.clone()),
        worker_control: Some(worker_handle),
        config_warning: initial_warning.map(Arc::<str>::from),
        discovered_models: initial_models,
        reloadable_config: Some(reloadable_config),
        runtime_config_resolver: Some(runtime_config_resolver),
        config_mutation_lock: Arc::new(AsyncMutex::new(())),
        shutdown_controller: Some(shutdown_controller),
    };

    application_router(application_state)
}

fn application_router(application_state: ApplicationState) -> Router {
    let supports_config_reload = application_state.reloadable_config.is_some()
        && application_state.runtime_config_resolver.is_some();
    let supports_shutdown = application_state.shutdown_controller.is_some();
    let supports_live_mlx_memory_control =
        supports_config_reload && application_state.worker_control.is_some();
    let router = Router::new()
        .merge(console_routes())
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
    let router = if supports_config_reload {
        router.route(
            "/v1/config/reload",
            post(crate::config_reload_endpoint::reload_config),
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
