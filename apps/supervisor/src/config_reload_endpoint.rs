//! Internal localhost `POST /v1/config/reload` endpoint.

use std::{sync::Arc, time::Duration};

use astronomical_ipc_protocol::WorkerRuntimeFeatureConfiguration;
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

use crate::application::ApplicationState;
use crate::config_reload::{ConfigReloadDecision, ConfigReloadDiff};
use crate::{
    ChatGenerationExecutor, MlxMemoryLimitUpdateOutcome, WorkerActivity, WorkerControlError,
};

/// Internal response contract consumed only by the menu bar app.
#[derive(Serialize)]
struct ConfigReloadResponse {
    status: &'static str,
    message: String,
    worker_restart_completed: bool,
    rest_api_restart_required: bool,
    restart_required_fields: Vec<String>,
    reloaded_fields: Vec<String>,
    discovered_model_count: usize,
    worker_runtime_feature_configuration: Option<WorkerRuntimeFeatureConfiguration>,
}

impl ConfigReloadResponse {
    fn reloaded(reloaded_fields: Vec<String>, discovered_model_count: usize) -> Self {
        Self {
            status: "reloaded",
            message: "Config reloaded".to_owned(),
            worker_restart_completed: false,
            rest_api_restart_required: false,
            restart_required_fields: Vec::new(),
            reloaded_fields,
            discovered_model_count,
            worker_runtime_feature_configuration: None,
        }
    }

    fn restart_required(
        reloaded_fields: Vec<String>,
        restart_required_fields: Vec<String>,
        discovered_model_count: usize,
    ) -> Self {
        Self {
            status: "restart_required",
            message: "Config is valid, but a full server restart is required".to_owned(),
            worker_restart_completed: false,
            rest_api_restart_required: true,
            restart_required_fields,
            reloaded_fields,
            discovered_model_count,
            worker_runtime_feature_configuration: None,
        }
    }

    fn worker_restart_completed(
        reloaded_fields: Vec<String>,
        discovered_model_count: usize,
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration,
    ) -> Self {
        Self {
            status: "reloaded",
            message: "Config reloaded and applied by the worker".to_owned(),
            worker_restart_completed: true,
            rest_api_restart_required: false,
            restart_required_fields: Vec::new(),
            reloaded_fields,
            discovered_model_count,
            worker_runtime_feature_configuration: Some(worker_runtime_feature_configuration),
        }
    }

    fn invalid_config(validation_error: String) -> Self {
        Self {
            status: "invalid_config",
            message: validation_error,
            worker_restart_completed: false,
            rest_api_restart_required: false,
            restart_required_fields: Vec::new(),
            reloaded_fields: Vec::new(),
            discovered_model_count: 0,
            worker_runtime_feature_configuration: None,
        }
    }

    fn busy() -> Self {
        Self {
            status: "busy",
            message: "A generation is active or queued; reload aborted".to_owned(),
            worker_restart_completed: false,
            rest_api_restart_required: false,
            restart_required_fields: Vec::new(),
            reloaded_fields: Vec::new(),
            discovered_model_count: 0,
            worker_runtime_feature_configuration: None,
        }
    }

    fn failed(message: String, discovered_model_count: usize) -> Self {
        Self {
            status: "failed",
            message,
            worker_restart_completed: false,
            rest_api_restart_required: false,
            restart_required_fields: Vec::new(),
            reloaded_fields: Vec::new(),
            discovered_model_count,
            worker_runtime_feature_configuration: None,
        }
    }
}

/// Reloads config without cancelling active or queued generation work.
pub(crate) async fn reload_config(State(application_state): State<ApplicationState>) -> Response {
    let Some(reloadable_config) = application_state.reloadable_config.as_ref() else {
        return (StatusCode::NOT_FOUND, "reload not supported").into_response();
    };
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.as_ref() else {
        return (StatusCode::NOT_FOUND, "reload not supported").into_response();
    };
    let _config_mutation_guard = application_state.config_mutation_lock.lock().await;
    if application_state.worker_control.is_none()
        && application_state
            .generation_executor
            .worker_health_snapshot()
            .activity
            != WorkerActivity::Idle
    {
        return (StatusCode::CONFLICT, Json(ConfigReloadResponse::busy())).into_response();
    }

    let candidate_resolved = match runtime_config_resolver.load() {
        Ok(candidate_resolved) => candidate_resolved,
        Err(config_error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ConfigReloadResponse::invalid_config(
                    config_error.to_string(),
                )),
            )
                .into_response();
        }
    };
    let current_resolved = match reloadable_config.read() {
        Ok(current_resolved) => current_resolved.clone(),
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ConfigReloadResponse::failed(
                    "Internal config lock is unavailable".to_owned(),
                    0,
                )),
            )
                .into_response();
        }
    };
    let reload_decision = ConfigReloadDiff::compare(&current_resolved, &candidate_resolved);
    let discovered_model_count = reload_decision.discovered_model_count();
    let reloaded_fields = reload_decision.reloaded_fields().to_vec();
    let memory_limit_changed =
        current_resolved.maximum_mlx_memory_bytes != candidate_resolved.maximum_mlx_memory_bytes;
    let is_generation_busy = application_state.worker_control.as_ref().map_or_else(
        || {
            application_state
                .generation_executor
                .worker_health_snapshot()
                .activity
                != WorkerActivity::Idle
        },
        |worker_handle| !worker_handle.is_generation_idle_for_control_action(),
    );
    if is_generation_busy && reload_decision.worker_restart_required() {
        return (StatusCode::CONFLICT, Json(ConfigReloadResponse::busy())).into_response();
    }
    if memory_limit_changed
        && !reload_decision.worker_restart_required()
        && let Some(worker_handle) = application_state.worker_control.as_ref()
    {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        let effective_mlx_memory_ceiling_bytes = candidate_resolved
            .maximum_mlx_memory_bytes
            .unwrap_or(worker_health_snapshot.machine_mlx_memory_ceiling_bytes);
        if effective_mlx_memory_ceiling_bytes == 0
            || effective_mlx_memory_ceiling_bytes
                > worker_health_snapshot.machine_mlx_memory_ceiling_bytes
            || effective_mlx_memory_ceiling_bytes
                < worker_health_snapshot.minimum_mlx_memory_ceiling_bytes
        {
            return (
                StatusCode::BAD_REQUEST,
                Json(ConfigReloadResponse::invalid_config(
                    "maximum_mlx_memory_gb is outside the worker's reported limits".to_owned(),
                )),
            )
                .into_response();
        }
        match worker_handle
            .update_mlx_memory_limit(effective_mlx_memory_ceiling_bytes)
            .await
        {
            Ok(MlxMemoryLimitUpdateOutcome::Applied | MlxMemoryLimitUpdateOutcome::Queued) => {}
            Ok(MlxMemoryLimitUpdateOutcome::Rejected) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ConfigReloadResponse::invalid_config(
                        worker_handle
                            .worker_health_snapshot()
                            .mlx_memory_limit_error
                            .unwrap_or_else(|| "worker rejected the MLX memory limit".to_owned()),
                    )),
                )
                    .into_response();
            }
            Err(worker_control_error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ConfigReloadResponse::failed(
                        format!("could not apply the MLX memory limit: {worker_control_error}"),
                        discovered_model_count,
                    )),
                )
                    .into_response();
            }
        }
    }

    match reload_decision {
        ConfigReloadDecision::NoWorkerRestart { .. } => {
            let Ok(mut live_config) = reloadable_config.write() else {
                return internal_config_lock_error_response(discovered_model_count);
            };
            *live_config = candidate_resolved;
            (
                StatusCode::OK,
                Json(ConfigReloadResponse::reloaded(
                    reloaded_fields,
                    discovered_model_count,
                )),
            )
                .into_response()
        }
        ConfigReloadDecision::RestApiRestartRequired {
            restart_required_fields,
            ..
        } => {
            let Ok(mut live_config) = reloadable_config.write() else {
                return internal_config_lock_error_response(discovered_model_count);
            };
            apply_in_place_reload_fields(&mut live_config, &candidate_resolved);
            (
                StatusCode::OK,
                Json(ConfigReloadResponse::restart_required(
                    reloaded_fields,
                    restart_required_fields,
                    discovered_model_count,
                )),
            )
                .into_response()
        }
        ConfigReloadDecision::RestartWorker { .. } => {
            restart_worker(
                &application_state,
                candidate_resolved,
                reloaded_fields,
                discovered_model_count,
            )
            .await
        }
    }
}

async fn restart_worker(
    application_state: &ApplicationState,
    candidate_resolved: crate::ResolvedRuntimeConfig,
    reloaded_fields: Vec<String>,
    discovered_model_count: usize,
) -> Response {
    let Some(worker_handle) = application_state.worker_control.as_ref() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ConfigReloadResponse::failed(
                "Config reload cannot replace this application's worker".to_owned(),
                discovered_model_count,
            )),
        )
            .into_response();
    };
    let worker_restart_result = worker_handle
        .restart_worker_with_startup_configuration(
            candidate_resolved.worker_executable_path.clone(),
            Arc::clone(&candidate_resolved.model_directories),
            candidate_resolved.max_output_tokens,
            candidate_resolved.worker_startup_configuration(),
        )
        .await;
    // Process replacement only proves a child was launched. Wait for the child to report its
    // own resolved policy before replacing the supervisor's live configuration or telling the
    // menu that reload succeeded. This keeps candidate configuration from becoming observable
    // when startup, protocol initialization, or policy acknowledgement later fails.
    let worker_runtime_feature_configuration_result = match worker_restart_result {
        Ok(()) => {
            worker_handle
                .wait_for_worker_runtime_feature_configuration(Duration::from_secs(60))
                .await
        }
        Err(worker_restart_error) => Err(worker_restart_error),
    };
    match worker_runtime_feature_configuration_result {
        Ok(worker_runtime_feature_configuration) => {
            let Some(reloadable_config) = application_state.reloadable_config.as_ref() else {
                return internal_config_lock_error_response(discovered_model_count);
            };
            let Ok(mut live_config) = reloadable_config.write() else {
                return internal_config_lock_error_response(discovered_model_count);
            };
            *live_config = candidate_resolved;
            (
                StatusCode::OK,
                Json(ConfigReloadResponse::worker_restart_completed(
                    reloaded_fields,
                    discovered_model_count,
                    worker_runtime_feature_configuration,
                )),
            )
                .into_response()
        }
        Err(WorkerControlError::GenerationBusy) => {
            (StatusCode::CONFLICT, Json(ConfigReloadResponse::busy())).into_response()
        }
        Err(worker_restart_error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ConfigReloadResponse::failed(
                format!("Config was valid, but worker replacement failed: {worker_restart_error}"),
                discovered_model_count,
            )),
        )
            .into_response(),
    }
}

fn apply_in_place_reload_fields(
    live_config: &mut crate::ResolvedRuntimeConfig,
    candidate_resolved: &crate::ResolvedRuntimeConfig,
) {
    // Copy only fields that are safe to apply without a worker or REST restart.
    // Worker and listener settings remain live-old until the requested restart succeeds.
    live_config.config_warning = candidate_resolved.config_warning.clone();
    live_config.maximum_mlx_memory_bytes = candidate_resolved.maximum_mlx_memory_bytes;
}

fn internal_config_lock_error_response(discovered_model_count: usize) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ConfigReloadResponse::failed(
            "Internal config lock is unavailable".to_owned(),
            discovered_model_count,
        )),
    )
        .into_response()
}
