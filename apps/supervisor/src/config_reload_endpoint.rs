//! Internal localhost `POST /v1/config/reload` endpoint.

use std::sync::Arc;

use crate::application::ApplicationState;
use crate::config_reload::{ConfigReloadDecision, ConfigReloadDiff};
use crate::config_reload_response::ConfigReloadResponse;
use crate::{
    ChatGenerationExecutor, MlxMemoryLimitUpdateOutcome, WorkerActivity, WorkerControlError,
};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

/// Reloads config without cancelling active or queued generation work.
pub(crate) async fn reload_config(State(application_state): State<ApplicationState>) -> Response {
    let Some(reloadable_config) = application_state.reloadable_config.as_ref() else {
        return (StatusCode::NOT_FOUND, "reload not supported").into_response();
    };
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.as_ref() else {
        return (StatusCode::NOT_FOUND, "reload not supported").into_response();
    };
    let _configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    if application_state
        .pending_memory_config_generation
        .lock()
        .await
        .is_some()
    {
        return (StatusCode::CONFLICT, Json(ConfigReloadResponse::busy())).into_response();
    }
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
            tracing::warn!(error = %config_error, "configuration reload validation failed");
            *application_state
                .configuration_validation_error
                .write()
                .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) = Some(
                "Configuration is invalid; correct the local configuration file and retry"
                    .to_owned(),
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(ConfigReloadResponse::invalid_config(
                    "Configuration is invalid; correct the local configuration file and retry"
                        .to_owned(),
                )),
            )
                .into_response();
        }
    };
    *application_state
        .configuration_validation_error
        .write()
        .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) = None;
    if let Some(configured_config_snapshot) = application_state.configured_config_snapshot.as_ref()
    {
        *configured_config_snapshot
            .write()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) =
            candidate_resolved.clone();
    }
    let candidate_generation = candidate_resolved.configuration_generation.clone();
    if let Some(discovery_diagnostic) = candidate_resolved.model_discovery_diagnostics.first() {
        let configured_root_numbers = discovery_diagnostic
            .configured_root_numbers
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        return (
            StatusCode::BAD_REQUEST,
            Json(
                ConfigReloadResponse::invalid_config(format!(
                    "Model '{}' appears in model_directories entries {}; remove one duplicate root and retry",
                    discovery_diagnostic.model_id, configured_root_numbers
                ))
                .with_generations(
                    &candidate_generation,
                    effective_worker_generation(&application_state),
                ),
            ),
        )
            .into_response();
    }
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
    let memory_effective_generation = if memory_limit_changed
        && matches!(
            &reload_decision,
            ConfigReloadDecision::RestApiRestartRequired { .. }
        ) {
        crate::ResolvedConfigurationGeneration::derive_memory_only_transition(
            &current_resolved.configuration_generation,
            candidate_resolved.maximum_mlx_memory_bytes,
        )
    } else {
        candidate_generation.clone()
    };
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
        return (
            StatusCode::CONFLICT,
            Json(ConfigReloadResponse::busy().with_generations(
                &candidate_generation,
                effective_worker_generation(&application_state),
            )),
        )
            .into_response();
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
                Json(
                    ConfigReloadResponse::invalid_config(
                        "maximum_mlx_memory_gb is outside the worker's reported limits".to_owned(),
                    )
                    .with_generations(
                        &candidate_generation,
                        effective_worker_generation(&application_state),
                    ),
                ),
            )
                .into_response();
        }
        *application_state
            .pending_memory_config_generation
            .lock()
            .await = Some(memory_effective_generation.clone());
        worker_handle.stage_memory_configuration_generation(memory_effective_generation.clone());
        match worker_handle
            .update_mlx_memory_limit(effective_mlx_memory_ceiling_bytes)
            .await
        {
            Ok(
                update_outcome @ (MlxMemoryLimitUpdateOutcome::Applied
                | MlxMemoryLimitUpdateOutcome::Queued),
            ) => {
                worker_handle.record_memory_configuration_generation(
                    memory_effective_generation.clone(),
                    update_outcome,
                );
                if update_outcome == MlxMemoryLimitUpdateOutcome::Queued {
                    tokio::spawn(
                        crate::queued_memory_reload::reconcile_reloaded_memory_config(
                            application_state.clone(),
                            memory_effective_generation.clone(),
                            current_resolved.clone(),
                        ),
                    );
                } else {
                    *application_state
                        .pending_memory_config_generation
                        .lock()
                        .await = None;
                }
            }
            Ok(MlxMemoryLimitUpdateOutcome::Rejected) => {
                *application_state
                    .pending_memory_config_generation
                    .lock()
                    .await = None;
                worker_handle.record_memory_configuration_generation(
                    memory_effective_generation.clone(),
                    MlxMemoryLimitUpdateOutcome::Rejected,
                );
                return (
                    StatusCode::BAD_REQUEST,
                    Json(
                        ConfigReloadResponse::invalid_config(
                            worker_handle
                                .worker_health_snapshot()
                                .mlx_memory_limit_error
                                .unwrap_or_else(|| {
                                    "worker rejected the MLX memory limit".to_owned()
                                }),
                        )
                        .with_generations(
                            &candidate_generation,
                            effective_worker_generation(&application_state),
                        ),
                    ),
                )
                    .into_response();
            }
            Err(worker_control_error) => {
                *application_state
                    .pending_memory_config_generation
                    .lock()
                    .await = None;
                worker_handle.record_memory_configuration_generation(
                    memory_effective_generation.clone(),
                    MlxMemoryLimitUpdateOutcome::Rejected,
                );
                tracing::error!(error = %worker_control_error, "MLX memory reload failed");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        ConfigReloadResponse::failed(
                            "Could not apply the MLX memory limit; inspect local diagnostics and retry"
                                .to_owned(),
                            discovered_model_count,
                        )
                        .with_generations(
                            &candidate_generation,
                            effective_worker_generation(&application_state),
                        ),
                    ),
                )
                    .into_response();
            }
        }
    }

    match reload_decision {
        ConfigReloadDecision::NoWorkerRestart { .. } => {
            let mut live_config = reloadable_config
                .write()
                .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner());
            *live_config = candidate_resolved;
            (
                StatusCode::OK,
                Json(
                    ConfigReloadResponse::reloaded(reloaded_fields, discovered_model_count)
                        .with_generations(
                            &candidate_generation,
                            effective_worker_generation(&application_state),
                        ),
                ),
            )
                .into_response()
        }
        ConfigReloadDecision::RestApiRestartRequired {
            restart_required_fields,
            ..
        } => {
            let mut live_config = reloadable_config
                .write()
                .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner());
            apply_in_place_reload_fields(
                &mut live_config,
                &candidate_resolved,
                memory_limit_changed.then_some(memory_effective_generation),
            );
            (
                StatusCode::OK,
                Json(
                    ConfigReloadResponse::restart_required(
                        reloaded_fields,
                        restart_required_fields,
                        discovered_model_count,
                    )
                    .with_generations(
                        &candidate_generation,
                        effective_worker_generation(&application_state),
                    ),
                ),
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
    let candidate_generation = candidate_resolved.configuration_generation.clone();
    let Some(worker_handle) = application_state.worker_control.as_ref() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                ConfigReloadResponse::failed(
                    "Config reload cannot replace this application's worker".to_owned(),
                    discovered_model_count,
                )
                .with_generations(
                    &candidate_generation,
                    effective_worker_generation(application_state),
                ),
            ),
        )
            .into_response();
    };
    let worker_runtime_feature_configuration_result = worker_handle
        .restart_worker_with_startup_configuration(
            candidate_resolved.worker_executable_path.clone(),
            Arc::clone(&candidate_resolved.model_policy_catalog),
            candidate_resolved.worker_startup_configuration(),
        )
        .await;
    match worker_runtime_feature_configuration_result {
        Ok(worker_runtime_feature_configuration) => {
            let Some(reloadable_config) = application_state.reloadable_config.as_ref() else {
                return internal_config_lock_error_response(discovered_model_count);
            };
            let mut live_config = reloadable_config
                .write()
                .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner());
            *live_config = candidate_resolved;
            (
                StatusCode::OK,
                Json(
                    ConfigReloadResponse::worker_restart_completed(
                        reloaded_fields,
                        discovered_model_count,
                        worker_runtime_feature_configuration,
                    )
                    .with_generations(&candidate_generation, Some(candidate_generation.clone())),
                ),
            )
                .into_response()
        }
        Err(WorkerControlError::GenerationBusy) => {
            (StatusCode::CONFLICT, Json(ConfigReloadResponse::busy())).into_response()
        }
        Err(worker_restart_error) => {
            tracing::error!(error = %worker_restart_error, "configuration worker replacement failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    ConfigReloadResponse::failed(
                        "Config was valid, but worker replacement failed; inspect local diagnostics and retry"
                            .to_owned(),
                        discovered_model_count,
                    )
                    .with_generations(
                        &candidate_generation,
                        effective_worker_generation(application_state),
                    ),
                ),
            )
                .into_response()
        }
    }
}

fn effective_worker_generation(application_state: &ApplicationState) -> Option<String> {
    application_state
        .generation_executor
        .worker_health_snapshot()
        .worker_runtime_feature_configuration
        .map(|configuration| configuration.configuration_generation)
}

fn apply_in_place_reload_fields(
    live_config: &mut crate::ResolvedRuntimeConfig,
    candidate_resolved: &crate::ResolvedRuntimeConfig,
    memory_effective_generation: Option<String>,
) {
    // Copy only fields that are safe to apply without a worker or REST restart.
    // Worker and listener settings remain live-old until the requested restart succeeds.
    live_config.maximum_mlx_memory_bytes = candidate_resolved.maximum_mlx_memory_bytes;
    if let Some(memory_effective_generation) = memory_effective_generation {
        live_config.configuration_generation = memory_effective_generation;
    }
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
