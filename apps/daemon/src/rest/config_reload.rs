//! Internal localhost `POST /v1/config/reload` endpoint.

use std::sync::Arc;

use crate::application::ApplicationState;
use crate::config::config_reload::{
    ConfigReloadDecision, ConfigReloadDiff, ResolvedRuntimeConfigResolver,
};
use crate::config::config_reload_response::ConfigReloadResponse;
use crate::{WorkerActivity, WorkerControlError};
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
    // Held for the whole reload: no second reload or memory transition may
    // interleave with this one's decision sequence.
    let _configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    if let Some(busy_response) = reload_in_progress_response(&application_state).await {
        return busy_response;
    }

    let candidate_resolved =
        match load_candidate_configuration(runtime_config_resolver, &application_state) {
            Ok(candidate_resolved) => candidate_resolved,
            Err(config_error_response) => return *config_error_response,
        };
    let candidate_generation = candidate_resolved.configuration_generation.clone();
    if let Some(ambiguous_identity_response) = ambiguous_model_identity_response(
        &candidate_resolved,
        &candidate_generation,
        &application_state,
    ) {
        return ambiguous_identity_response;
    }
    let current_resolved = match reloadable_config.read() {
        Ok(current_resolved) => current_resolved.clone(),
        Err(_) => return internal_config_lock_error_response(0),
    };
    let reload_decision = ConfigReloadDiff::compare(&current_resolved, &candidate_resolved);
    let discovered_model_count = reload_decision.discovered_model_count();
    let reloaded_fields = reload_decision.reloaded_fields().to_vec();
    let memory_limit_changed =
        current_resolved.maximum_mlx_memory_bytes != candidate_resolved.maximum_mlx_memory_bytes;
    let memory_effective_generation = memory_effective_generation_for_reload(
        memory_limit_changed,
        &reload_decision,
        &current_resolved.configuration_generation,
        candidate_resolved.maximum_mlx_memory_bytes,
        &candidate_generation,
    );
    if generation_is_busy(&application_state) && reload_decision.worker_restart_required() {
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
        && let Some(memory_failure_response) =
            super::config_reload_memory::stage_and_apply_memory_limit(
                &application_state,
                &candidate_resolved,
                &candidate_generation,
                &memory_effective_generation,
                &current_resolved,
                discovered_model_count,
            )
            .await
    {
        return memory_failure_response;
    }

    respond_to_reload_decision(
        &application_state,
        reloadable_config,
        reload_decision,
        candidate_resolved,
        candidate_generation,
        reloaded_fields,
        memory_limit_changed,
        memory_effective_generation,
        discovered_model_count,
    )
    .await
}

/// The busy rejections that apply before any candidate is loaded: a pending
/// memory transition, or (when no worker handle exists) live generation
/// activity. Mirrors the inlined precondition checks.
async fn reload_in_progress_response(application_state: &ApplicationState) -> Option<Response> {
    if application_state
        .pending_memory_config_generation
        .lock()
        .await
        .is_some()
    {
        return Some((StatusCode::CONFLICT, Json(ConfigReloadResponse::busy())).into_response());
    }
    if application_state.worker_control.is_none()
        && application_state
            .generation_executor
            .worker_health_snapshot()
            .activity
            != WorkerActivity::Idle
    {
        return Some((StatusCode::CONFLICT, Json(ConfigReloadResponse::busy())).into_response());
    }
    None
}

/// Loads and validates the candidate configuration. On failure, records the
/// validation error for the status surface and builds the 400 response;
/// on success, clears any stale validation error and mirrors the candidate
/// into the configured snapshot. Mirrors the inlined load block.
fn load_candidate_configuration(
    runtime_config_resolver: &ResolvedRuntimeConfigResolver,
    application_state: &ApplicationState,
) -> Result<crate::ResolvedRuntimeConfig, Box<Response>> {
    let candidate_resolved =
        match runtime_config_resolver.load() {
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
                return Err(Box::new((
                StatusCode::BAD_REQUEST,
                Json(ConfigReloadResponse::invalid_config(
                    "Configuration is invalid; correct the local configuration file and retry"
                        .to_owned(),
                )),
            )
                .into_response()));
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
    Ok(candidate_resolved)
}

/// Builds the 400 response for an ambiguous model identity in the candidate's
/// discovery diagnostics, or `None` when the candidate is unambiguous.
/// Mirrors the inlined diagnostic scan.
fn ambiguous_model_identity_response(
    candidate_resolved: &crate::ResolvedRuntimeConfig,
    candidate_generation: &str,
    application_state: &ApplicationState,
) -> Option<Response> {
    let discovery_diagnostic =
        candidate_resolved
            .model_discovery_diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.code
                    == astronomical_config::ModelDiscoveryDiagnosticCode::AmbiguousModelIdentity
            })?;
    let configured_root_numbers = discovery_diagnostic
        .configured_root_numbers
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    Some(
        (
            StatusCode::BAD_REQUEST,
            Json(
                ConfigReloadResponse::invalid_config(format!(
                    "Model '{}' appears in model_directories entries {}; remove one duplicate root and retry",
                    discovery_diagnostic.model_id, configured_root_numbers
                ))
                .with_generations(
                    candidate_generation,
                    effective_worker_generation(application_state),
                ),
            ),
        )
            .into_response(),
    )
}

/// The generation the reloaded memory limit takes effect under: a memory-only
/// transition when the limit change rides along a REST-restart-required
/// reload, and the plain candidate generation otherwise.
pub(super) fn memory_effective_generation_for_reload(
    memory_limit_changed: bool,
    reload_decision: &ConfigReloadDecision,
    current_configuration_generation: &str,
    candidate_maximum_mlx_memory_bytes: Option<u64>,
    candidate_generation: &str,
) -> String {
    if memory_limit_changed
        && matches!(
            reload_decision,
            ConfigReloadDecision::RestApiRestartRequired { .. }
        )
    {
        crate::config::resolved_configuration_generation::ResolvedConfigurationGeneration::derive_memory_only_transition(
            current_configuration_generation,
            candidate_maximum_mlx_memory_bytes,
        )
    } else {
        candidate_generation.to_owned()
    }
}

/// A reload that needs a worker restart is rejected while a generation is
/// busy: the restart cannot proceed under live work.
fn generation_is_busy(application_state: &ApplicationState) -> bool {
    application_state.worker_control.as_ref().map_or_else(
        || {
            application_state
                .generation_executor
                .worker_health_snapshot()
                .activity
                != WorkerActivity::Idle
        },
        |worker_handle| !worker_handle.is_generation_idle_for_control_action(),
    )
}

/// The live ceiling the worker can accept: zero is no limit at all (the
/// caller meant "unrestricted" but the worker treats 0 as unset) and anything
/// above the machine's reported ceiling is not allocatable.
pub(super) fn effective_ceiling_violates_worker_limit(
    effective_ceiling_bytes: u64,
    machine_ceiling_bytes: u64,
) -> bool {
    effective_ceiling_bytes == 0 || effective_ceiling_bytes > machine_ceiling_bytes
}

/// Writes the reload outcome under the reload decision and builds the final
/// response: an in-place reload, a REST-restart notice, or a worker restart.
#[allow(clippy::too_many_arguments)]
async fn respond_to_reload_decision(
    application_state: &ApplicationState,
    reloadable_config: &std::sync::RwLock<crate::ResolvedRuntimeConfig>,
    reload_decision: ConfigReloadDecision,
    candidate_resolved: crate::ResolvedRuntimeConfig,
    candidate_generation: String,
    reloaded_fields: Vec<String>,
    memory_limit_changed: bool,
    memory_effective_generation: String,
    discovered_model_count: usize,
) -> Response {
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
                            effective_worker_generation(application_state),
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
                        effective_worker_generation(application_state),
                    ),
                ),
            )
                .into_response()
        }
        ConfigReloadDecision::RestartWorker { .. } => {
            restart_worker(
                application_state,
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

pub(super) fn effective_worker_generation(application_state: &ApplicationState) -> Option<String> {
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

pub(super) fn internal_config_lock_error_response(discovered_model_count: usize) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ConfigReloadResponse::failed(
            "Internal config lock is unavailable".to_owned(),
            discovered_model_count,
        )),
    )
        .into_response()
}
