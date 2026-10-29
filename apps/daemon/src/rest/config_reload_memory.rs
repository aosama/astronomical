//! The live memory-limit path of the config reload endpoint: ceiling
//! validation, staging, per-outcome application, and the restart-required
//! demotion. Split from config_reload so each file stays within the
//! repository's source-size limit.

use std::sync::Arc;

use crate::application::ApplicationState;
use crate::config::config_reload_response::ConfigReloadResponse;
use crate::{ChatGenerationExecutor, MlxMemoryLimitUpdateOutcome};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::config_reload::{effective_ceiling_violates_worker_limit, effective_worker_generation};

/// Validates the reloaded ceiling against the worker's reported limits,
/// stages the memory-effective generation, and applies the live limit.
/// `None` means the limit was applied (or queued) and the reload continues;
/// `Some(response)` ends the reload. Mirrors the inlined memory-limit block,
/// including the guard that a restart-required reload takes the demotion path
/// elsewhere instead.
#[allow(clippy::too_many_arguments)]
pub(super) async fn stage_and_apply_memory_limit(
    application_state: &ApplicationState,
    candidate_resolved: &crate::ResolvedRuntimeConfig,
    candidate_generation: &str,
    memory_effective_generation: &str,
    current_resolved: &crate::ResolvedRuntimeConfig,
    discovered_model_count: usize,
) -> Option<Response> {
    let worker_handle = application_state.worker_control.as_ref()?;
    let worker_health_snapshot = worker_handle.worker_health_snapshot();
    let effective_mlx_memory_ceiling_bytes = candidate_resolved
        .maximum_mlx_memory_bytes
        .unwrap_or(worker_health_snapshot.machine_mlx_memory_ceiling_bytes);
    if effective_ceiling_violates_worker_limit(
        effective_mlx_memory_ceiling_bytes,
        worker_health_snapshot.machine_mlx_memory_ceiling_bytes,
    ) {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(
                    ConfigReloadResponse::invalid_config(
                        "maximum_mlx_memory_gb is outside the worker's reported limits".to_owned(),
                    )
                    .with_generations(
                        candidate_generation,
                        effective_worker_generation(application_state),
                    ),
                ),
            )
                .into_response(),
        );
    }
    *application_state
        .pending_memory_config_generation
        .lock()
        .await = Some(memory_effective_generation.to_owned());
    worker_handle.stage_memory_configuration_generation(memory_effective_generation.to_owned());
    apply_live_memory_limit(
        application_state,
        worker_handle,
        effective_mlx_memory_ceiling_bytes,
        memory_effective_generation,
        candidate_resolved,
        candidate_generation,
        current_resolved,
        discovered_model_count,
    )
    .await
}

/// Applies the reloaded memory limit to the live worker and handles every
/// update outcome. `None` means the limit was applied (or queued for idle
/// application) and the reload continues; `Some(response)` ends the reload.
#[allow(clippy::too_many_arguments)]
pub(super) async fn apply_live_memory_limit(
    application_state: &ApplicationState,
    worker_handle: &crate::worker::handle::WorkerHandle,
    effective_mlx_memory_ceiling_bytes: u64,
    memory_effective_generation: &str,
    candidate_resolved: &crate::ResolvedRuntimeConfig,
    candidate_generation: &str,
    current_resolved: &crate::ResolvedRuntimeConfig,
    discovered_model_count: usize,
) -> Option<Response> {
    // A restart-required candidate is not wholly effective yet. Sending its full
    // generation would make the worker's next model acknowledgement disagree with
    // the daemon's memory-only live configuration.
    match worker_handle
        .update_mlx_memory_limit(
            effective_mlx_memory_ceiling_bytes,
            memory_effective_generation.to_owned(),
        )
        .await
    {
        Ok(
            update_outcome @ (MlxMemoryLimitUpdateOutcome::Applied
            | MlxMemoryLimitUpdateOutcome::Queued),
        ) => {
            worker_handle.record_memory_configuration_generation(
                memory_effective_generation.to_owned(),
                update_outcome,
            );
            if update_outcome == MlxMemoryLimitUpdateOutcome::Queued {
                tokio::spawn(
                    crate::config::queued_memory_reload::reconcile_reloaded_memory_config(
                        application_state.clone(),
                        memory_effective_generation.to_owned(),
                        current_resolved.clone(),
                    ),
                );
            } else {
                *application_state
                    .pending_memory_config_generation
                    .lock()
                    .await = None;
            }
            None
        }
        Ok(MlxMemoryLimitUpdateOutcome::RestartRequired) => {
            restart_worker_for_memory_limit_reload(
                application_state,
                worker_handle,
                memory_effective_generation,
                candidate_resolved,
                candidate_generation,
                discovered_model_count,
            )
            .await
        }
        Ok(MlxMemoryLimitUpdateOutcome::Rejected) => Some(
            (
                StatusCode::BAD_REQUEST,
                Json(
                    ConfigReloadResponse::invalid_config(
                        worker_handle
                            .worker_health_snapshot()
                            .mlx_memory_limit_error
                            .unwrap_or_else(|| "worker rejected the MLX memory limit".to_owned()),
                    )
                    .with_generations(
                        candidate_generation,
                        effective_worker_generation(application_state),
                    ),
                ),
            )
                .into_response(),
        ),
        Err(worker_control_error) => {
            *application_state
                .pending_memory_config_generation
                .lock()
                .await = None;
            worker_handle.record_memory_configuration_generation(
                memory_effective_generation.to_owned(),
                MlxMemoryLimitUpdateOutcome::Rejected,
            );
            tracing::error!(error = %worker_control_error, "MLX memory reload failed");
            Some(
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        ConfigReloadResponse::failed(
                            "Could not apply the MLX memory limit; inspect local diagnostics and retry"
                                .to_owned(),
                            discovered_model_count,
                        )
                        .with_generations(
                            candidate_generation,
                            effective_worker_generation(application_state),
                        ),
                    ),
                )
                    .into_response(),
            )
        }
    }
}

/// Handles a memory limit the live worker can only accept through a
/// stop-first restart. Mirrors the inlined `RestartRequired` outcome block.
#[allow(clippy::too_many_arguments)]
pub(super) async fn restart_worker_for_memory_limit_reload(
    application_state: &ApplicationState,
    worker_handle: &crate::worker::handle::WorkerHandle,
    memory_effective_generation: &str,
    candidate_resolved: &crate::ResolvedRuntimeConfig,
    candidate_generation: &str,
    discovered_model_count: usize,
) -> Option<Response> {
    // A reload candidate that lowers the ceiling below resident
    // capacity needs the same stop-first worker restart as the
    // dedicated memory endpoint. The candidate configuration is
    // already persisted, so its startup settings carry the lower
    // ceiling for the replacement worker.
    let demotion_result = match worker_handle.reserve_generation_admission_for_demotion() {
        Ok(demotion_admission) => {
            let demotion_result = worker_handle
                .restart_worker_for_memory_demotion(
                    candidate_resolved.worker_executable_path.clone(),
                    Arc::clone(&candidate_resolved.model_policy_catalog),
                    candidate_resolved.worker_startup_configuration(),
                )
                .await;
            drop(demotion_admission);
            demotion_result
        }
        Err(demotion_admission_error) => Err(demotion_admission_error),
    };
    match demotion_result {
        Ok(_) => {
            worker_handle.record_memory_configuration_generation(
                memory_effective_generation.to_owned(),
                MlxMemoryLimitUpdateOutcome::Applied,
            );
            *application_state
                .pending_memory_config_generation
                .lock()
                .await = None;
            None
        }
        Err(demotion_error) => {
            *application_state
                .pending_memory_config_generation
                .lock()
                .await = None;
            worker_handle.record_memory_configuration_generation(
                memory_effective_generation.to_owned(),
                MlxMemoryLimitUpdateOutcome::Rejected,
            );
            tracing::error!(error = %demotion_error, "MLX memory demotion restart failed during reload");
            Some(
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        ConfigReloadResponse::failed(
                            "The reloaded memory setting needs a worker restart, but the restart failed; inspect local diagnostics and retry".to_owned(),
                            discovered_model_count,
                        )
                        .with_generations(
                            candidate_generation,
                            effective_worker_generation(application_state),
                        ),
                    ),
                )
                    .into_response(),
            )
        }
    }
}
