//! The live memory-limit path of the config reload endpoint: ceiling
//! validation, staging and per-outcome application. Split from config_reload
//! so each file stays within the repository's source-size limit.

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
/// `None` means the limit was applied and the reload continues;
/// `Some(response)` ends the reload.
pub(super) async fn stage_and_apply_memory_limit(
    application_state: &ApplicationState,
    candidate_resolved: &crate::ResolvedRuntimeConfig,
    candidate_generation: &str,
    memory_effective_generation: &str,
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
    worker_handle.stage_memory_configuration_generation(memory_effective_generation.to_owned());
    apply_live_memory_limit(
        application_state,
        worker_handle,
        effective_mlx_memory_ceiling_bytes,
        memory_effective_generation,
        candidate_generation,
        discovered_model_count,
    )
    .await
}

/// Applies the reloaded memory limit to the live worker and handles every
/// update outcome. `None` means the limit was applied and the reload
/// continues; `Some(response)` ends the reload.
pub(super) async fn apply_live_memory_limit(
    application_state: &ApplicationState,
    worker_handle: &crate::worker::handle::WorkerHandle,
    effective_mlx_memory_ceiling_bytes: u64,
    memory_effective_generation: &str,
    candidate_generation: &str,
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
        Ok(MlxMemoryLimitUpdateOutcome::Applied) => {
            worker_handle.record_memory_configuration_generation(
                memory_effective_generation.to_owned(),
                MlxMemoryLimitUpdateOutcome::Applied,
            );
            None
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
