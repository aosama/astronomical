//! Deferred stop-first demotion for a queued MLX ceiling whose live
//! application reported a restart-required outcome.
//!
//! A ceiling below the loaded resident model cannot be applied while a
//! generation is active, so the endpoint queues it and reconciliation watches
//! worker health. When the worker later reports that only a streaming
//! replacement can reach the persisted ceiling, this module performs the same
//! stop-first restart the endpoint uses for idle applies: wait for an idle
//! window, reserve generation admission, replace the worker process, and let
//! the demotion lifecycle restore the previously selected model.

use std::{sync::Arc, time::Duration};

use tokio::time::{Instant, sleep};

use crate::{MlxMemoryLimitUpdateOutcome, ResolvedRuntimeConfig, WorkerControlError, WorkerHandle};

/// Poll cadence while waiting for the active generation to finalize.
const IDLE_WINDOW_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// Progress cadence while waiting for the active generation to finalize.
const IDLE_WINDOW_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// Replaces the worker with a streaming-capable process under the persisted
/// lower ceiling, then attributes the applied generation to the worker record.
///
/// The caller keeps ownership of configuration snapshots and the pending
/// memory-generation guard; this function only owns the worker lifecycle.
pub(crate) async fn demote_queued_restart_required_ceiling(
    worker_handle: &WorkerHandle,
    candidate_resolved_config: &ResolvedRuntimeConfig,
) -> Result<(), WorkerControlError> {
    let demotion_started_at = Instant::now();
    let requested_ceiling_bytes = candidate_resolved_config
        .maximum_mlx_memory_bytes
        .unwrap_or_default();
    tracing::info!(
        requested_ceiling_bytes,
        "queued reconciliation detected a restart-required MLX ceiling; waiting for an idle generation window"
    );
    let mut last_progress_logged_at = Instant::now();
    loop {
        match worker_handle.reserve_generation_admission_for_demotion() {
            Ok(demotion_admission) => {
                let demotion_result =
                    perform_stop_first_demotion(worker_handle, candidate_resolved_config).await;
                drop(demotion_admission);
                match demotion_result {
                    Ok(()) => {
                        tracing::info!(
                            total_elapsed_millis = demotion_started_at.elapsed().as_millis(),
                            "queued reconciliation committed the stop-first memory demotion"
                        );
                        worker_handle.record_memory_configuration_generation(
                            candidate_resolved_config.configuration_generation.clone(),
                            MlxMemoryLimitUpdateOutcome::Applied,
                        );
                        return Ok(());
                    }
                    Err(demotion_error) => {
                        tracing::error!(
                            error = %demotion_error,
                            total_elapsed_millis = demotion_started_at.elapsed().as_millis(),
                            "queued reconciliation failed its stop-first memory demotion"
                        );
                        worker_handle.record_memory_configuration_generation(
                            candidate_resolved_config.configuration_generation.clone(),
                            MlxMemoryLimitUpdateOutcome::Rejected,
                        );
                        return Err(demotion_error);
                    }
                }
            }
            Err(WorkerControlError::GenerationBusy) => {
                if last_progress_logged_at.elapsed() >= IDLE_WINDOW_PROGRESS_INTERVAL {
                    tracing::info!(
                        waited_millis = demotion_started_at.elapsed().as_millis(),
                        "queued memory demotion is still waiting for the active generation to finalize"
                    );
                    last_progress_logged_at = Instant::now();
                }
                sleep(IDLE_WINDOW_POLL_INTERVAL).await;
            }
            Err(reservation_error) => {
                tracing::error!(
                    error = %reservation_error,
                    "queued memory demotion could not reserve generation admission"
                );
                return Err(reservation_error);
            }
        }
    }
}

async fn perform_stop_first_demotion(
    worker_handle: &WorkerHandle,
    candidate_resolved_config: &ResolvedRuntimeConfig,
) -> Result<(), WorkerControlError> {
    let mut candidate_startup_configuration =
        candidate_resolved_config.worker_startup_configuration();
    candidate_startup_configuration.configured_maximum_mlx_memory_bytes =
        candidate_resolved_config.maximum_mlx_memory_bytes;
    worker_handle
        .restart_worker_for_memory_demotion(
            candidate_resolved_config.worker_executable_path.clone(),
            Arc::clone(&candidate_resolved_config.model_policy_catalog),
            candidate_startup_configuration,
        )
        .await
        .map(|_| ())
}
