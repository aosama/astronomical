//! Idle-time pending-work draining for the daemon worker loop.
//!
//! Before every select iteration the loop must apply an enacted MLX memory
//! limit change and a pending prompt-cache clear while no generation is
//! active; this module owns that pre-loop work so the loop itself stays
//! inside the source-size budget. Dependencies stay explicit parameters,
//! matching the process-loop style noted in `worker.rs`.

use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::worker::cache_clear;
use crate::worker::loop_types::ActiveWorkerRequest;
use crate::worker::loop_types::WorkerLoopState;
use crate::worker::memory::limit::{
    MlxMemoryLimitUpdateOutcome, PendingMlxMemoryLimitUpdate,
    apply_pending_mlx_memory_limit_if_idle, contain_mlx_memory_limit_failure,
};

pub(super) async fn drain_pending_idle_work(
    worker_loop_state: &mut WorkerLoopState<'_>,
    pending_mlx_memory_limit_update: &mut Option<PendingMlxMemoryLimitUpdate>,
    pending_prompt_cache_clear: &mut Option<crate::PendingPromptCacheClear>,
    active_generation_permits: &Arc<Semaphore>,
    generation_queue_permits: &Arc<Semaphore>,
) {
    let WorkerLoopState {
        worker_process,
        health_snapshot,
        active_request: active_generation,
        is_ready,
        model_load_deadline,
        performance_log,
        completion_log,
        model_load_timeout,
    } = worker_loop_state;
    let worker_process = &mut **worker_process;
    let health_snapshot = *health_snapshot;
    let is_ready = &mut **is_ready;
    let model_load_deadline = &mut **model_load_deadline;
    let active_generation = &mut **active_generation;
    let performance_log = &mut **performance_log;
    let completion_log = &mut **completion_log;
    let model_load_timeout = *model_load_timeout;
    if active_generation.is_none()
        && let drained_memory_limit_outcome = apply_pending_mlx_memory_limit_if_idle(
            pending_mlx_memory_limit_update,
            worker_process,
            model_load_timeout,
            health_snapshot,
            is_ready,
            model_load_deadline,
            active_generation,
            performance_log,
            completion_log,
        )
        .await
    {
        match drained_memory_limit_outcome {
            Err(memory_limit_error) => {
                contain_mlx_memory_limit_failure(
                    worker_process,
                    health_snapshot,
                    active_generation,
                    is_ready,
                    memory_limit_error,
                )
                .await;
            }
            Ok(Some(MlxMemoryLimitUpdateOutcome::RestartRequired)) => {
                // A queued ceiling the resident engine cannot adopt in place.
                // Health now carries the restart-required marker, and the
                // queued reconciliation owns the stop-first demotion; without
                // this trace the outcome used to disappear silently.
                tracing::info!(
                    "queued MLX memory update requires a streaming demotion restart; deferring to queued reconciliation"
                );
            }
            Ok(_) => {}
        }
    }
    cache_clear::apply_pending_prompt_cache_clear_if_idle(
        pending_prompt_cache_clear,
        worker_process,
        health_snapshot,
        active_generation,
        is_ready,
        model_load_deadline,
        performance_log,
        completion_log,
        active_generation_permits,
        generation_queue_permits,
    )
    .await;
}

/// Waits until the image request's earliest deadline (execution or progress stall).
pub(super) async fn wait_for_image_execution_deadline(
    active_request: &Option<ActiveWorkerRequest>,
) {
    let Some(ActiveWorkerRequest::Image(active_image)) = active_request else {
        std::future::pending::<()>().await;
        return;
    };
    let next_deadline = active_image
        .execution_deadline
        .min(active_image.progress_stall_deadline);
    tokio::time::sleep_until(next_deadline).await;
}

/// Waits until the embeddings request's execution deadline.
pub(super) async fn wait_for_embeddings_execution_deadline(
    active_request: &Option<ActiveWorkerRequest>,
) {
    let Some(ActiveWorkerRequest::Embeddings(active_embeddings)) = active_request else {
        std::future::pending::<()>().await;
        return;
    };
    tokio::time::sleep_until(active_embeddings.execution_deadline).await;
}
