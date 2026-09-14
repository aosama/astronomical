//! Idle-time pending-work draining for the supervisor worker loop.
//!
//! Before every select iteration the loop must apply an enacted MLX memory
//! limit change and a pending prompt-cache clear while no generation is
//! active; this module owns that pre-loop work so the loop itself stays
//! inside the source-size budget. Dependencies stay explicit parameters,
//! matching the process-loop style noted in `worker.rs`.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::Semaphore;
use tokio::time::Instant;

use crate::worker_cache_clear::apply_pending_prompt_cache_clear_if_idle;
use crate::worker_memory_limit::{
    PendingMlxMemoryLimitUpdate, apply_pending_mlx_memory_limit_if_idle,
    contain_mlx_memory_limit_failure,
};
use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, WorkerHealthSnapshot, WorkerProcess,
    worker_loop_types::ActiveWorkerRequest,
};

pub(super) async fn drain_pending_idle_work(
    pending_mlx_memory_limit_update: &mut Option<PendingMlxMemoryLimitUpdate>,
    pending_prompt_cache_clear: &mut Option<crate::PendingPromptCacheClear>,
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_generation: &mut Option<ActiveWorkerRequest>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    active_generation_permits: &Arc<Semaphore>,
    generation_queue_permits: &Arc<Semaphore>,
    model_load_timeout: Duration,
) {
    if active_generation.is_none()
        && let Err(memory_limit_error) = apply_pending_mlx_memory_limit_if_idle(
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
        contain_mlx_memory_limit_failure(
            worker_process,
            health_snapshot,
            active_generation,
            is_ready,
            memory_limit_error,
        )
        .await;
    }
    apply_pending_prompt_cache_clear_if_idle(
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
