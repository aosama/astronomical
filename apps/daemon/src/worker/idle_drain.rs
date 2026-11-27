//! Idle-time pending-work draining for the daemon worker loop.
//!
//! Before every select iteration the loop must apply a pending
//! `astro_ssd_prompt_cache` clear while no generation is active; this module
//! owns that pre-loop work so the loop itself stays inside the source-size
//! budget. Dependencies stay explicit parameters,
//! matching the process-loop style noted in `worker.rs`.

use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::worker::astro_ssd_prompt_cache_clear;
use crate::worker::loop_types::ActiveWorkerRequest;
use crate::worker::loop_types::WorkerLoopState;

pub(super) async fn drain_pending_idle_work(
    worker_loop_state: &mut WorkerLoopState<'_>,
    pending_astro_ssd_prompt_cache_clear: &mut Option<crate::PendingAstroSsdPromptCacheClear>,
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
        model_load_timeout: _,
    } = worker_loop_state;
    let worker_process = &mut **worker_process;
    let health_snapshot = *health_snapshot;
    let is_ready = &mut **is_ready;
    let model_load_deadline = &mut **model_load_deadline;
    let active_generation = &mut **active_generation;
    let performance_log = &mut **performance_log;
    let completion_log = &mut **completion_log;
    astro_ssd_prompt_cache_clear::apply_pending_astro_ssd_prompt_cache_clear_if_idle(
        pending_astro_ssd_prompt_cache_clear,
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
async fn wait_for_image_execution_deadline(active_request: &Option<ActiveWorkerRequest>) {
    let Some(ActiveWorkerRequest::Image(active_image)) = active_request else {
        std::future::pending::<()>().await;
        return;
    };
    let next_deadline = active_image
        .execution_deadline
        .min(active_image.progress_stall_deadline);
    tokio::time::sleep_until(next_deadline).await;
}

/// Waits until an image or speech request's earliest deadline (execution or
/// progress stall); never resolves for any other request.
pub(super) async fn wait_for_stepped_request_deadline(
    active_request: &Option<ActiveWorkerRequest>,
) {
    if matches!(active_request, Some(ActiveWorkerRequest::Speech(_))) {
        wait_for_speech_execution_deadline(active_request).await;
        return;
    }
    wait_for_image_execution_deadline(active_request).await;
}

/// Waits until the speech request's earliest deadline (execution or progress
/// stall); never resolves for any other request.
async fn wait_for_speech_execution_deadline(active_request: &Option<ActiveWorkerRequest>) {
    let Some(ActiveWorkerRequest::Speech(active_speech)) = active_request else {
        std::future::pending::<()>().await;
        return;
    };
    // A request whose audio ran past its bound ends now rather than after
    // the worker has spoken every remaining segment.
    if active_speech.has_exceeded_audio_limit {
        return;
    }
    let next_deadline = active_speech
        .execution_deadline
        .min(active_speech.progress_stall_deadline);
    tokio::time::sleep_until(next_deadline).await;
}

/// Waits until the execution deadline of a request that runs as one forward
/// pass (embeddings or decisions); never resolves for any other request.
pub(super) async fn wait_for_single_shot_execution_deadline(
    active_request: &Option<ActiveWorkerRequest>,
) {
    let execution_deadline = match active_request {
        Some(single_shot_request) => single_shot_request.single_shot_execution_deadline(),
        None => None,
    };
    let Some(execution_deadline) = execution_deadline else {
        std::future::pending::<()>().await;
        return;
    };
    tokio::time::sleep_until(execution_deadline).await;
}
