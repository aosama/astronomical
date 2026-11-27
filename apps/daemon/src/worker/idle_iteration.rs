//! The idle-time steps of one worker loop iteration: draining pending work
//! before the loop waits, and what the periodic idle control tick does.
//!
//! They live apart from `command_arms::run_worker_iteration` so that the
//! `select!` there stays inside the function length budget.

use crate::worker::event_reaction::sample_idle_mlx_memory_telemetry;
use crate::worker::idle_drain::drain_pending_idle_work;
use crate::worker::idle_model_unload::{
    ModelUnloadCause, should_unload_idle_model, unload_loaded_model,
};
use crate::worker::loop_types::{WorkerLoopRuntime, WorkerLoopState};

/// What decides whether the idle control tick has anything to do.
pub(super) struct IdleControlTickConditions {
    pub is_worker_running: bool,
    pub is_ready: bool,
    pub is_serving_a_request: bool,
    pub has_loaded_model: bool,
    pub has_pending_astro_ssd_prompt_cache_clear: bool,
}

impl IdleControlTickConditions {
    /// The tick runs only on a running, ready worker that serves no request,
    /// and only while a model is loaded or an `astro_ssd_prompt_cache` clear
    /// is pending.
    pub(super) fn are_met(&self) -> bool {
        self.is_worker_running
            && self.is_ready
            && !self.is_serving_a_request
            && (self.has_loaded_model || self.has_pending_astro_ssd_prompt_cache_clear)
    }
}

/// Reads the loop's idle control tick conditions from its carried state.
pub(super) fn idle_control_tick_conditions(
    runtime: &WorkerLoopRuntime,
    is_worker_running: bool,
    has_loaded_model: bool,
) -> IdleControlTickConditions {
    IdleControlTickConditions {
        is_worker_running,
        is_ready: runtime.is_ready,
        is_serving_a_request: runtime.active_generation.is_some(),
        has_loaded_model,
        has_pending_astro_ssd_prompt_cache_clear: runtime
            .pending_astro_ssd_prompt_cache_clear
            .is_some(),
    }
}

/// Applies a pending `astro_ssd_prompt_cache` clear, when no generation is
/// active, before the loop waits for its next event.
pub(super) async fn drain_idle_work_before_waiting(runtime: &mut WorkerLoopRuntime) {
    let mut worker_loop_state = WorkerLoopState {
        worker_process: &mut runtime.worker_process,
        health_snapshot: &runtime.health_snapshot,
        is_ready: &mut runtime.is_ready,
        model_load_deadline: &mut runtime.model_load_deadline,
        active_request: &mut runtime.active_generation,
        performance_log: &mut runtime.performance_log,
        completion_log: &mut runtime.completion_log,
        model_load_timeout: runtime.model_load_timeout,
    };
    drain_pending_idle_work(
        &mut worker_loop_state,
        &mut runtime.pending_astro_ssd_prompt_cache_clear,
        &runtime.active_generation_permits,
        &runtime.generation_queue_permits,
    )
    .await;
}

/// Requests the idle MLX memory telemetry and unloads the loaded model once
/// the configured idle span has passed.
pub(super) async fn run_idle_control_tick(runtime: &mut WorkerLoopRuntime, has_loaded_model: bool) {
    sample_idle_mlx_memory_telemetry(runtime, has_loaded_model).await;
    if should_unload_idle_model(runtime, has_loaded_model, tokio::time::Instant::now()) {
        let _empty_worker_is_ready = unload_loaded_model(runtime, ModelUnloadCause::IdleSpan).await;
    }
}

#[cfg(test)]
#[path = "idle_iteration_tests.rs"]
mod idle_iteration_tests;
