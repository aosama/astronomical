//! The main worker event loop: receives commands and events, coordinates
//! generation lifecycle, model swaps, memory limits, and idle telemetry.
//!
//! The loop owns one `WorkerProcess` and one optional `ActiveGeneration`.
//! Commands arrive through `WorkerLoopCommand`; events arrive through the
//! worker's IPC stream. Generation-scoped events must match the active
//! request or are treated as protocol violations.

pub(crate) mod generation;
pub(crate) mod memory;
pub(crate) mod process;

pub(crate) mod cache_clear;
pub(crate) mod control_error;
pub(crate) mod embeddings_event;
pub(crate) mod embeddings_request;
pub(crate) mod event_handler;
pub(crate) mod handle;
pub(crate) mod health;
pub(crate) mod idle_drain;
pub(crate) mod image_event;
pub(crate) mod image_request;
pub(crate) mod loop_types;

pub(crate) mod command_arms;
pub(crate) mod event_reaction;

#[cfg(test)]
mod command_arms_tests;
#[cfg(test)]
mod event_handler_tests;
#[cfg(test)]
mod event_reaction_tests;
#[cfg(test)]
mod loop_types_tests;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::{Semaphore, mpsc};
use tokio::time::{Instant, MissedTickBehavior, interval};

use crate::worker::command_arms::{WorkerLoopFlow, run_worker_iteration};
use crate::worker::health::publish_health;
use crate::worker::loop_types::{WorkerLoopCommand, WorkerLoopRuntime};
use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, ImageGenerationTimeouts,
    RuntimeModelPolicy, WorkerHealthSnapshot, WorkerHealthStatus, WorkerProcess,
};

// run_worker is now the imperative shell: it publishes the initial health
// state, moves the loop-carried locals into one owned WorkerLoopRuntime, and
// drives run_worker_iteration until an iteration reports Exit. All per-command
// glue lives in command_arms; the non-command branches live in event_reaction.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_worker(
    worker_process: WorkerProcess,
    mut command_receiver: mpsc::Receiver<WorkerLoopCommand>,
    health_snapshot: Arc<RwLock<WorkerHealthSnapshot>>,
    model_load_timeout: Duration,
    cancellation_acknowledgement_timeout: Duration,
    image_generation_timeouts: ImageGenerationTimeouts,
    performance_log: GenerationPerformanceLog,
    completion_log: CompletionAttributionLog,
    model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
    active_generation_permits: Arc<Semaphore>,
    generation_queue_permits: Arc<Semaphore>,
) {
    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading),
    );
    let mut runtime = WorkerLoopRuntime {
        worker_process,
        health_snapshot,
        is_ready: false,
        model_load_deadline: Some(Instant::now() + model_load_timeout),
        active_generation: None,
        performance_log,
        completion_log,
        model_policy_catalog,
        pending_mlx_memory_limit_update: None,
        pending_prompt_cache_clear: None,
        active_generation_permits,
        generation_queue_permits,
        model_load_timeout,
        cancellation_acknowledgement_timeout,
        image_generation_timeouts,
    };
    let mut idle_control_interval = interval(Duration::from_secs(1));
    idle_control_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        let iteration_flow = run_worker_iteration(
            &mut runtime,
            &mut command_receiver,
            &mut idle_control_interval,
        )
        .await;
        if iteration_flow == WorkerLoopFlow::Exit {
            return;
        }
    }
}
