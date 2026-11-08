//! Frees a loaded model's memory after the worker has served nothing for the
//! configured idle span (`runtime.idle_model_unload_seconds`).
//!
//! A resident model holds gigabytes of unified memory the rest of the Mac
//! needs, and MLX does not hand wired memory back reliably while its process
//! lives. The unload therefore replaces the worker process with an idle one
//! from the same trusted launch inputs, exactly as failure containment does;
//! the next request loads its model into the fresh worker as usual.
//!
//! The unload runs inside the worker loop, so any request that arrives while
//! it is under way waits in the command channel and is served by the fresh
//! worker instead of being refused.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::time::Instant;

use crate::worker::health::publish_health;
use crate::worker::loop_types::WorkerLoopRuntime;
use crate::worker::process::containment::replace_terminated_worker;
use crate::{WorkerHealthSnapshot, WorkerHealthStatus};

/// The setting's "never unload" value: zero keeps a model loaded until another
/// model is requested, rather than meaning a zero-second span.
const IDLE_UNLOAD_DISABLED_SENTINEL_SECONDS: u64 = 0;

/// The worker loop's idle clock and the live idle-unload setting it reads.
pub(crate) struct IdleModelUnloadClock {
    idle_model_unload_seconds: Arc<AtomicU64>,
    last_activity_at: Instant,
}

impl IdleModelUnloadClock {
    /// Starts the idle clock at `now`. The setting is shared with the worker
    /// handle so a configuration reload changes it without a worker restart.
    pub(crate) fn new(idle_model_unload_seconds: Arc<AtomicU64>, now: Instant) -> Self {
        Self {
            idle_model_unload_seconds,
            last_activity_at: now,
        }
    }

    /// Restarts the idle clock, as after a worker restart that may have
    /// loaded a model.
    pub(crate) fn record_activity(&mut self, now: Instant) {
        self.last_activity_at = now;
    }

    /// Restarts the idle clock at the end of every loop iteration that was
    /// serving a request, so the clock starts from the request's last event
    /// rather than from whenever the loop began waiting for it.
    pub(crate) fn observe_iteration_end(&mut self, was_serving_a_request: bool, now: Instant) {
        if was_serving_a_request {
            self.last_activity_at = now;
        }
    }

    /// Whether the worker has been unused for at least the configured span.
    pub(crate) fn is_unload_due(&self, now: Instant) -> bool {
        let idle_model_unload_seconds = self.idle_model_unload_seconds.load(Ordering::Relaxed);
        if idle_model_unload_seconds == IDLE_UNLOAD_DISABLED_SENTINEL_SECONDS {
            return false;
        }
        now.saturating_duration_since(self.last_activity_at)
            >= Duration::from_secs(idle_model_unload_seconds)
    }
}

/// What the worker loop knows about its own load at one idle check.
pub(crate) struct WorkerIdleness {
    pub(crate) has_loaded_model: bool,
    pub(crate) has_active_request: bool,
    pub(crate) free_active_generation_permits: usize,
    pub(crate) free_generation_queue_permits: usize,
}

impl WorkerIdleness {
    /// A request holding either semaphore is already on its way to the loop,
    /// so the worker counts as busy until both are back to full.
    pub(crate) fn is_free_to_unload(&self) -> bool {
        self.has_loaded_model
            && !self.has_active_request
            && self.free_active_generation_permits == crate::worker::handle::ACTIVE_GENERATION_SLOTS
            && self.free_generation_queue_permits == crate::worker::handle::GENERATION_QUEUE_DEPTH
    }
}

/// Whether the loop may unload now: a model is loaded, the configured span
/// has passed, and no request is active or waiting for admission.
pub(in crate::worker) fn should_unload_idle_model(
    runtime: &WorkerLoopRuntime,
    has_loaded_model: bool,
    now: Instant,
) -> bool {
    let worker_idleness = WorkerIdleness {
        has_loaded_model,
        has_active_request: runtime.active_generation.is_some(),
        free_active_generation_permits: runtime.active_generation_permits.available_permits(),
        free_generation_queue_permits: runtime.generation_queue_permits.available_permits(),
    };
    worker_idleness.is_free_to_unload() && runtime.idle_model_unload.is_unload_due(now)
}

/// Closes the worker holding the idle model and launches an empty one.
///
/// A worker that will not close gracefully is force-terminated; a relaunch
/// that fails leaves the worker `Unavailable`, which the contained-failure
/// relaunch then retries under its crash-loop budget.
pub(in crate::worker) async fn unload_idle_model(runtime: &mut WorkerLoopRuntime) {
    let unloaded_model_id = runtime
        .health_snapshot
        .read()
        .ok()
        .and_then(|worker_health_snapshot| worker_health_snapshot.ready_model_id.clone());
    tracing::info!(
        unloaded_model_id = ?unloaded_model_id,
        idle_seconds = runtime.idle_model_unload.idle_model_unload_seconds.load(Ordering::Relaxed),
        "unloading idle model by replacing the worker with an empty one"
    );
    // Loading is the closest published state: the worker is being replaced and
    // cannot serve until the empty one acknowledges, a second or two later.
    publish_health(
        &runtime.health_snapshot,
        WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading),
    );
    runtime.is_ready = false;
    if let Err(close_error) = runtime.worker_process.close().await {
        tracing::warn!(error = %close_error, "idle worker did not close gracefully; terminating it");
        let _termination_outcome = runtime.worker_process.force_terminate().await;
    }
    let relaunch_outcome = replace_terminated_worker(
        &mut runtime.worker_process,
        &runtime.health_snapshot,
        runtime.model_load_timeout,
    )
    .await;
    runtime.idle_model_unload.record_activity(Instant::now());
    match relaunch_outcome {
        Ok(()) => {
            runtime.is_ready = true;
            runtime.model_load_deadline = None;
            tracing::info!("idle model unloaded; an empty worker is ready");
        }
        Err(relaunch_error) => {
            tracing::error!(
                error = %relaunch_error,
                "could not relaunch the worker after unloading the idle model"
            );
        }
    }
}

#[cfg(test)]
#[path = "idle_model_unload_tests.rs"]
mod idle_model_unload_tests;
