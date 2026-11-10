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

/// How long a Neural Engine preparation may go without finishing another
/// program before the idle clock stops waiting for it.
///
/// One program compiles in seconds, or about a minute for the largest models
/// on a first run, so ten minutes without progress means the preparation is
/// stuck. Waiting for it any longer would keep gigabytes of model loaded for
/// good, which is the very thing the idle unload exists to prevent.
const STALLED_PREPARATION_HOLD: Duration = Duration::from_secs(10 * 60);

/// How far a Neural Engine preparation has come, as the idle clock sees it.
/// Any change between two idle checks counts as progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NeuralEnginePreparationProgress {
    pub(crate) completed_program_count: u32,
    pub(crate) is_waiting_to_measure: bool,
}

impl NeuralEnginePreparationProgress {
    /// The progress a published status describes, or `None` when the status
    /// is not a preparation under way.
    pub(crate) fn from_status(
        neural_engine_prefill: Option<&astronomical_ipc_protocol::NeuralEnginePrefillStatus>,
    ) -> Option<Self> {
        match neural_engine_prefill {
            Some(astronomical_ipc_protocol::NeuralEnginePrefillStatus::Preparing {
                completed_program_count,
                is_waiting_to_measure,
                ..
            }) => Some(Self {
                completed_program_count: *completed_program_count,
                is_waiting_to_measure: *is_waiting_to_measure,
            }),
            _ => None,
        }
    }
}

/// The worker loop's idle clock and the live idle-unload setting it reads.
pub(crate) struct IdleModelUnloadClock {
    idle_model_unload_seconds: Arc<AtomicU64>,
    last_activity_at: Instant,
    /// The preparation progress seen at the last idle check and when it last
    /// changed, so a preparation that stops advancing stops holding the model.
    last_preparation_progress: Option<NeuralEnginePreparationProgress>,
    preparation_progress_at: Instant,
    /// The same restart on the idle clock's own millisecond scale, shared with
    /// the worker handle so `/v1/status` can say when the model will unload.
    last_activity_clock_millis: Arc<AtomicU64>,
}

impl IdleModelUnloadClock {
    /// Starts the idle clock at `now`. The setting is shared with the worker
    /// handle so a configuration reload changes it without a worker restart.
    pub(crate) fn new(
        idle_model_unload_seconds: Arc<AtomicU64>,
        last_activity_clock_millis: Arc<AtomicU64>,
        now: Instant,
    ) -> Self {
        let mut idle_clock = Self {
            idle_model_unload_seconds,
            last_activity_at: now,
            last_preparation_progress: None,
            preparation_progress_at: now,
            last_activity_clock_millis,
        };
        idle_clock.record_activity(now);
        idle_clock
    }

    /// Restarts the idle clock, as after a worker restart that may have
    /// loaded a model.
    pub(crate) fn record_activity(&mut self, now: Instant) {
        self.last_activity_at = now;
        self.last_activity_clock_millis
            .store(idle_clock_millis(now), Ordering::Relaxed);
    }

    /// Restarts the idle clock at the end of every loop iteration that was
    /// serving a request, so the clock starts from the request's last event
    /// rather than from whenever the loop began waiting for it.
    pub(crate) fn observe_iteration_end(&mut self, was_serving_a_request: bool, now: Instant) {
        if was_serving_a_request {
            self.record_activity(now);
        }
    }

    /// Restarts the idle clock while the worker prepares Neural Engine
    /// programs beside a served model, as long as the preparation advances.
    ///
    /// Preparation runs in the background for about a minute after the last
    /// request, which is the default idle span. Counted as idleness, it had
    /// the model unloaded the moment its programs were ready, so the next
    /// message loaded the model and prepared them all over again and the
    /// Neural Engine never engaged. The span therefore starts when
    /// preparation ends. A preparation that has made no progress for
    /// `STALLED_PREPARATION_HOLD` no longer restarts the clock, so a stuck
    /// one cannot keep a model loaded for good.
    pub(crate) fn observe_background_preparation(
        &mut self,
        preparation_progress: Option<NeuralEnginePreparationProgress>,
        now: Instant,
    ) {
        let Some(preparation_progress) = preparation_progress else {
            self.last_preparation_progress = None;
            return;
        };
        if self.last_preparation_progress != Some(preparation_progress) {
            self.last_preparation_progress = Some(preparation_progress);
            self.preparation_progress_at = now;
        }
        if now.saturating_duration_since(self.preparation_progress_at) < STALLED_PREPARATION_HOLD {
            self.record_activity(now);
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

/// The start of the idle clock's millisecond scale, fixed on first use.
static IDLE_CLOCK_EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

/// Milliseconds on the same monotonic clock the unload decision uses, so the
/// countdown agrees with it after the Mac sleeps or the wall clock is set.
/// One is added so zero keeps meaning "the clock never ran".
pub(crate) fn idle_clock_millis(at: Instant) -> u64 {
    let idle_clock_epoch = IDLE_CLOCK_EPOCH.get_or_init(std::time::Instant::now);
    let since_epoch_millis = at
        .into_std()
        .saturating_duration_since(*idle_clock_epoch)
        .as_millis();
    u64::try_from(since_epoch_millis)
        .unwrap_or(u64::MAX)
        .saturating_add(1)
}

/// Whole seconds left before an idle model unloads, rounded up, so 0 means the
/// span has passed and the unload is due at the next idle check. `None` when
/// unloading is off (a span of zero) or the clock has never run.
pub(crate) fn idle_unload_remaining_seconds(
    idle_model_unload_seconds: u64,
    last_activity_clock_millis: u64,
    now_clock_millis: u64,
) -> Option<u64> {
    if idle_model_unload_seconds == IDLE_UNLOAD_DISABLED_SENTINEL_SECONDS
        || last_activity_clock_millis == 0
    {
        return None;
    }
    let idle_millis = now_clock_millis.saturating_sub(last_activity_clock_millis);
    let remaining_millis = idle_model_unload_seconds
        .saturating_mul(1000)
        .saturating_sub(idle_millis);
    Some(remaining_millis.div_ceil(1000))
}

/// What `/v1/status` reports about the idle unload of the loaded model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IdleModelUnloadCountdown {
    /// The configured span; zero means a model stays loaded.
    pub(crate) after_seconds: u64,
    /// Seconds until the loaded model unloads, when unloading is on.
    pub(crate) remaining_seconds: Option<u64>,
}

impl IdleModelUnloadCountdown {
    pub(crate) fn read(
        idle_model_unload_seconds: &AtomicU64,
        last_activity_clock_millis: &AtomicU64,
    ) -> Self {
        let after_seconds = idle_model_unload_seconds.load(Ordering::Relaxed);
        Self {
            after_seconds,
            remaining_seconds: idle_unload_remaining_seconds(
                after_seconds,
                last_activity_clock_millis.load(Ordering::Relaxed),
                idle_clock_millis(Instant::now()),
            ),
        }
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

/// The Neural Engine preparation the published worker health reports, if one
/// is under way. A poisoned snapshot answers none, so a fault elsewhere never
/// keeps a model loaded.
fn published_preparation_progress(
    runtime: &WorkerLoopRuntime,
) -> Option<NeuralEnginePreparationProgress> {
    let worker_health_snapshot = runtime.health_snapshot.read().ok()?;
    NeuralEnginePreparationProgress::from_status(
        worker_health_snapshot.neural_engine_prefill.as_ref(),
    )
}

/// Whether the loop may unload now: a model is loaded, the configured span
/// has passed since the last request or the end of Neural Engine
/// preparation, and no request is active or waiting for admission.
pub(in crate::worker) fn should_unload_idle_model(
    runtime: &mut WorkerLoopRuntime,
    has_loaded_model: bool,
    now: Instant,
) -> bool {
    let preparation_progress = published_preparation_progress(runtime);
    runtime
        .idle_model_unload
        .observe_background_preparation(preparation_progress, now);
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
