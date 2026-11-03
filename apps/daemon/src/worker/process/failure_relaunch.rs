//! Brings an idle replacement worker back after failure containment.
//!
//! Containment terminates a worker that died or broke protocol and publishes
//! `Unavailable`. Without a relaunch, every later request is rejected with
//! `worker_unavailable` until the daemon restarts. The worker loop therefore
//! watches for a stopped, unavailable worker and relaunches it from its
//! trusted launch inputs, through the same relaunch-and-acknowledge path the
//! cancellation failure uses, bounded by a crash-loop budget.
//!
//! The rule is keyed on observable state rather than on each of the many
//! containment call sites, so every path that leaves the loop with a stopped,
//! unavailable worker recovers the same way and none can forget to.

use std::time::Duration;

use tokio::time::Instant;

use crate::worker::health::publish_health;
use crate::worker::loop_types::WorkerLoopRuntime;
use crate::worker::process::containment::replace_terminated_worker;
use crate::worker::process::crash_loop_guard::{CrashLoopGuard, RelaunchAdmission};
use crate::{WorkerHealthSnapshot, WorkerHealthStatus};

/// Three relaunches inside ten minutes is far beyond a one-off crash, while a
/// worker that dies on every start burns through it within seconds of backoff.
const MAXIMUM_CONTAINED_FAILURE_RELAUNCHES_PER_WINDOW: usize = 3;
const CONTAINED_FAILURE_RELAUNCH_WINDOW: Duration = Duration::from_secs(10 * 60);
/// A short pause before each relaunch keeps a crashing worker from spinning
/// the loop and gives the operating system time to reclaim the dead worker's
/// unified memory before the replacement maps its own.
const CONTAINED_FAILURE_RELAUNCH_BACKOFF: Duration = Duration::from_secs(1);

/// The worker loop's relaunch schedule after failure containment.
pub(crate) struct ContainedFailureRelaunch {
    crash_loop_guard: CrashLoopGuard,
    relaunch_backoff: Duration,
    relaunch_deadline: Option<Instant>,
}

impl ContainedFailureRelaunch {
    pub(crate) fn new() -> Self {
        Self::with_budget(
            CrashLoopGuard::new(
                MAXIMUM_CONTAINED_FAILURE_RELAUNCHES_PER_WINDOW,
                CONTAINED_FAILURE_RELAUNCH_WINDOW,
            ),
            CONTAINED_FAILURE_RELAUNCH_BACKOFF,
        )
    }

    pub(crate) fn with_budget(
        crash_loop_guard: CrashLoopGuard,
        relaunch_backoff: Duration,
    ) -> Self {
        Self {
            crash_loop_guard,
            relaunch_backoff,
            relaunch_deadline: None,
        }
    }

    /// When the scheduled relaunch is due, if one is scheduled.
    pub(crate) fn relaunch_deadline(&self) -> Option<Instant> {
        self.relaunch_deadline
    }

    /// Schedules a relaunch when the worker is stopped and unavailable and the
    /// crash-loop budget allows it. Called once per loop iteration.
    pub(crate) fn observe_worker(
        &mut self,
        is_worker_running: bool,
        worker_health_status: Option<WorkerHealthStatus>,
        now: Instant,
    ) {
        if is_worker_running {
            self.crash_loop_guard.observe_running_worker();
            self.relaunch_deadline = None;
            return;
        }
        if self.relaunch_deadline.is_some()
            || worker_health_status != Some(WorkerHealthStatus::Unavailable)
        {
            return;
        }
        match self.crash_loop_guard.admit_relaunch(now) {
            RelaunchAdmission::Admitted {
                relaunch_attempt_in_window,
            } => {
                tracing::warn!(
                    relaunch_attempt_in_window,
                    maximum_relaunches_per_window = MAXIMUM_CONTAINED_FAILURE_RELAUNCHES_PER_WINDOW,
                    relaunch_backoff_millis = self.relaunch_backoff.as_millis(),
                    "scheduling replacement worker after failure containment"
                );
                self.relaunch_deadline = Some(now + self.relaunch_backoff);
            }
            RelaunchAdmission::Exhausted { should_report } => {
                if should_report {
                    report_crash_loop_exhaustion();
                }
            }
        }
    }
}

fn report_crash_loop_exhaustion() {
    tracing::error!(
        maximum_relaunches_per_window = MAXIMUM_CONTAINED_FAILURE_RELAUNCHES_PER_WINDOW,
        relaunch_window_seconds = CONTAINED_FAILURE_RELAUNCH_WINDOW.as_secs(),
        "worker crash loop detected; no further relaunch, worker stays unavailable until \
         the app or daemon restarts it"
    );
}

impl Default for ContainedFailureRelaunch {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs the scheduled relaunch. On success the replacement worker is ready
/// (idle, or with the model it reported); on failure it stays `Unavailable`
/// and the next loop iteration asks the crash-loop budget for another try.
pub(in crate::worker) async fn relaunch_contained_worker(runtime: &mut WorkerLoopRuntime) {
    runtime.contained_failure_relaunch.relaunch_deadline = None;
    publish_health(
        &runtime.health_snapshot,
        WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading),
    );
    let relaunch_outcome = replace_terminated_worker(
        &mut runtime.worker_process,
        &runtime.health_snapshot,
        runtime.model_load_timeout,
    )
    .await;
    match relaunch_outcome {
        Ok(()) => {
            runtime.is_ready = true;
            runtime.model_load_deadline = None;
            tracing::info!(
                worker_process_id = ?runtime.worker_process.process_id(),
                "replacement worker ready after failure containment"
            );
        }
        Err(relaunch_error) => {
            runtime.is_ready = false;
            tracing::error!(
                error = %relaunch_error,
                "failed to relaunch worker after failure containment; worker stays unavailable"
            );
        }
    }
}

#[cfg(test)]
#[path = "failure_relaunch_tests.rs"]
mod failure_relaunch_tests;
