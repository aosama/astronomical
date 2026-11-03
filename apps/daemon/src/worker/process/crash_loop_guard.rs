//! Bounds how often the worker loop relaunches a worker that failure
//! containment terminated, so a worker that dies on every start cannot pin the
//! daemon in a relaunch loop.

use std::collections::VecDeque;
use std::time::Duration;

use tokio::time::Instant;

/// Whether one more relaunch fits inside the crash-loop budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RelaunchAdmission {
    /// The relaunch may run; `relaunch_attempt_in_window` counts it, starting at 1.
    Admitted { relaunch_attempt_in_window: usize },
    /// The budget is spent. `should_report` is true only for the first refusal
    /// of an episode, so the log carries one clear line instead of one per
    /// loop iteration.
    Exhausted { should_report: bool },
}

/// A sliding-window relaunch budget that latches once spent.
///
/// The latch matters: a purely sliding window would quietly start relaunching
/// again once old attempts age out, turning "stop relaunching" into "relaunch
/// every window". Only a worker that some other path brought back to life
/// (an explicit restart or replacement) clears the latch.
pub(crate) struct CrashLoopGuard {
    maximum_relaunches_per_window: usize,
    relaunch_window: Duration,
    recent_relaunch_instants: VecDeque<Instant>,
    is_exhausted: bool,
}

impl CrashLoopGuard {
    pub(crate) fn new(maximum_relaunches_per_window: usize, relaunch_window: Duration) -> Self {
        Self {
            maximum_relaunches_per_window,
            relaunch_window,
            recent_relaunch_instants: VecDeque::with_capacity(maximum_relaunches_per_window),
            is_exhausted: false,
        }
    }

    /// Records one relaunch at `now` when the budget allows it.
    pub(crate) fn admit_relaunch(&mut self, now: Instant) -> RelaunchAdmission {
        if self.is_exhausted {
            return RelaunchAdmission::Exhausted {
                should_report: false,
            };
        }
        self.forget_relaunches_outside_window(now);
        if self.recent_relaunch_instants.len() >= self.maximum_relaunches_per_window {
            self.is_exhausted = true;
            return RelaunchAdmission::Exhausted {
                should_report: true,
            };
        }
        self.recent_relaunch_instants.push_back(now);
        RelaunchAdmission::Admitted {
            relaunch_attempt_in_window: self.recent_relaunch_instants.len(),
        }
    }

    /// Clears the exhaustion latch once a worker runs again. The attempt
    /// history stays, so a worker that dies right after an explicit restart
    /// still counts against the same window.
    pub(crate) fn observe_running_worker(&mut self) {
        self.is_exhausted = false;
    }

    fn forget_relaunches_outside_window(&mut self, now: Instant) {
        while let Some(oldest_relaunch_instant) = self.recent_relaunch_instants.front() {
            if now.saturating_duration_since(*oldest_relaunch_instant) < self.relaunch_window {
                return;
            }
            self.recent_relaunch_instants.pop_front();
        }
    }
}

#[cfg(test)]
#[path = "crash_loop_guard_tests.rs"]
mod crash_loop_guard_tests;
