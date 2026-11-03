//! Unit contracts for when the worker loop schedules a relaunch after failure
//! containment.

use std::time::Duration;

use tokio::time::Instant;

use super::ContainedFailureRelaunch;
use crate::WorkerHealthStatus;
use crate::worker::process::crash_loop_guard::CrashLoopGuard;

const RELAUNCH_BACKOFF: Duration = Duration::from_millis(250);

fn two_relaunch_schedule() -> ContainedFailureRelaunch {
    ContainedFailureRelaunch::with_budget(
        CrashLoopGuard::new(2, Duration::from_secs(600)),
        RELAUNCH_BACKOFF,
    )
}

#[test]
fn should_schedule_a_relaunch_after_the_backoff_when_a_contained_worker_is_unavailable() {
    let mut relaunch_schedule = two_relaunch_schedule();
    let containment_observed_at = Instant::now();

    relaunch_schedule.observe_worker(
        false,
        Some(WorkerHealthStatus::Unavailable),
        containment_observed_at,
    );

    assert_eq!(
        relaunch_schedule.relaunch_deadline(),
        Some(containment_observed_at + RELAUNCH_BACKOFF)
    );
}

#[test]
fn should_not_schedule_a_relaunch_while_the_worker_runs() {
    let mut relaunch_schedule = two_relaunch_schedule();

    relaunch_schedule.observe_worker(true, Some(WorkerHealthStatus::Unavailable), Instant::now());

    assert_eq!(relaunch_schedule.relaunch_deadline(), None);
}

#[test]
fn should_not_schedule_a_relaunch_for_a_stopped_worker_that_is_not_unavailable() {
    let mut relaunch_schedule = two_relaunch_schedule();

    relaunch_schedule.observe_worker(false, Some(WorkerHealthStatus::Loading), Instant::now());
    relaunch_schedule.observe_worker(false, None, Instant::now());

    assert_eq!(
        relaunch_schedule.relaunch_deadline(),
        None,
        "only a published Unavailable means containment finished"
    );
}

#[test]
fn should_keep_the_first_deadline_when_observed_again_before_it_fires() {
    let mut relaunch_schedule = two_relaunch_schedule();
    let containment_observed_at = Instant::now();
    relaunch_schedule.observe_worker(
        false,
        Some(WorkerHealthStatus::Unavailable),
        containment_observed_at,
    );

    relaunch_schedule.observe_worker(
        false,
        Some(WorkerHealthStatus::Unavailable),
        containment_observed_at + Duration::from_millis(100),
    );

    assert_eq!(
        relaunch_schedule.relaunch_deadline(),
        Some(containment_observed_at + RELAUNCH_BACKOFF),
        "repeated iterations must neither push the deadline out nor spend budget"
    );
}

#[test]
fn should_stop_scheduling_once_the_crash_loop_budget_is_spent() {
    let mut relaunch_schedule = two_relaunch_schedule();
    let first_containment_at = Instant::now();
    for crash_offset_seconds in [0_u64, 5] {
        relaunch_schedule.relaunch_deadline = None;
        relaunch_schedule.observe_worker(
            false,
            Some(WorkerHealthStatus::Unavailable),
            first_containment_at + Duration::from_secs(crash_offset_seconds),
        );
        assert!(relaunch_schedule.relaunch_deadline().is_some());
    }
    relaunch_schedule.relaunch_deadline = None;

    relaunch_schedule.observe_worker(
        false,
        Some(WorkerHealthStatus::Unavailable),
        first_containment_at + Duration::from_secs(10),
    );

    assert_eq!(relaunch_schedule.relaunch_deadline(), None);
}

#[test]
fn should_drop_a_pending_deadline_when_the_worker_runs_again() {
    let mut relaunch_schedule = two_relaunch_schedule();
    relaunch_schedule.observe_worker(false, Some(WorkerHealthStatus::Unavailable), Instant::now());

    relaunch_schedule.observe_worker(true, Some(WorkerHealthStatus::Ready), Instant::now());

    assert_eq!(
        relaunch_schedule.relaunch_deadline(),
        None,
        "an explicit restart that beat the backoff must not be relaunched over"
    );
}
