//! Unit contracts for the crash-loop relaunch budget.

use std::time::Duration;

use tokio::time::Instant;

use super::{CrashLoopGuard, RelaunchAdmission};

const MAXIMUM_RELAUNCHES_PER_WINDOW: usize = 3;
const RELAUNCH_WINDOW: Duration = Duration::from_secs(600);

fn three_per_ten_minutes_guard() -> CrashLoopGuard {
    CrashLoopGuard::new(MAXIMUM_RELAUNCHES_PER_WINDOW, RELAUNCH_WINDOW)
}

fn admit_until_exhausted(crash_loop_guard: &mut CrashLoopGuard, first_crash_at: Instant) {
    for relaunch_attempt in 1..=MAXIMUM_RELAUNCHES_PER_WINDOW {
        let crash_offset = Duration::from_secs(u64::try_from(relaunch_attempt).unwrap_or(0));
        assert_eq!(
            crash_loop_guard.admit_relaunch(first_crash_at + crash_offset),
            RelaunchAdmission::Admitted {
                relaunch_attempt_in_window: relaunch_attempt
            }
        );
    }
}

#[test]
fn should_admit_the_first_relaunch_after_a_single_crash() {
    let mut crash_loop_guard = three_per_ten_minutes_guard();

    assert_eq!(
        crash_loop_guard.admit_relaunch(Instant::now()),
        RelaunchAdmission::Admitted {
            relaunch_attempt_in_window: 1
        }
    );
}

#[test]
fn should_report_exhaustion_once_when_the_window_budget_is_spent() {
    let mut crash_loop_guard = three_per_ten_minutes_guard();
    let first_crash_at = Instant::now();
    admit_until_exhausted(&mut crash_loop_guard, first_crash_at);

    let refused_at = first_crash_at + Duration::from_secs(10);
    assert_eq!(
        crash_loop_guard.admit_relaunch(refused_at),
        RelaunchAdmission::Exhausted {
            should_report: true
        }
    );
    assert_eq!(
        crash_loop_guard.admit_relaunch(refused_at + Duration::from_secs(1)),
        RelaunchAdmission::Exhausted {
            should_report: false
        },
        "the loop asks every iteration; only the first refusal may log"
    );
}

#[test]
fn should_stay_exhausted_after_the_window_elapses_while_no_worker_runs() {
    let mut crash_loop_guard = three_per_ten_minutes_guard();
    let first_crash_at = Instant::now();
    admit_until_exhausted(&mut crash_loop_guard, first_crash_at);
    let _first_refusal = crash_loop_guard.admit_relaunch(first_crash_at + Duration::from_secs(10));

    assert_eq!(
        crash_loop_guard.admit_relaunch(first_crash_at + RELAUNCH_WINDOW * 2),
        RelaunchAdmission::Exhausted {
            should_report: false
        },
        "a crash loop must stop for good, not resume once old attempts age out"
    );
}

#[test]
fn should_admit_again_after_the_window_when_a_worker_ran_in_between() {
    let mut crash_loop_guard = three_per_ten_minutes_guard();
    let first_crash_at = Instant::now();
    admit_until_exhausted(&mut crash_loop_guard, first_crash_at);
    let _first_refusal = crash_loop_guard.admit_relaunch(first_crash_at + Duration::from_secs(10));

    crash_loop_guard.observe_running_worker();

    assert_eq!(
        crash_loop_guard.admit_relaunch(first_crash_at + RELAUNCH_WINDOW + Duration::from_secs(4)),
        RelaunchAdmission::Admitted {
            relaunch_attempt_in_window: 1
        }
    );
}

#[test]
fn should_refuse_immediately_when_a_restarted_worker_crashes_inside_the_same_window() {
    let mut crash_loop_guard = three_per_ten_minutes_guard();
    let first_crash_at = Instant::now();
    admit_until_exhausted(&mut crash_loop_guard, first_crash_at);
    let _first_refusal = crash_loop_guard.admit_relaunch(first_crash_at + Duration::from_secs(10));

    crash_loop_guard.observe_running_worker();

    assert_eq!(
        crash_loop_guard.admit_relaunch(first_crash_at + Duration::from_secs(20)),
        RelaunchAdmission::Exhausted {
            should_report: true
        },
        "an explicit restart clears the latch but not the attempt history"
    );
}

#[test]
fn should_forget_a_relaunch_exactly_one_window_later() {
    let mut crash_loop_guard = three_per_ten_minutes_guard();
    let first_crash_at = Instant::now();
    admit_until_exhausted(&mut crash_loop_guard, first_crash_at);

    // The first attempt sat at first_crash_at + 1s, so it ages out at exactly
    // one window after that instant; the boundary is inclusive.
    assert_eq!(
        crash_loop_guard.admit_relaunch(first_crash_at + Duration::from_secs(1) + RELAUNCH_WINDOW),
        RelaunchAdmission::Admitted {
            relaunch_attempt_in_window: MAXIMUM_RELAUNCHES_PER_WINDOW
        }
    );
}

#[test]
fn should_refuse_every_relaunch_when_the_budget_is_zero() {
    let mut crash_loop_guard = CrashLoopGuard::new(0, RELAUNCH_WINDOW);

    assert_eq!(
        crash_loop_guard.admit_relaunch(Instant::now()),
        RelaunchAdmission::Exhausted {
            should_report: true
        }
    );
}
