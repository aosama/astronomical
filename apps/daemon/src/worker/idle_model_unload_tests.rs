use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::time::Instant;

use super::{
    IdleModelUnloadClock, WorkerIdleness, idle_clock_millis, idle_unload_remaining_seconds,
};
use crate::worker::handle::GENERATION_QUEUE_DEPTH;

fn idle_clock_with_span(
    idle_model_unload_seconds: u64,
) -> (IdleModelUnloadClock, Arc<AtomicU64>, Instant) {
    let shared_setting = Arc::new(AtomicU64::new(idle_model_unload_seconds));
    let started_at = Instant::now();
    (
        IdleModelUnloadClock::new(
            Arc::clone(&shared_setting),
            Arc::new(AtomicU64::new(0)),
            started_at,
        ),
        shared_setting,
        started_at,
    )
}

#[test]
fn should_be_due_once_the_configured_span_has_passed_without_activity() {
    let (idle_clock, _shared_setting, started_at) = idle_clock_with_span(60);

    assert!(idle_clock.is_unload_due(started_at + Duration::from_secs(61)));
}

#[test]
fn should_not_be_due_before_the_configured_span_has_passed() {
    let (idle_clock, _shared_setting, started_at) = idle_clock_with_span(60);

    assert!(!idle_clock.is_unload_due(started_at + Duration::from_secs(59)));
}

#[test]
fn should_be_due_exactly_at_the_configured_span() {
    let (idle_clock, _shared_setting, started_at) = idle_clock_with_span(60);

    assert!(idle_clock.is_unload_due(started_at + Duration::from_secs(60)));
}

#[test]
fn should_never_be_due_when_the_span_is_zero() {
    let (idle_clock, _shared_setting, started_at) = idle_clock_with_span(0);

    assert!(!idle_clock.is_unload_due(started_at + Duration::from_secs(7 * 24 * 60 * 60)));
}

#[test]
fn should_restart_the_clock_when_a_request_is_served() {
    let (mut idle_clock, _shared_setting, started_at) = idle_clock_with_span(60);

    idle_clock.record_activity(started_at + Duration::from_secs(50));

    assert!(!idle_clock.is_unload_due(started_at + Duration::from_secs(100)));
    assert!(idle_clock.is_unload_due(started_at + Duration::from_secs(110)));
}

#[test]
fn should_follow_a_live_change_of_the_shared_setting() {
    let (idle_clock, shared_setting, started_at) = idle_clock_with_span(600);
    let checked_at = started_at + Duration::from_secs(120);
    assert!(!idle_clock.is_unload_due(checked_at));

    shared_setting.store(60, Ordering::Relaxed);

    assert!(idle_clock.is_unload_due(checked_at));
}

#[test]
fn should_not_be_due_when_the_check_time_is_before_the_last_activity() {
    let (mut idle_clock, _shared_setting, started_at) = idle_clock_with_span(1);
    idle_clock.record_activity(started_at + Duration::from_secs(10));

    assert!(!idle_clock.is_unload_due(started_at));
}

fn idle_worker_with_a_loaded_model() -> WorkerIdleness {
    WorkerIdleness {
        has_loaded_model: true,
        has_active_request: false,
        free_active_generation_permits: 1,
        free_generation_queue_permits: GENERATION_QUEUE_DEPTH,
    }
}

#[test]
fn should_be_free_to_unload_a_loaded_model_when_nothing_is_running_or_queued() {
    assert!(idle_worker_with_a_loaded_model().is_free_to_unload());
}

#[test]
fn should_not_unload_when_no_model_is_loaded() {
    let worker_idleness = WorkerIdleness {
        has_loaded_model: false,
        ..idle_worker_with_a_loaded_model()
    };

    assert!(!worker_idleness.is_free_to_unload());
}

#[test]
fn should_not_unload_while_a_request_is_active() {
    let worker_idleness = WorkerIdleness {
        has_active_request: true,
        ..idle_worker_with_a_loaded_model()
    };

    assert!(!worker_idleness.is_free_to_unload());
}

#[test]
fn should_not_unload_while_a_request_holds_the_generation_slot() {
    let worker_idleness = WorkerIdleness {
        free_active_generation_permits: 0,
        ..idle_worker_with_a_loaded_model()
    };

    assert!(!worker_idleness.is_free_to_unload());
}

#[test]
fn should_not_unload_while_even_one_request_waits_in_the_queue() {
    let worker_idleness = WorkerIdleness {
        free_generation_queue_permits: GENERATION_QUEUE_DEPTH - 1,
        ..idle_worker_with_a_loaded_model()
    };

    assert!(!worker_idleness.is_free_to_unload());
}

#[test]
fn should_start_the_idle_span_from_the_last_event_of_a_long_request() {
    let (mut idle_clock, _shared_setting, started_at) = idle_clock_with_span(60);
    // A two-minute image whose only event lands at the end of the wait.
    let final_event_at = started_at + Duration::from_secs(120);

    idle_clock.observe_iteration_end(true, final_event_at);

    assert!(!idle_clock.is_unload_due(final_event_at + Duration::from_secs(1)));
    assert!(idle_clock.is_unload_due(final_event_at + Duration::from_secs(60)));
}

#[test]
fn should_leave_the_clock_alone_after_an_iteration_that_served_nothing() {
    let (mut idle_clock, _shared_setting, started_at) = idle_clock_with_span(60);

    idle_clock.observe_iteration_end(false, started_at + Duration::from_secs(59));

    assert!(idle_clock.is_unload_due(started_at + Duration::from_secs(60)));
}

#[test]
fn should_count_down_the_seconds_left_rounding_up() {
    assert_eq!(
        idle_unload_remaining_seconds(60, 1_000_000, 1_000_000),
        Some(60)
    );
    assert_eq!(
        idle_unload_remaining_seconds(60, 1_000_000, 1_018_500),
        Some(42)
    );
}

#[test]
fn should_report_zero_once_the_span_has_passed() {
    assert_eq!(
        idle_unload_remaining_seconds(60, 1_000_000, 1_090_000),
        Some(0)
    );
}

#[test]
fn should_report_no_countdown_when_unloading_is_off_or_the_clock_never_ran() {
    assert_eq!(idle_unload_remaining_seconds(0, 1_000_000, 1_010_000), None);
    assert_eq!(idle_unload_remaining_seconds(60, 0, 1_010_000), None);
}

#[test]
fn should_publish_only_restarts_of_the_clock() {
    let last_activity_clock_millis = Arc::new(AtomicU64::new(0));
    let mut idle_clock = IdleModelUnloadClock::new(
        Arc::new(AtomicU64::new(60)),
        Arc::clone(&last_activity_clock_millis),
        Instant::now(),
    );
    let published_at_start = last_activity_clock_millis.load(Ordering::Relaxed);
    assert!(
        published_at_start > 0,
        "starting the clock publishes a time"
    );

    idle_clock.observe_iteration_end(false, Instant::now());
    assert_eq!(
        last_activity_clock_millis.load(Ordering::Relaxed),
        published_at_start
    );
}

#[test]
fn should_publish_the_time_of_a_request_that_ended() {
    let last_activity_clock_millis = Arc::new(AtomicU64::new(0));
    let started_at = Instant::now();
    let mut idle_clock = IdleModelUnloadClock::new(
        Arc::new(AtomicU64::new(60)),
        Arc::clone(&last_activity_clock_millis),
        started_at,
    );
    let request_ended_at = started_at + Duration::from_secs(30);

    idle_clock.observe_iteration_end(true, request_ended_at);

    assert_eq!(
        last_activity_clock_millis.load(Ordering::Relaxed),
        idle_clock_millis(request_ended_at)
    );
}

#[test]
fn should_give_the_full_span_when_the_check_lands_before_the_last_activity() {
    assert_eq!(idle_unload_remaining_seconds(60, 2_000, 1_000), Some(60));
}

#[test]
fn should_saturate_rather_than_overflow_for_the_largest_span() {
    assert_eq!(
        idle_unload_remaining_seconds(u64::MAX, 1, 1),
        Some(u64::MAX.saturating_mul(1000).div_ceil(1000))
    );
}

#[test]
fn should_not_be_due_while_neural_engine_programs_are_preparing() {
    let (mut idle_clock, _shared_setting, started_at) = idle_clock_with_span(60);

    idle_clock.observe_background_preparation(true, started_at + Duration::from_secs(58));

    assert!(!idle_clock.is_unload_due(started_at + Duration::from_secs(61)));
}

#[test]
fn should_count_the_span_from_the_end_of_preparation() {
    let (mut idle_clock, _shared_setting, started_at) = idle_clock_with_span(60);
    let preparation_ended_at = started_at + Duration::from_secs(58);
    idle_clock.observe_background_preparation(true, preparation_ended_at);
    idle_clock.observe_background_preparation(false, started_at + Duration::from_secs(59));

    assert!(!idle_clock.is_unload_due(preparation_ended_at + Duration::from_secs(59)));
    assert!(idle_clock.is_unload_due(preparation_ended_at + Duration::from_secs(60)));
}

#[test]
fn should_leave_the_clock_alone_when_nothing_is_preparing() {
    let (mut idle_clock, _shared_setting, started_at) = idle_clock_with_span(60);

    idle_clock.observe_background_preparation(false, started_at + Duration::from_secs(58));

    assert!(idle_clock.is_unload_due(started_at + Duration::from_secs(60)));
}

#[test]
fn should_still_never_be_due_with_a_zero_span_after_preparation() {
    let (mut idle_clock, _shared_setting, started_at) = idle_clock_with_span(0);

    idle_clock.observe_background_preparation(true, started_at + Duration::from_secs(58));

    assert!(!idle_clock.is_unload_due(started_at + Duration::from_secs(7 * 24 * 60 * 60)));
}
