use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::time::Instant;

use super::{IdleModelUnloadClock, WorkerIdleness};
use crate::worker::handle::GENERATION_QUEUE_DEPTH;

fn idle_clock_with_span(
    idle_model_unload_seconds: u64,
) -> (IdleModelUnloadClock, Arc<AtomicU64>, Instant) {
    let shared_setting = Arc::new(AtomicU64::new(idle_model_unload_seconds));
    let started_at = Instant::now();
    (
        IdleModelUnloadClock::new(Arc::clone(&shared_setting), started_at),
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
