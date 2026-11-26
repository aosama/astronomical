use std::time::{Duration, Instant};

use super::{IDLE_BEFORE_MEASURING, IdleWatch, STEP_GAP_THAT_MEANS_A_REQUEST_RAN};

const STEP: Duration = Duration::from_millis(100);

/// Steps the watch every 100 ms for `duration`, starting at `started_at`.
fn step_for(
    idle_watch: &mut IdleWatch,
    started_at: Instant,
    duration: Duration,
) -> (Instant, bool) {
    let mut now = started_at;
    let mut is_idle_long_enough = false;
    while now.saturating_duration_since(started_at) < duration {
        now += STEP;
        is_idle_long_enough = idle_watch.step(now, false);
    }
    (now, is_idle_long_enough)
}

#[test]
fn should_not_measure_before_the_gpu_has_sat_idle() {
    let started_at = Instant::now();
    let mut idle_watch = IdleWatch::new(started_at);

    let (_, is_idle_long_enough) =
        step_for(&mut idle_watch, started_at, IDLE_BEFORE_MEASURING - STEP);

    assert!(!is_idle_long_enough);
}

#[test]
fn should_measure_once_the_gpu_has_sat_idle_long_enough() {
    let started_at = Instant::now();
    let mut idle_watch = IdleWatch::new(started_at);

    let (_, is_idle_long_enough) = step_for(&mut idle_watch, started_at, IDLE_BEFORE_MEASURING);

    assert!(is_idle_long_enough);
}

#[test]
fn should_start_waiting_again_after_a_layer_read_or_a_finished_program() {
    let started_at = Instant::now();
    let mut idle_watch = IdleWatch::new(started_at);
    let (read_at, _) = step_for(&mut idle_watch, started_at, IDLE_BEFORE_MEASURING);

    assert!(!idle_watch.step(read_at + STEP, true));
    assert!(!idle_watch.step(read_at + STEP + STEP, false));
}

#[test]
fn should_start_waiting_again_when_a_gap_shows_a_request_ran() {
    let started_at = Instant::now();
    let mut idle_watch = IdleWatch::new(started_at);
    let (last_step_at, _) = step_for(&mut idle_watch, started_at, IDLE_BEFORE_MEASURING);

    assert!(!idle_watch.step(last_step_at + STEP_GAP_THAT_MEANS_A_REQUEST_RAN * 2, false));
}

#[test]
fn should_start_waiting_again_when_a_forward_says_it_used_the_gpu() {
    let started_at = Instant::now();
    let mut idle_watch = IdleWatch::new(started_at);
    let (used_at, _) = step_for(&mut idle_watch, started_at, IDLE_BEFORE_MEASURING - STEP);

    idle_watch.note_use(used_at);

    assert!(!idle_watch.step(used_at + STEP, false));
}

#[test]
fn should_say_how_much_of_the_wait_is_left() {
    let started_at = Instant::now();
    let idle_watch = IdleWatch::new(started_at);

    assert_eq!(idle_watch.remaining(started_at), IDLE_BEFORE_MEASURING);
    assert_eq!(
        idle_watch.remaining(started_at + IDLE_BEFORE_MEASURING * 2),
        Duration::ZERO
    );
}

#[test]
fn should_not_treat_a_gap_of_exactly_the_limit_as_a_request() {
    let started_at = Instant::now();
    let mut idle_watch = IdleWatch::new(started_at);
    let (last_step_at, _) = step_for(&mut idle_watch, started_at, IDLE_BEFORE_MEASURING);

    assert!(idle_watch.step(last_step_at + STEP_GAP_THAT_MEANS_A_REQUEST_RAN, false));
}
