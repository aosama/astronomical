use super::progress_is_within_requested_steps;

#[test]
fn should_accept_progress_partway_through_the_requested_steps() {
    assert!(progress_is_within_requested_steps(3, 8, 8));
}

#[test]
fn should_accept_progress_at_zero_and_at_the_last_requested_step() {
    assert!(progress_is_within_requested_steps(0, 8, 8));
    assert!(progress_is_within_requested_steps(8, 8, 8));
}

#[test]
fn should_reject_progress_past_the_requested_step_count() {
    assert!(!progress_is_within_requested_steps(9, 8, 8));
}

#[test]
fn should_reject_progress_from_an_engine_running_a_different_step_count() {
    assert!(!progress_is_within_requested_steps(3, 4, 8));
    assert!(!progress_is_within_requested_steps(3, 9, 8));
}
