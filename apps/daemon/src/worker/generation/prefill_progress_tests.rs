use std::time::Duration;

use super::running_prefill_elapsed_millis;

#[test]
fn should_use_the_measured_prompt_time_once_the_worker_reports_one() {
    assert_eq!(
        running_prefill_elapsed_millis(4_200, Duration::from_secs(9)),
        4_200
    );
}

#[test]
fn should_use_the_requests_clock_while_a_layer_by_layer_prompt_has_no_measured_time() {
    assert_eq!(
        running_prefill_elapsed_millis(0, Duration::from_millis(12_900)),
        12_900
    );
}

#[test]
fn should_give_no_time_for_a_request_that_has_just_started() {
    assert_eq!(running_prefill_elapsed_millis(0, Duration::ZERO), 0);
}
