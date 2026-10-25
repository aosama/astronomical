use super::pending_successor_reports_zero_first_decode_time;

#[test]
fn should_report_zero_for_the_first_pending_successor_before_any_decode_forward() {
    assert!(pending_successor_reports_zero_first_decode_time(0, None));
}

#[test]
fn should_not_report_zero_once_tokens_have_been_generated() {
    assert!(!pending_successor_reports_zero_first_decode_time(1, None));
}

#[test]
fn should_not_report_zero_once_a_real_decode_forward_was_timed() {
    assert!(!pending_successor_reports_zero_first_decode_time(
        0,
        Some(42)
    ));
}
