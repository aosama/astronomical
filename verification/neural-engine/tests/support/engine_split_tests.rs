use std::time::Duration;

use super::{milliseconds, summarize};

#[test]
fn should_convert_a_duration_to_fractional_milliseconds() {
    assert_eq!(milliseconds(Duration::from_micros(1500)), 1.5);
    assert_eq!(milliseconds(Duration::ZERO), 0.0);
}

#[test]
fn should_summarize_timed_calls() {
    let summary = summarize(&[Duration::from_millis(2), Duration::from_millis(1)]);

    assert_eq!(summary.sample_count, 2);
    assert_eq!(summary.minimum, Duration::from_millis(1));
}

#[test]
#[should_panic(expected = "calls were timed")]
fn should_refuse_to_summarize_when_no_call_was_timed() {
    summarize(&[]);
}
