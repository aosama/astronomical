use std::time::Duration;

use super::{SLOW_STREAM_WINDOW, StreamRateReference, bytes_per_second};

const MEBIBYTE: u64 = 1 << 20;

#[test]
fn should_call_a_stream_slow_when_it_runs_far_behind_the_others() {
    let rate_reference = StreamRateReference::default();
    rate_reference.report(5 * MEBIBYTE);

    // 100 KiB/s against streams doing 5 MiB/s.
    assert!(rate_reference.is_slow(1_000 * 1024, SLOW_STREAM_WINDOW));
    assert!(rate_reference.is_slow(0, SLOW_STREAM_WINDOW));
}

#[test]
fn should_leave_a_stream_alone_at_or_above_an_eighth_of_the_reference() {
    let rate_reference = StreamRateReference::default();
    rate_reference.report(8 * MEBIBYTE);

    assert!(!rate_reference.is_slow(10 * MEBIBYTE, SLOW_STREAM_WINDOW));
    assert!(rate_reference.is_slow(10 * MEBIBYTE - 1_024, SLOW_STREAM_WINDOW));
}

#[test]
fn should_judge_no_stream_before_any_rate_is_known_or_on_a_slow_link() {
    let unmeasured_reference = StreamRateReference::default();
    assert!(!unmeasured_reference.is_slow(0, SLOW_STREAM_WINDOW));

    let slow_link_reference = StreamRateReference::default();
    slow_link_reference.report(100 * 1024);
    assert!(!slow_link_reference.is_slow(0, SLOW_STREAM_WINDOW));
}

#[test]
fn should_follow_a_network_that_slows_down_within_a_few_reports() {
    let rate_reference = StreamRateReference::default();
    rate_reference.report(8 * MEBIBYTE);
    let degraded_stream_bytes_per_second = MEBIBYTE / 2;
    assert!(rate_reference.is_slow(degraded_stream_bytes_per_second * 10, SLOW_STREAM_WINDOW));

    for _report in 0..8 {
        rate_reference.report(degraded_stream_bytes_per_second);
    }

    assert!(!rate_reference.is_slow(degraded_stream_bytes_per_second * 10, SLOW_STREAM_WINDOW));
}

#[test]
fn should_raise_the_reference_at_once_for_a_faster_stream() {
    let rate_reference = StreamRateReference::default();
    rate_reference.report(MEBIBYTE);
    rate_reference.report(16 * MEBIBYTE);

    assert!(rate_reference.is_slow(10 * MEBIBYTE, SLOW_STREAM_WINDOW));
}

#[test]
fn should_measure_rates_and_treat_an_unmeasurable_duration_as_zero() {
    assert_eq!(
        bytes_per_second(10 * MEBIBYTE, Duration::from_secs(10)),
        MEBIBYTE
    );
    assert_eq!(
        bytes_per_second(MEBIBYTE, Duration::from_millis(500)),
        2 * MEBIBYTE
    );
    assert_eq!(bytes_per_second(MEBIBYTE, Duration::ZERO), 0);
    assert_eq!(
        bytes_per_second(u64::MAX, Duration::from_secs(1)),
        u64::MAX / 1_000
    );
}
