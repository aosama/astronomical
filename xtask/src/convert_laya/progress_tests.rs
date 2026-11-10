//! Unit contracts for the conversion's progress line.

use std::time::Duration;

use super::progress_line;

#[test]
fn should_report_the_share_done_and_the_time_left_at_the_rate_so_far() {
    let line = progress_line(250, 1_000, Duration::from_secs(10));

    assert_eq!(
        line,
        "[convert-laya] status=progress percentage=25 ETA_seconds=30 elapsed_seconds=10"
    );
}

#[test]
fn should_report_no_time_left_once_everything_is_converted() {
    let line = progress_line(1_000, 1_000, Duration::from_secs(40));

    assert!(line.contains("percentage=100 ETA_seconds=0"));
}

#[test]
fn should_not_divide_by_zero_before_anything_is_converted_or_for_an_empty_checkpoint() {
    assert!(progress_line(0, 1_000, Duration::from_secs(1)).contains("percentage=0 ETA_seconds=0"));
    assert!(progress_line(0, 0, Duration::ZERO).contains("percentage=100 ETA_seconds=0"));
}
