//! Unit tests for the decision verbs' progress lines.

use super::*;

#[test]
fn should_report_progress_at_the_start_then_once_per_interval_and_at_the_end() {
    let mut progress_text = Vec::new();
    let mut timed_progress =
        TimedProgress::new("Scoring examples", "examples", Duration::from_secs(3_600));
    timed_progress.report(&mut progress_text, 0, 10);
    timed_progress.report(&mut progress_text, 5, 10);
    timed_progress.report(&mut progress_text, 10, 10);
    let progress_lines = String::from_utf8(progress_text).unwrap();
    assert_eq!(progress_lines.lines().count(), 2, "{progress_lines}");
    assert!(progress_lines.ends_with("10 of 10 examples (100%), done\n"));
}

#[test]
fn should_report_the_copy_in_megabytes() {
    let mut progress_text = Vec::new();
    let mut copy_progress = CopyProgressLine {
        stderr: &mut progress_text,
        timed_progress: TimedProgress::new("Copying and checking the model", "MB", Duration::ZERO),
    };
    copy_progress.report(3 * BYTES_PER_MEGABYTE, 8 * BYTES_PER_MEGABYTE + 1);
    copy_progress.report(8 * BYTES_PER_MEGABYTE + 1, 8 * BYTES_PER_MEGABYTE + 1);
    let progress_lines = String::from_utf8(progress_text).unwrap();
    assert!(progress_lines.contains("3 of 9 MB"), "{progress_lines}");
    assert!(
        progress_lines.contains("9 of 9 MB (100%), done"),
        "{progress_lines}"
    );
}
