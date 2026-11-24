//! Unit tests for the progress line.

use super::*;

#[test]
fn should_report_the_percentage_and_the_time_left_partway_through() {
    let line = progress_line(
        "Scoring examples",
        100,
        400,
        "examples",
        Duration::from_secs(60),
    );
    assert_eq!(
        line,
        "Scoring examples: 100 of 400 examples (25%), about 3 min left"
    );
}

#[test]
fn should_say_it_is_estimating_before_there_is_enough_to_go_on() {
    let line = progress_line(
        "Scoring examples",
        0,
        400,
        "examples",
        Duration::from_secs(2),
    );
    assert_eq!(
        line,
        "Scoring examples: 0 of 400 examples (0%), estimating time left"
    );
}

#[test]
fn should_say_done_at_the_end_and_for_an_empty_job() {
    let finished = progress_line("Copying", 10, 10, "MB", Duration::from_secs(5));
    let empty = progress_line("Copying", 0, 0, "MB", Duration::ZERO);
    assert_eq!(finished, "Copying: 10 of 10 MB (100%), done");
    assert_eq!(empty, "Copying: 0 of 0 MB (100%), done");
}

#[test]
fn should_give_seconds_minutes_and_hours_as_the_wait_grows() {
    let seconds = progress_line("Scoring", 50, 100, "examples", Duration::from_secs(30));
    let hours = progress_line("Scoring", 1, 100, "examples", Duration::from_secs(60));
    assert!(seconds.ends_with("about 30 s left"), "{seconds}");
    assert!(hours.ends_with("about 1 h 39 min left"), "{hours}");
}

#[test]
fn should_not_pass_one_hundred_percent_when_more_is_reported_than_expected() {
    let line = progress_line("Scoring", 12, 10, "examples", Duration::from_secs(5));
    assert!(line.contains("(100%)"), "{line}");
}
