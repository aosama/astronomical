use std::time::{Duration, Instant};

use super::{ConsoleEcho, OutputStyle, PROGRESS_INTERVAL, SupervisionClock, console_echo_for_line};

const CARGO_DIAGNOSTIC_LINE: &str =
    r#"{"reason":"compiler-message","message":{"rendered":"warning: unused variable\n"}}"#;
const CARGO_ARTIFACT_LINE: &str = r#"{"reason":"compiler-artifact","target":{"name":"xtask"}}"#;

#[test]
fn should_echo_only_the_rendered_diagnostic_of_a_cargo_message() {
    assert_eq!(
        console_echo_for_line(OutputStyle::Cargo, true, CARGO_DIAGNOSTIC_LINE),
        ConsoleEcho::StderrVerbatim("warning: unused variable\n".to_owned())
    );
}

#[test]
fn should_keep_cargo_messages_without_a_rendered_diagnostic_silent() {
    assert_eq!(
        console_echo_for_line(OutputStyle::Cargo, true, CARGO_ARTIFACT_LINE),
        ConsoleEcho::Silent
    );
}

#[test]
fn should_pass_a_non_json_cargo_stdout_line_through_unchanged() {
    assert_eq!(
        console_echo_for_line(OutputStyle::Cargo, true, "running 3 tests"),
        ConsoleEcho::Stdout("running 3 tests".to_owned())
    );
    assert_eq!(
        console_echo_for_line(OutputStyle::Cargo, true, ""),
        ConsoleEcho::Stdout(String::new())
    );
}

#[test]
fn should_echo_test_output_on_the_stream_it_arrived_on() {
    assert_eq!(
        console_echo_for_line(OutputStyle::Test, true, "test ok"),
        ConsoleEcho::Stdout("test ok".to_owned())
    );
    assert_eq!(
        console_echo_for_line(OutputStyle::Cargo, false, "Compiling xtask"),
        ConsoleEcho::Stderr("Compiling xtask".to_owned())
    );
}

#[test]
fn should_echo_nothing_in_the_quiet_style() {
    for is_stdout in [true, false] {
        assert_eq!(
            console_echo_for_line(OutputStyle::Quiet, is_stdout, CARGO_DIAGNOSTIC_LINE),
            ConsoleEcho::Silent
        );
    }
}

#[test]
fn should_exhaust_the_budget_exactly_at_its_end_and_not_before() {
    let started_at = Instant::now();
    let budget = Duration::from_secs(120);
    let supervision_clock = SupervisionClock::start(started_at, budget);
    assert!(!supervision_clock.is_budget_exhausted(started_at));
    assert!(!supervision_clock.is_budget_exhausted(started_at + budget - Duration::from_millis(1)));
    assert!(supervision_clock.is_budget_exhausted(started_at + budget));
}

#[test]
fn should_treat_a_zero_budget_as_already_exhausted() {
    let started_at = Instant::now();
    let supervision_clock = SupervisionClock::start(started_at, Duration::ZERO);
    assert!(supervision_clock.is_budget_exhausted(started_at));
}

#[test]
fn should_report_progress_once_per_interval_and_restart_the_interval() {
    let started_at = Instant::now();
    let mut supervision_clock = SupervisionClock::start(started_at, Duration::from_secs(120));
    assert!(!supervision_clock.take_progress_due(started_at + PROGRESS_INTERVAL / 2));
    let first_report_at = started_at + PROGRESS_INTERVAL;
    assert!(supervision_clock.take_progress_due(first_report_at));
    assert!(!supervision_clock.take_progress_due(first_report_at));
    assert!(supervision_clock.take_progress_due(first_report_at + PROGRESS_INTERVAL));
}
