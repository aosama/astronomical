//! Unit tests for the worker exit descriptions the daemon logs on a death.

use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;

use super::{worker_exit_cause_label, worker_exit_status_description};

/// A wait status for a process that exited normally: the code sits in the
/// second byte, as `waitpid` reports it.
fn exited_with_code(exit_code: i32) -> ExitStatus {
    ExitStatus::from_raw(exit_code << 8)
}

/// A wait status for a process a signal ended: the signal is the low byte.
fn killed_by_signal(signal_number: i32) -> ExitStatus {
    ExitStatus::from_raw(signal_number)
}

#[test]
fn should_name_sigsegv_and_point_at_the_crash_report() {
    let description = worker_exit_status_description(killed_by_signal(11));

    assert!(description.starts_with("terminated by signal 11 (SIGSEGV)"));
    assert!(description.contains("DiagnosticReports"));
}

#[test]
fn should_name_sigkill_and_say_no_crash_report_exists() {
    let description = worker_exit_status_description(killed_by_signal(9));

    assert!(description.starts_with("terminated by signal 9 (SIGKILL)"));
    assert!(description.contains("no crash report exists"));
}

#[test]
fn should_name_every_fault_signal_with_the_crash_report_hint() {
    for (signal_number, signal_name) in [
        (4, "SIGILL"),
        (5, "SIGTRAP"),
        (6, "SIGABRT"),
        (8, "SIGFPE"),
        (10, "SIGBUS"),
    ] {
        let description = worker_exit_status_description(killed_by_signal(signal_number));

        assert!(description.contains(signal_name), "{description}");
        assert!(description.contains("DiagnosticReports"), "{description}");
    }
}

#[test]
fn should_name_sigterm_without_a_crash_report_hint() {
    let description = worker_exit_status_description(killed_by_signal(15));

    assert_eq!(description, "terminated by signal 15 (SIGTERM)");
}

#[test]
fn should_report_an_unnamed_signal_by_number_only() {
    assert_eq!(
        worker_exit_status_description(killed_by_signal(30)),
        "terminated by signal 30"
    );
}

#[test]
fn should_report_exit_codes_at_both_ends_of_the_byte() {
    assert_eq!(
        worker_exit_status_description(exited_with_code(0)),
        "exit code 0"
    );
    assert_eq!(
        worker_exit_status_description(exited_with_code(255)),
        "exit code 255"
    );
}

#[test]
fn should_label_a_fault_a_kill_and_a_failing_exit_code_for_a_problem_report() {
    assert_eq!(
        worker_exit_cause_label(ExitStatus::from_raw(6)).as_deref(),
        Some("signal 6 (SIGABRT)")
    );
    assert_eq!(
        worker_exit_cause_label(ExitStatus::from_raw(9)).as_deref(),
        Some("signal 9 (SIGKILL, killed from outside)")
    );
    assert_eq!(
        worker_exit_cause_label(ExitStatus::from_raw(101 << 8)).as_deref(),
        Some("exit code 101")
    );
    assert_eq!(
        worker_exit_cause_label(ExitStatus::from_raw(31)).as_deref(),
        Some("signal 31")
    );
}

#[test]
fn should_label_nothing_for_an_exit_the_daemon_asks_for() {
    assert_eq!(worker_exit_cause_label(ExitStatus::from_raw(0)), None);
    assert_eq!(worker_exit_cause_label(ExitStatus::from_raw(15)), None);
}
