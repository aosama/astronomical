//! Unit tests for the worker exit descriptions the daemon logs on a death.

use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;

use super::worker_exit_status_description;

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
