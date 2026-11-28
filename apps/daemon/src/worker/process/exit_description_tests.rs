//! Unit tests for the worker exit descriptions the daemon logs on a death.

use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;

use super::{CRASH_RECORD_LOCATION, worker_exit_cause_label, worker_exit_status_description};

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
    assert!(description.contains(CRASH_RECORD_LOCATION));
}

#[test]
fn should_name_sigkill_and_say_no_crash_record_exists() {
    let description = worker_exit_status_description(killed_by_signal(9));

    assert!(description.starts_with("terminated by signal 9 (SIGKILL)"));
    assert!(description.contains("no crash record exists"));
}

#[test]
fn should_name_every_fault_signal_with_the_crash_report_hint() {
    for (signal_number, signal_name) in [
        (libc::SIGILL, "SIGILL"),
        (libc::SIGTRAP, "SIGTRAP"),
        (libc::SIGABRT, "SIGABRT"),
        (libc::SIGFPE, "SIGFPE"),
        (libc::SIGBUS, "SIGBUS"),
    ] {
        let description = worker_exit_status_description(killed_by_signal(signal_number));

        assert!(description.contains(signal_name), "{description}");
        assert!(description.contains(CRASH_RECORD_LOCATION), "{description}");
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

#[test]
fn should_not_take_a_user_signal_for_a_fault_on_any_platform() {
    // SIGUSR1 is 10 on Linux, the number macOS gives SIGBUS.
    let description = worker_exit_status_description(killed_by_signal(libc::SIGUSR1));

    assert_eq!(
        description,
        format!("terminated by signal {}", libc::SIGUSR1)
    );
    assert!(!description.contains(CRASH_RECORD_LOCATION));
}

#[cfg(target_os = "linux")]
#[test]
fn should_name_the_process_name_coredumpctl_files_a_linux_core_under() {
    // The kernel keeps 15 characters of a process name, and systemd-coredump
    // files the core under that, so the full binary name finds nothing.
    let worker_process_name: String = "astronomical-inference-worker".chars().take(15).collect();

    let description = worker_exit_status_description(killed_by_signal(libc::SIGSEGV));

    assert!(
        description.contains(&format!("coredumpctl list {worker_process_name}`")),
        "{description}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn should_keep_the_macos_signal_numbers_and_wording_unchanged() {
    // Pinned as literals, not through the constants the code shares: the
    // macOS log lines people search for must not change when the Linux
    // wording does (issue #2699 changed only "report" to "record").
    assert_eq!(
        worker_exit_status_description(killed_by_signal(10)),
        "terminated by signal 10 (SIGBUS); the faulting stack is in the \
         astronomical-inference-worker crash report under ~/Library/Logs/DiagnosticReports"
    );
    assert_eq!(
        worker_exit_status_description(killed_by_signal(9)),
        "terminated by signal 9 (SIGKILL); killed from outside the process (jetsam memory \
         pressure, the GPU watchdog, or the daemon), so no crash record exists"
    );
    assert_eq!(
        worker_exit_status_description(killed_by_signal(15)),
        "terminated by signal 15 (SIGTERM)"
    );
    for (signal_number, signal_name) in [
        (4, "SIGILL"),
        (5, "SIGTRAP"),
        (6, "SIGABRT"),
        (8, "SIGFPE"),
        (11, "SIGSEGV"),
    ] {
        assert!(
            worker_exit_status_description(killed_by_signal(signal_number)).starts_with(&format!(
                "terminated by signal {signal_number} ({signal_name}); the faulting stack"
            ))
        );
    }
}
