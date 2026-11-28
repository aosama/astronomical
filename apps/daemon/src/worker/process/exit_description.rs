//! Human-readable descriptions of how a worker process ended.
//!
//! A worker that dies inside native code (MLX, Metal, CUDA) never writes a
//! final log line, so the daemon's description of the exit is the only
//! evidence in the logs. The signal number decides where to look next: a
//! fault signal leaves a crash record with the faulting stack (a macOS crash
//! report, or on Linux a systemd-coredump core), while SIGKILL comes from
//! outside the process (memory pressure, the GPU watchdog, or the daemon
//! itself) and leaves no record at all.

use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;

/// Signals whose arrival means the worker faulted on its own; each leaves a
/// crash record with the faulting thread's stack. The numbers come from the
/// platform's own headers: SIGBUS is 10 on macOS but 7 on Linux, where 10 is
/// SIGUSR1.
const FAULT_SIGNAL_NAMES: [(i32, &str); 6] = [
    (libc::SIGILL, "SIGILL"),
    (libc::SIGTRAP, "SIGTRAP"),
    (libc::SIGABRT, "SIGABRT"),
    (libc::SIGFPE, "SIGFPE"),
    (libc::SIGBUS, "SIGBUS"),
    (libc::SIGSEGV, "SIGSEGV"),
];

/// Where a faulting worker's stack is recorded on this platform.
#[cfg(target_os = "macos")]
const CRASH_RECORD_LOCATION: &str =
    "the astronomical-inference-worker crash report under ~/Library/Logs/DiagnosticReports";
/// On Linux, systemd-coredump files the core under the process name, which
/// the kernel cuts to 15 characters: `coredumpctl list` with the full name
/// finds nothing.
#[cfg(not(target_os = "macos"))]
const CRASH_RECORD_LOCATION: &str = "the worker's core dump (`coredumpctl list astronomical-in`; \
     the kernel cuts the process name to 15 characters)";

/// What usually kills a worker from outside on this platform.
#[cfg(target_os = "macos")]
const OUTSIDE_KILLERS: &str = "jetsam memory pressure, the GPU watchdog, or the daemon";
#[cfg(not(target_os = "macos"))]
const OUTSIDE_KILLERS: &str = "the out-of-memory killer, or the daemon";

const SIGKILL_NUMBER: i32 = libc::SIGKILL;
const SIGTERM_NUMBER: i32 = libc::SIGTERM;

/// Describes a finished worker's wait status for the daemon's error log,
/// naming the signal and where its evidence lives.
pub(crate) fn worker_exit_status_description(worker_exit_status: ExitStatus) -> String {
    match worker_exit_status.code() {
        Some(exit_code) => format!("exit code {exit_code}"),
        None => terminating_signal_description(worker_exit_status.signal()),
    }
}

/// The exit's cause in a fixed format safe for a problem report, such as
/// "signal 6 (SIGABRT)" or "exit code 101", or `None` for an exit the daemon
/// asks for (exit code 0, SIGTERM), which is not a problem.
pub(crate) fn worker_exit_cause_label(worker_exit_status: ExitStatus) -> Option<String> {
    if let Some(exit_code) = worker_exit_status.code() {
        return (exit_code != 0).then(|| format!("exit code {exit_code}"));
    }
    let signal_number = worker_exit_status.signal()?;
    if signal_number == SIGTERM_NUMBER {
        return None;
    }
    for (fault_signal_number, fault_signal_name) in FAULT_SIGNAL_NAMES {
        if fault_signal_number == signal_number {
            return Some(format!("signal {signal_number} ({fault_signal_name})"));
        }
    }
    if signal_number == SIGKILL_NUMBER {
        return Some(format!(
            "signal {signal_number} (SIGKILL, killed from outside)"
        ));
    }
    Some(format!("signal {signal_number}"))
}

fn terminating_signal_description(terminating_signal: Option<i32>) -> String {
    let Some(signal_number) = terminating_signal else {
        return "terminated without an exit code or signal".to_owned();
    };
    for (fault_signal_number, fault_signal_name) in FAULT_SIGNAL_NAMES {
        if fault_signal_number == signal_number {
            return format!(
                "terminated by signal {signal_number} ({fault_signal_name}); the faulting stack \
                 is in {CRASH_RECORD_LOCATION}"
            );
        }
    }
    match signal_number {
        SIGKILL_NUMBER => format!(
            "terminated by signal {signal_number} (SIGKILL); killed from outside the process \
             ({OUTSIDE_KILLERS}), so no crash record exists"
        ),
        SIGTERM_NUMBER => format!("terminated by signal {signal_number} (SIGTERM)"),
        _ => format!("terminated by signal {signal_number}"),
    }
}

#[cfg(test)]
#[path = "exit_description_tests.rs"]
mod exit_description_tests;
