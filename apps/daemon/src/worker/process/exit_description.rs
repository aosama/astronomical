//! Human-readable descriptions of how a worker process ended.
//!
//! A worker that dies inside native code (MLX, Metal) never writes a final
//! log line, so the daemon's description of the exit is the only evidence in
//! the logs. The signal number decides where to look next: a fault signal
//! leaves a macOS crash report with the faulting stack, while SIGKILL comes
//! from outside the process (jetsam memory pressure, the GPU watchdog, or the
//! daemon itself) and leaves no report at all.

use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;

/// Signals whose arrival means the worker faulted on its own; macOS writes a
/// crash report with the faulting thread's stack for each of them.
const FAULT_SIGNAL_NAMES: [(i32, &str); 6] = [
    (4, "SIGILL"),
    (5, "SIGTRAP"),
    (6, "SIGABRT"),
    (8, "SIGFPE"),
    (10, "SIGBUS"),
    (11, "SIGSEGV"),
];

const SIGKILL_NUMBER: i32 = 9;
const SIGTERM_NUMBER: i32 = 15;

/// Describes a finished worker's wait status for the daemon's error log,
/// naming the signal and where its evidence lives.
pub(crate) fn worker_exit_status_description(worker_exit_status: ExitStatus) -> String {
    match worker_exit_status.code() {
        Some(exit_code) => format!("exit code {exit_code}"),
        None => terminating_signal_description(worker_exit_status.signal()),
    }
}

fn terminating_signal_description(terminating_signal: Option<i32>) -> String {
    let Some(signal_number) = terminating_signal else {
        return "terminated without an exit code or signal".to_owned();
    };
    for (fault_signal_number, fault_signal_name) in FAULT_SIGNAL_NAMES {
        if fault_signal_number == signal_number {
            return format!(
                "terminated by signal {signal_number} ({fault_signal_name}); the faulting stack \
                 is in the astronomical-inference-worker crash report under \
                 ~/Library/Logs/DiagnosticReports"
            );
        }
    }
    match signal_number {
        SIGKILL_NUMBER => format!(
            "terminated by signal {signal_number} (SIGKILL); killed from outside the process \
             (jetsam memory pressure, the GPU watchdog, or the daemon), so no crash report exists"
        ),
        SIGTERM_NUMBER => format!("terminated by signal {signal_number} (SIGTERM)"),
        _ => format!("terminated by signal {signal_number}"),
    }
}

#[cfg(test)]
#[path = "exit_description_tests.rs"]
mod exit_description_tests;
