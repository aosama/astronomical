//! Beside-source tests for how three shutdown outcomes become one result.

use std::io;
use std::path::PathBuf;

use astronomical_daemon::WorkerControlError;
use ipc_protocol::DaemonTransportError;

use super::{DaemonError, settle_shutdown_outcomes};

fn http_failure() -> DaemonError {
    DaemonError::ServeHttp(io::Error::other("listener closed unexpectedly"))
}

fn worker_failure() -> DaemonError {
    DaemonError::ShutdownWorker(WorkerControlError::CommandWriterClosed)
}

fn daemon_ipc_failure() -> DaemonTransportError {
    DaemonTransportError::DaemonNotRunning {
        socket_path: PathBuf::from("daemon.sock"),
    }
}

#[test]
fn should_succeed_when_all_three_shutdown_steps_succeed() {
    assert!(settle_shutdown_outcomes(Ok(()), Ok(()), Ok(())).is_ok());
}

#[test]
fn should_report_the_http_failure_before_the_worker_and_ipc_failures() {
    let settled_outcome = settle_shutdown_outcomes(
        Err(http_failure()),
        Err(worker_failure()),
        Err(daemon_ipc_failure()),
    );

    assert!(matches!(settled_outcome, Err(DaemonError::ServeHttp(_))));
}

#[test]
fn should_report_the_worker_failure_before_the_ipc_failure() {
    let settled_outcome =
        settle_shutdown_outcomes(Ok(()), Err(worker_failure()), Err(daemon_ipc_failure()));

    assert!(matches!(
        settled_outcome,
        Err(DaemonError::ShutdownWorker(
            WorkerControlError::CommandWriterClosed
        ))
    ));
}

#[test]
fn should_wrap_a_lone_ipc_failure_as_a_daemon_ipc_shutdown_error() {
    let settled_outcome = settle_shutdown_outcomes(Ok(()), Ok(()), Err(daemon_ipc_failure()));

    assert!(matches!(
        settled_outcome,
        Err(DaemonError::ShutdownDaemonIpc(
            DaemonTransportError::DaemonNotRunning { .. }
        ))
    ));
}
