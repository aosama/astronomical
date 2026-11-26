//! Serving until shutdown, and what the daemon reports once the HTTP server,
//! the inference worker and the IPC service have all been stopped.

use astronomical_daemon::{DaemonIpcService, WorkerHandle};
use ipc_protocol::DaemonTransportError;
use tokio::net::TcpListener;

use super::{DaemonError, shutdown_signal};

/// Serves the REST API until a shutdown signal arrives, then stops the worker
/// and the IPC service. Both are stopped even when serving failed, so the
/// worker process and the socket file never outlive the daemon.
pub(super) async fn serve_until_shutdown(
    listener: TcpListener,
    application: axum::Router,
    internal_shutdown_receiver: tokio::sync::watch::Receiver<bool>,
    worker_handle: WorkerHandle,
    daemon_ipc_service: DaemonIpcService,
) -> Result<(), DaemonError> {
    let serve_result = axum::serve(listener, application)
        .with_graceful_shutdown(shutdown_signal(internal_shutdown_receiver))
        .await
        .map_err(DaemonError::ServeHttp);
    let worker_shutdown_result = worker_handle
        .shutdown()
        .await
        .map(|_| ())
        .map_err(DaemonError::ShutdownWorker);
    let daemon_ipc_shutdown_result = daemon_ipc_service.shutdown().await;

    settle_shutdown_outcomes(
        serve_result,
        worker_shutdown_result,
        daemon_ipc_shutdown_result,
    )
}

/// Reports the first failure among the three shutdown steps, in the order the
/// steps were started: the HTTP server, the inference worker, then the IPC
/// service. All three have already run by the time this is called, so a failed
/// early step never leaves a later one unshut; the caller still sees only the
/// earliest failure.
fn settle_shutdown_outcomes(
    http_serve_outcome: Result<(), DaemonError>,
    worker_shutdown_outcome: Result<(), DaemonError>,
    daemon_ipc_shutdown_outcome: Result<(), DaemonTransportError>,
) -> Result<(), DaemonError> {
    http_serve_outcome?;
    worker_shutdown_outcome?;
    daemon_ipc_shutdown_outcome.map_err(DaemonError::ShutdownDaemonIpc)?;
    Ok(())
}

#[cfg(test)]
#[path = "shutdown_sequence_tests.rs"]
mod shutdown_sequence_tests;
