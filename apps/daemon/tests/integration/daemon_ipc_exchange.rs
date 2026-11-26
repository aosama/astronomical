//! Client side of the daemon IPC journeys: connecting, sending one request and
//! reading frames, every step bounded by the shared test timeout so a hung
//! daemon fails the test instead of stalling the suite.

use std::path::Path;

use ipc_protocol::{DaemonIpcClient, DaemonRequest, DaemonResponse};
use tokio::time::timeout;

use crate::support::daemon_ipc::HANDSHAKE_TEST_TIMEOUT;

pub(crate) async fn connect_client(socket_path: &Path) -> DaemonIpcClient {
    DaemonIpcClient::connect(socket_path.to_path_buf())
        .await
        .expect("the client should connect to the running daemon IPC service")
}

pub(crate) async fn send_request_within_timeout(
    daemon_client: &mut DaemonIpcClient,
    daemon_request: &DaemonRequest,
) {
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(daemon_request),
    )
    .await
    .expect("the request send should finish inside the test timeout")
    .expect("the request should transmit");
}

/// Reads the next frame, or `None` once the daemon has closed the connection.
pub(crate) async fn next_response_or_close(
    daemon_client: &mut DaemonIpcClient,
) -> Option<DaemonResponse> {
    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the response frame should arrive inside the test timeout")
        .expect("the response frame read should not fail at the transport layer")
}

pub(crate) async fn next_response_within_timeout(
    daemon_client: &mut DaemonIpcClient,
) -> DaemonResponse {
    next_response_or_close(daemon_client)
        .await
        .expect("the daemon should answer the request")
}

pub(crate) async fn connect_and_send_request(
    socket_path: &Path,
    daemon_request: &DaemonRequest,
) -> DaemonIpcClient {
    let mut daemon_client = connect_client(socket_path).await;
    send_request_within_timeout(&mut daemon_client, daemon_request).await;
    daemon_client
}

/// Sends one request on a fresh connection and returns the daemon's first
/// reply. The wire contract serves one request per connection, so every
/// request that follows needs its own call.
pub(crate) async fn request_and_read_first_response(
    socket_path: &Path,
    daemon_request: &DaemonRequest,
) -> DaemonResponse {
    let mut daemon_client = connect_and_send_request(socket_path, daemon_request).await;
    next_response_within_timeout(&mut daemon_client).await
}

/// Performs the handshake on one connection, then sends the request on a
/// fresh connection per the wire contract, and returns the request client
/// for the caller to read the streamed frames.
pub(crate) async fn handshake_then_send_request(
    socket_path: &Path,
    daemon_request: &DaemonRequest,
) -> DaemonIpcClient {
    let _handshake_accepted =
        request_and_read_first_response(socket_path, &DaemonRequest::Handshake).await;
    connect_and_send_request(socket_path, daemon_request).await
}

/// Reads frames until the daemon closes the connection and returns the final
/// frame: the completion for an accepted journey, or the single rejection.
pub(crate) async fn read_final_frame(
    daemon_client: &mut DaemonIpcClient,
) -> Option<DaemonResponse> {
    let mut final_frame = None;
    while let Some(frame) = next_response_or_close(daemon_client).await {
        final_frame = Some(frame);
    }
    final_frame
}
