//! Daemon IPC service lifecycle journeys: handshake identity, socket file
//! hygiene, and single-listener enforcement on the instance socket.

use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};
use astronomical_ipc_protocol::{
    DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION, DaemonIpcClient, DaemonRequest,
    DaemonResponse,
};
use astronomical_supervisor::DaemonIpcService;
use tokio::time::timeout;

use crate::common::daemon_ipc::{
    HANDSHAKE_TEST_TIMEOUT, disabled_supervisor_attribution_log, fresh_instance_state_directory,
    unavailable_generation_context,
};

#[tokio::test]
async fn should_answer_handshake_with_application_identity_on_the_instance_socket() {
    let state_directory = fresh_instance_state_directory("handshake-identity");
    let instance_paths = AstronomicalInstancePaths::for_state_directory(
        state_directory.clone(),
        AstronomicalRuntimeInstance::Development,
    );
    let supervisor_attribution_log = disabled_supervisor_attribution_log(&state_directory);

    let daemon_ipc_service: DaemonIpcService = timeout(
        HANDSHAKE_TEST_TIMEOUT,
        astronomical_supervisor::start_daemon_ipc_service(
            &instance_paths,
            unavailable_generation_context(),
            &supervisor_attribution_log,
        ),
    )
    .await
    .expect("the daemon IPC service start should finish inside the test timeout")
    .expect("the daemon IPC service should start on the instance socket");

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&DaemonRequest::Handshake),
    )
    .await
    .expect("the handshake send should finish inside the test timeout")
    .expect("the handshake request should transmit");

    let handshake_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the handshake response should arrive inside the test timeout")
        .expect("the handshake response read should not fail at the transport layer")
        .expect("the daemon should answer the handshake");

    assert_eq!(
        handshake_response,
        DaemonResponse::HandshakeAccepted {
            protocol_version: DAEMON_PROTOCOL_VERSION,
            application_name: DAEMON_APPLICATION_NAME.to_owned(),
        }
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_remove_the_daemon_socket_file_on_shutdown() {
    let state_directory = fresh_instance_state_directory("shutdown");
    let instance_paths = AstronomicalInstancePaths::for_state_directory(
        state_directory.clone(),
        AstronomicalRuntimeInstance::Development,
    );
    let supervisor_attribution_log = disabled_supervisor_attribution_log(&state_directory);

    let daemon_ipc_service: DaemonIpcService = timeout(
        HANDSHAKE_TEST_TIMEOUT,
        astronomical_supervisor::start_daemon_ipc_service(
            &instance_paths,
            unavailable_generation_context(),
            &supervisor_attribution_log,
        ),
    )
    .await
    .expect("the daemon IPC service start should finish inside the test timeout")
    .expect("the daemon IPC service should start on the instance socket");
    let socket_path = daemon_ipc_service.socket_path().to_path_buf();
    assert!(
        socket_path.exists(),
        "the socket file should exist while serving"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");

    assert!(
        !socket_path.exists(),
        "the socket file should be removed when the daemon IPC service stops"
    );
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_refuse_a_second_daemon_listener_on_the_same_socket() {
    let state_directory = fresh_instance_state_directory("second-listener");
    let instance_paths = AstronomicalInstancePaths::for_state_directory(
        state_directory.clone(),
        AstronomicalRuntimeInstance::Development,
    );
    let supervisor_attribution_log = disabled_supervisor_attribution_log(&state_directory);

    let running_service: DaemonIpcService = timeout(
        HANDSHAKE_TEST_TIMEOUT,
        astronomical_supervisor::start_daemon_ipc_service(
            &instance_paths,
            unavailable_generation_context(),
            &supervisor_attribution_log,
        ),
    )
    .await
    .expect("the daemon IPC service start should finish inside the test timeout")
    .expect("the first daemon IPC service should start on the instance socket");

    let second_service_start = timeout(
        HANDSHAKE_TEST_TIMEOUT,
        astronomical_supervisor::start_daemon_ipc_service(
            &instance_paths,
            unavailable_generation_context(),
            &supervisor_attribution_log,
        ),
    )
    .await
    .expect("the second daemon IPC service start attempt should finish inside the test timeout");

    assert!(
        second_service_start.is_err(),
        "a second daemon listener on the same instance socket must be refused while the first is live"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, running_service.shutdown())
        .await
        .expect("the first daemon IPC service shutdown should finish inside the test timeout")
        .expect("the first daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}
