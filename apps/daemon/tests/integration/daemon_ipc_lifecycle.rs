//! Daemon IPC service lifecycle journeys: handshake identity, socket file
//! hygiene, and single-listener enforcement on the instance socket.

use ipc_protocol::{
    DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION, DaemonRequest, DaemonResponse,
};

use crate::integration::daemon_ipc_exchange::request_and_read_first_response;
use crate::integration::daemon_ipc_service_fixtures::{
    development_instance_paths, shut_down_and_remove_state_directory, shut_down_service,
    try_start_service_on_instance_socket,
};
use crate::support::daemon_ipc::{disabled_daemon_attribution_log, fresh_instance_state_directory};

#[tokio::test]
async fn should_answer_handshake_with_application_identity_on_the_instance_socket() {
    let state_directory = fresh_instance_state_directory("handshake-identity");
    let instance_paths = development_instance_paths(&state_directory);
    let daemon_attribution_log = disabled_daemon_attribution_log(&state_directory);
    let daemon_ipc_service =
        try_start_service_on_instance_socket(&instance_paths, &daemon_attribution_log)
            .await
            .expect("the daemon IPC service should start on the instance socket");

    let handshake_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &DaemonRequest::Handshake,
    )
    .await;

    assert_eq!(
        handshake_response,
        DaemonResponse::HandshakeAccepted {
            protocol_version: DAEMON_PROTOCOL_VERSION,
            application_name: DAEMON_APPLICATION_NAME.to_owned(),
        }
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_remove_the_daemon_socket_file_on_shutdown() {
    let state_directory = fresh_instance_state_directory("shutdown");
    let instance_paths = development_instance_paths(&state_directory);
    let daemon_attribution_log = disabled_daemon_attribution_log(&state_directory);
    let daemon_ipc_service =
        try_start_service_on_instance_socket(&instance_paths, &daemon_attribution_log)
            .await
            .expect("the daemon IPC service should start on the instance socket");
    let socket_path = daemon_ipc_service.socket_path().to_path_buf();
    assert!(
        socket_path.exists(),
        "the socket file should exist while serving"
    );

    // The state directory stays until the assertion: removing it first would
    // make the socket file vanish for a reason other than the shutdown.
    shut_down_service(daemon_ipc_service).await;

    assert!(
        !socket_path.exists(),
        "the socket file should be removed when the daemon IPC service stops"
    );
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_refuse_a_second_daemon_listener_on_the_same_socket() {
    let state_directory = fresh_instance_state_directory("second-listener");
    let instance_paths = development_instance_paths(&state_directory);
    let daemon_attribution_log = disabled_daemon_attribution_log(&state_directory);
    let running_service =
        try_start_service_on_instance_socket(&instance_paths, &daemon_attribution_log)
            .await
            .expect("the first daemon IPC service should start on the instance socket");

    let second_service_start =
        try_start_service_on_instance_socket(&instance_paths, &daemon_attribution_log).await;

    assert!(
        second_service_start.is_err(),
        "a second daemon listener on the same instance socket must be refused while the first is live"
    );

    shut_down_and_remove_state_directory(running_service, &state_directory).await;
}
