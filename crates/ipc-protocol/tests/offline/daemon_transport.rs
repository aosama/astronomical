use std::{
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process,
    time::{SystemTime, UNIX_EPOCH},
};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationSettings, ChatMessage, DAEMON_APPLICATION_NAME,
    DAEMON_PROTOCOL_VERSION, DaemonIpcClient, DaemonIpcListener, DaemonRequest, DaemonResponse,
    DaemonTransportError,
};
use tokio::time::{Duration, timeout};

const HANDSHAKE_TEST_TIMEOUT: Duration = Duration::from_secs(10);
const SOCKET_FILE_NAME: &str = "ipc.sock";

fn fresh_transport_test_directory(test_name: &str) -> PathBuf {
    let nanos_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should provide time after the epoch")
        .as_nanos();
    let test_directory = std::env::temp_dir().join(format!(
        "ast-ipc-{}-{}-{test_name}",
        process::id(),
        nanos_since_epoch % 1_000_000_000
    ));
    std::fs::create_dir_all(&test_directory)
        .expect("the transport test directory should be creatable");
    test_directory
}

fn cleanup_transport_test_directory(test_directory: &Path) {
    let _ = std::fs::remove_dir_all(test_directory);
}

#[tokio::test]
async fn should_round_trip_handshake_request_and_accepted_response_over_unix_socket() {
    let test_directory = fresh_transport_test_directory("handshake-round-trip");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);

    let mut daemon_listener = DaemonIpcListener::bind(socket_path.clone())
        .await
        .expect("the daemon listener should bind a fresh socket path");
    let listener_task = tokio::spawn(async move {
        daemon_listener
            .serve_next_request(|daemon_request| async move {
                assert_eq!(daemon_request, DaemonRequest::Handshake);
                DaemonResponse::HandshakeAccepted {
                    protocol_version: DAEMON_PROTOCOL_VERSION,
                    application_name: DAEMON_APPLICATION_NAME.to_owned(),
                }
            })
            .await
            .expect("the listener should serve the handshake connection")
    });

    let mut daemon_client = timeout(
        HANDSHAKE_TEST_TIMEOUT,
        DaemonIpcClient::connect(socket_path.clone()),
    )
    .await
    .expect("the client connect should finish inside the test timeout")
    .expect("the client should connect to the bound daemon socket");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&DaemonRequest::Handshake),
    )
    .await
    .expect("the request send should finish inside the test timeout")
    .expect("the client should transmit the handshake request");

    let daemon_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the response read should finish inside the test timeout")
        .expect("the response read should not fail at the transport layer")
        .expect("the daemon should answer the handshake before closing");

    assert_eq!(
        daemon_response,
        DaemonResponse::HandshakeAccepted {
            protocol_version: DAEMON_PROTOCOL_VERSION,
            application_name: DAEMON_APPLICATION_NAME.to_owned(),
        }
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, listener_task)
        .await
        .expect("the listener task should finish inside the test timeout")
        .expect("the listener task should not panic");
    cleanup_transport_test_directory(&test_directory);
}

#[tokio::test]
async fn should_replace_stale_socket_file_when_binding() {
    let test_directory = fresh_transport_test_directory("stale-socket");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    std::fs::write(&socket_path, b"stale bytes from a dead daemon")
        .expect("the stale socket stand-in file should be writable");

    let bind_result = DaemonIpcListener::bind(socket_path.clone()).await;

    assert!(
        bind_result.is_ok(),
        "binding over a stale socket file should replace it, got: {:?}",
        bind_result.err()
    );
    cleanup_transport_test_directory(&test_directory);
}

#[tokio::test]
async fn should_create_daemon_socket_with_owner_only_permissions() {
    let test_directory = fresh_transport_test_directory("socket-permissions");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);

    let daemon_listener = DaemonIpcListener::bind(socket_path.clone())
        .await
        .expect("the daemon listener should bind a fresh socket path");

    let socket_permissions = std::fs::metadata(daemon_listener.socket_path())
        .expect("the bound socket file should exist")
        .permissions();
    assert_eq!(
        socket_permissions.mode() & 0o777,
        0o600,
        "the daemon socket must be readable and writable by the owning user only"
    );
    drop(daemon_listener);
    cleanup_transport_test_directory(&test_directory);
}

#[tokio::test]
async fn should_report_daemon_not_running_when_connecting_to_missing_socket() {
    let test_directory = fresh_transport_test_directory("not-running");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);

    let connect_result: Result<DaemonIpcClient, DaemonTransportError> = timeout(
        HANDSHAKE_TEST_TIMEOUT,
        DaemonIpcClient::connect(socket_path.clone()),
    )
    .await
    .expect("the client connect attempt should finish inside the test timeout");

    match connect_result {
        Err(DaemonTransportError::DaemonNotRunning {
            socket_path: reported_path,
        }) => {
            assert_eq!(reported_path, socket_path);
        }
        Err(unexpected_error) => panic!(
            "connecting to a missing daemon socket should report DaemonNotRunning, got: {unexpected_error:?}"
        ),
        Ok(_unexpected_client) => {
            panic!("connecting to a missing daemon socket should fail, but a client connected")
        }
    }
    cleanup_transport_test_directory(&test_directory);
}

#[tokio::test]
async fn should_keep_serving_after_client_disconnects_without_a_request() {
    let test_directory = fresh_transport_test_directory("silent-disconnect");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);

    let mut daemon_listener = DaemonIpcListener::bind(socket_path.clone())
        .await
        .expect("the daemon listener should bind a fresh socket path");

    {
        let _silent_client = DaemonIpcClient::connect(socket_path.clone())
            .await
            .expect("the silent client should connect before dropping");
    }

    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_listener.serve_next_request(|daemon_request| async move {
            let _ = daemon_request;
            DaemonResponse::HandshakeAccepted {
                protocol_version: DAEMON_PROTOCOL_VERSION,
                application_name: DAEMON_APPLICATION_NAME.to_owned(),
            }
        }),
    )
    .await
    .expect("serving the silent connection should finish inside the test timeout")
    .expect("a client that vanishes without a request must not fail the listener");

    let mut following_client = DaemonIpcClient::connect(socket_path.clone())
        .await
        .expect("the following client should connect after the silent disconnect");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        following_client.send_request(&DaemonRequest::Handshake),
    )
    .await
    .expect("the following request send should finish inside the test timeout")
    .expect("the following client should transmit after the silent disconnect");
    // The daemon serves connections one at a time in a loop, so the second
    // serve_next_request call mirrors the real serving loop after the silent
    // connection ended.
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_listener.serve_next_request(|daemon_request| async move {
            let _ = daemon_request;
            DaemonResponse::HandshakeAccepted {
                protocol_version: DAEMON_PROTOCOL_VERSION,
                application_name: DAEMON_APPLICATION_NAME.to_owned(),
            }
        }),
    )
    .await
    .expect("serving the following connection should finish inside the test timeout")
    .expect("the listener should serve the connection after the silent disconnect");
    let following_response = timeout(HANDSHAKE_TEST_TIMEOUT, following_client.next_response())
        .await
        .expect("the following response read should finish inside the test timeout")
        .expect("the following response read should not fail at the transport layer")
        .expect("the listener should still answer after the silent disconnect");
    assert!(matches!(
        following_response,
        DaemonResponse::HandshakeAccepted { .. }
    ));
    cleanup_transport_test_directory(&test_directory);
}

#[tokio::test]
async fn should_stream_multiple_responses_for_one_request() {
    let test_directory = fresh_transport_test_directory("stream-many");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let mut daemon_listener = DaemonIpcListener::bind(socket_path.clone())
        .await
        .expect("the daemon listener should bind a fresh socket path");

    let mut daemon_client = DaemonIpcClient::connect(socket_path.clone())
        .await
        .expect("the client should connect to the streaming listener");
    let chat_generate_request = DaemonRequest::ChatGenerate {
        model: "example/local-model".to_owned(),
        messages: vec![ChatMessage::User {
            content: "Say hello".to_owned(),
            images: vec![],
        }],
        settings: ChatGenerationSettings {
            max_output_tokens: 128,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        schema_json: None,
    };
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&chat_generate_request),
    )
    .await
    .expect("the chat generate send should finish inside the test timeout")
    .expect("the chat generate request should transmit");

    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_listener.serve_streaming_request(|daemon_request, mut response_writer| async move {
            assert_eq!(daemon_request, chat_generate_request);
            response_writer
                .send_response(&DaemonResponse::ChatGenerationText {
                    text: "Hello".to_owned(),
                })
                .await
                .expect("the first fragment should transmit");
            response_writer
                .send_response(&DaemonResponse::ChatGenerationText {
                    text: " world".to_owned(),
                })
                .await
                .expect("the second fragment should transmit");
            response_writer
                .send_response(&DaemonResponse::ChatGenerationCompleted {
                    prompt_token_count: 5,
                    generated_token_count: 2,
                    reasoning_token_count: 0,
                    cached_token_count: 0,
                    reason: ChatGenerationCompletionReason::EndOfSequence,
                })
                .await
                .expect("the completion frame should transmit");
            response_writer.close().await
        }),
    )
    .await
    .expect("the streaming serve should finish inside the test timeout")
    .expect("the streaming serve should succeed while the client reads");

    let first_fragment = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the first fragment read should finish inside the test timeout")
        .expect("the first fragment read should not fail at the transport layer")
        .expect("the first fragment should arrive");
    let second_fragment = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the second fragment read should finish inside the test timeout")
        .expect("the second fragment read should not fail at the transport layer")
        .expect("the second fragment should arrive");
    let completion_frame = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the completion read should finish inside the test timeout")
        .expect("the completion read should not fail at the transport layer")
        .expect("the completion frame should arrive");
    assert!(matches!(
        first_fragment,
        DaemonResponse::ChatGenerationText { text } if text == "Hello"
    ));
    assert!(matches!(
        second_fragment,
        DaemonResponse::ChatGenerationText { text } if text == " world"
    ));
    assert!(matches!(
        completion_frame,
        DaemonResponse::ChatGenerationCompleted {
            prompt_token_count: 5,
            generated_token_count: 2,
            ..
        }
    ));

    let connection_closed = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the terminal read should finish inside the test timeout")
        .expect("the terminal read should not fail at the transport layer");
    assert!(
        connection_closed.is_none(),
        "the daemon should close the connection after the terminal frame"
    );
    cleanup_transport_test_directory(&test_directory);
}

#[tokio::test]
async fn should_survive_a_client_disconnect_mid_stream() {
    let test_directory = fresh_transport_test_directory("stream-disconnect");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let mut daemon_listener = DaemonIpcListener::bind(socket_path.clone())
        .await
        .expect("the daemon listener should bind a fresh socket path");

    let mut abandoning_client = DaemonIpcClient::connect(socket_path.clone())
        .await
        .expect("the abandoning client should connect to the streaming listener");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        abandoning_client.send_request(&DaemonRequest::ChatGenerate {
            model: "example/local-model".to_owned(),
            messages: vec![ChatMessage::User {
                content: "Say hello".to_owned(),
                images: vec![],
            }],
            settings: ChatGenerationSettings {
                max_output_tokens: 128,
                temperature_thousandths: None,
                top_p_thousandths: None,
                seed: None,
                thinking_budget: None,
            },
            schema_json: None,
        }),
    )
    .await
    .expect("the abandoning send should finish inside the test timeout")
    .expect("the abandoning request should transmit");
    // The client vanishes mid-stream; the kernel buffers a few frames before
    // the sends start failing, which is exactly what the daemon must survive.
    drop(abandoning_client);

    let oversized_fragment_text = "x".repeat(64 * 1024);
    let stream_outcome = timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_listener.serve_streaming_request(
            |_daemon_request, mut response_writer| async move {
                for _fragment_index in 0..64 {
                    response_writer
                        .send_response(&DaemonResponse::ChatGenerationText {
                            text: oversized_fragment_text.clone(),
                        })
                        .await?;
                }
                response_writer.close().await
            },
        ),
    )
    .await
    .expect("the streaming serve should finish inside the test timeout");
    match stream_outcome {
        Ok(()) => {}
        Err(stream_error) => assert!(
            !matches!(stream_error, DaemonTransportError::AcceptFailed { .. }),
            "a vanished mid-stream client is a per-request failure, not a listener failure: {stream_error}"
        ),
    }

    let mut following_client = DaemonIpcClient::connect(socket_path.clone())
        .await
        .expect("the following client should connect after the mid-stream disconnect");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        following_client.send_request(&DaemonRequest::Handshake),
    )
    .await
    .expect("the following send should finish inside the test timeout")
    .expect("the following request should transmit");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_listener.serve_next_request(|_daemon_request| async move {
            DaemonResponse::HandshakeAccepted {
                protocol_version: DAEMON_PROTOCOL_VERSION,
                application_name: DAEMON_APPLICATION_NAME.to_owned(),
            }
        }),
    )
    .await
    .expect("serving the following connection should finish inside the test timeout")
    .expect("the listener should survive the mid-stream disconnect");
    let following_response = timeout(HANDSHAKE_TEST_TIMEOUT, following_client.next_response())
        .await
        .expect("the following response read should finish inside the test timeout")
        .expect("the following response read should not fail at the transport layer")
        .expect("the daemon should still answer after the mid-stream disconnect");
    assert!(matches!(
        following_response,
        DaemonResponse::HandshakeAccepted { .. }
    ));
    cleanup_transport_test_directory(&test_directory);
}
