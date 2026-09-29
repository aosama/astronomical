//! Daemon IPC chat and status journeys: status probes report the resident
//! model identity, chat requests stream frames, and rejections carry reasons.

use std::sync::{Arc, Mutex, RwLock};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationSettings, ChatMessage, DaemonIpcClient,
    DaemonRequest, DaemonResponse, DaemonWorkerStatus,
};
use astronomical_supervisor::{WorkerHealthSnapshot, WorkerHealthStatus};
use tokio::time::timeout;

use crate::common::daemon_ipc::{
    HANDSHAKE_TEST_TIMEOUT, StubGenerationExecutor, fresh_instance_state_directory,
    ipc_runtime_config, ready_stub_executor, start_stub_daemon_ipc_service,
    start_stub_daemon_ipc_service_with_runtime_config, text_only_chat_generate_request,
};

#[tokio::test]
async fn should_report_the_loaded_model_identity_for_status_requests() {
    let state_directory = fresh_instance_state_directory("status");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;

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
    assert!(matches!(
        handshake_response,
        DaemonResponse::HandshakeAccepted { .. }
    ));

    // The wire contract serves one request per connection: the CLI performs
    // its handshake, then opens a fresh connection for the status probe.
    drop(daemon_client);
    let mut status_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the status client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        status_client.send_request(&DaemonRequest::Status),
    )
    .await
    .expect("the status send should finish inside the test timeout")
    .expect("the status request should transmit");
    let status_response = timeout(HANDSHAKE_TEST_TIMEOUT, status_client.next_response())
        .await
        .expect("the status response should arrive inside the test timeout")
        .expect("the status response read should not fail at the transport layer")
        .expect("the daemon should answer the status request");
    assert_eq!(
        status_response,
        DaemonResponse::Status {
            worker_status: DaemonWorkerStatus::Ready,
            ready_model_id: Some("test/local-chatter".to_owned()),
            default_model_id: Some("Qwen3.5-2B-4bit".to_owned()),
        }
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_stream_chat_generation_frames_to_the_cli_client() {
    let state_directory = fresh_instance_state_directory("stream-generate");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, Arc::clone(&stub_executor)).await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&text_only_chat_generate_request("test/local-chatter")),
    )
    .await
    .expect("the chat generate send should finish inside the test timeout")
    .expect("the chat generate request should transmit");

    let first_fragment = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the first fragment should arrive inside the test timeout")
        .expect("the first fragment read should not fail at the transport layer")
        .expect("the daemon should stream the first fragment");
    let second_fragment = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the second fragment should arrive inside the test timeout")
        .expect("the second fragment read should not fail at the transport layer")
        .expect("the daemon should stream the second fragment");
    let completion_frame = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the completion frame should arrive inside the test timeout")
        .expect("the completion frame read should not fail at the transport layer")
        .expect("the daemon should stream the completion frame");
    assert!(matches!(
        first_fragment,
        DaemonResponse::ChatGenerationText { text } if text == "Hello"
    ));
    assert!(matches!(
        second_fragment,
        DaemonResponse::ChatGenerationText { text } if text == " world"
    ));
    assert_eq!(
        completion_frame,
        DaemonResponse::ChatGenerationCompleted {
            prompt_token_count: 5,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        }
    );
    let connection_closed = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the terminal read should finish inside the test timeout")
        .expect("the terminal read should not fail at the transport layer");
    assert!(
        connection_closed.is_none(),
        "the daemon should close the connection after the terminal frame"
    );

    let received_commands = stub_executor
        .received_commands
        .lock()
        .expect("the stub command log lock should not be poisoned");
    assert_eq!(received_commands.len(), 1);
    let received_command = &received_commands[0];
    assert!(
        received_command.request_id.value() >= 1,
        "the daemon should allocate a request identifier from the shared counter"
    );
    assert_eq!(received_command.model, "test/local-chatter");
    assert_eq!(
        received_command.messages,
        vec![ChatMessage::User {
            content: "Say hello".to_owned(),
            images: vec![],
        }]
    );
    assert_eq!(received_command.settings.max_output_tokens, 128);

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_admit_chat_generation_for_a_model_the_worker_has_not_loaded_yet() {
    // Core fix for CLI auto-load: a model the daemon knows but has not loaded
    // yet must be admitted, not rejected — the worker loop swaps or loads it
    // on demand. The stub's empty stream yields a failure terminal, which is
    // the proof the request passed the gate and reached the executor.
    let state_directory = fresh_instance_state_directory("admit-not-loaded");
    let stub_executor = Arc::new(StubGenerationExecutor {
        health_snapshot: WorkerHealthSnapshot::ready_without_model(0),
        stream_events: vec![],
        received_commands: Mutex::new(Vec::new()),
        embeddings_output: None,
        received_embeddings_commands: Mutex::new(Vec::new()),
    });
    let reloadable_config = Arc::new(RwLock::new(ipc_runtime_config(
        Vec::new(),
        &["test/local-chatter"],
    )));
    let daemon_ipc_service = start_stub_daemon_ipc_service_with_runtime_config(
        &state_directory,
        Arc::clone(&stub_executor),
        Some(reloadable_config),
    )
    .await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&text_only_chat_generate_request("test/local-chatter")),
    )
    .await
    .expect("the chat generate send should finish inside the test timeout")
    .expect("the chat generate request should transmit");
    let terminal_frame = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the terminal frame should arrive inside the test timeout")
        .expect("the terminal frame read should not fail at the transport layer")
        .expect("the daemon should answer the admitted request");
    assert!(
        matches!(&terminal_frame, DaemonResponse::ChatGenerationFailed { .. }),
        "a known model the worker has not loaded yet must be admitted to the \
         executor (the empty stub stream ends in failure, not rejection): {terminal_frame:?}"
    );
    let received_commands = stub_executor
        .received_commands
        .lock()
        .expect("the stub command log lock should not be poisoned");
    assert_eq!(
        received_commands.len(),
        1,
        "the daemon should dispatch the generation to the worker"
    );
    assert_eq!(received_commands[0].model, "test/local-chatter");

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_reject_chat_generation_for_a_model_unknown_to_the_policy_catalog() {
    let state_directory = fresh_instance_state_directory("reject-unknown-model");
    let stub_executor = Arc::new(StubGenerationExecutor {
        health_snapshot: WorkerHealthSnapshot::ready_without_model(0),
        stream_events: vec![],
        received_commands: Mutex::new(Vec::new()),
        embeddings_output: None,
        received_embeddings_commands: Mutex::new(Vec::new()),
    });
    let reloadable_config = Arc::new(RwLock::new(ipc_runtime_config(
        Vec::new(),
        &["test/local-chatter"],
    )));
    let daemon_ipc_service = start_stub_daemon_ipc_service_with_runtime_config(
        &state_directory,
        Arc::clone(&stub_executor),
        Some(reloadable_config),
    )
    .await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&text_only_chat_generate_request("nope/unknown-model")),
    )
    .await
    .expect("the chat generate send should finish inside the test timeout")
    .expect("the chat generate request should transmit");
    let rejection_frame = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the rejection frame should arrive inside the test timeout")
        .expect("the rejection frame read should not fail at the transport layer")
        .expect("the daemon should answer the rejected request");
    assert!(
        matches!(
            &rejection_frame,
            DaemonResponse::GenerationRejected { reason }
                if reason.contains("nope/unknown-model is unknown")
        ),
        "a model unknown to the policy catalog must be rejected with its id named: {rejection_frame:?}"
    );
    let received_commands = stub_executor
        .received_commands
        .lock()
        .expect("the stub command log lock should not be poisoned");
    assert!(
        received_commands.is_empty(),
        "a rejected request must not reach the worker"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_reject_chat_generation_when_the_worker_is_unavailable() {
    let state_directory = fresh_instance_state_directory("reject-unavailable");
    let stub_executor = Arc::new(StubGenerationExecutor {
        health_snapshot: WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
        stream_events: vec![],
        received_commands: Mutex::new(Vec::new()),
        embeddings_output: None,
        received_embeddings_commands: Mutex::new(Vec::new()),
    });
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&text_only_chat_generate_request("test/local-chatter")),
    )
    .await
    .expect("the chat generate send should finish inside the test timeout")
    .expect("the chat generate request should transmit");
    let rejection_frame = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the rejection frame should arrive inside the test timeout")
        .expect("the rejection frame read should not fail at the transport layer")
        .expect("the daemon should answer the rejected request");
    assert!(
        matches!(
            &rejection_frame,
            DaemonResponse::GenerationRejected { reason } if !reason.is_empty()
        ),
        "a chat request against an unavailable worker must be rejected: {rejection_frame:?}"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_allocate_distinct_request_ids_to_consecutive_chat_requests() {
    let state_directory = fresh_instance_state_directory("distinct-request-ids");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, Arc::clone(&stub_executor)).await;

    for connection_round in 0..2 {
        let mut daemon_client =
            DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
                .await
                .expect("the client should connect to the running daemon IPC service");
        timeout(
            HANDSHAKE_TEST_TIMEOUT,
            daemon_client.send_request(&text_only_chat_generate_request("test/local-chatter")),
        )
        .await
        .expect("the chat generate send should finish inside the test timeout")
        .expect("the chat generate request should transmit");
        for _frame_index in 0..3 {
            timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
                .await
                .expect("the stream frame should arrive inside the test timeout")
                .expect("the stream frame read should not fail at the transport layer")
                .unwrap_or_else(|| panic!("round {connection_round} should stream three frames"));
        }
    }

    let received_commands = stub_executor
        .received_commands
        .lock()
        .expect("the stub command log lock should not be poisoned");
    assert_eq!(received_commands.len(), 2);
    let first_request_id = received_commands[0].request_id.value();
    let second_request_id = received_commands[1].request_id.value();
    assert!(
        first_request_id >= 1 && second_request_id >= 1 && first_request_id != second_request_id,
        "consecutive chat requests must receive distinct request identifiers: {first_request_id} and {second_request_id}"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_fill_zero_max_output_tokens_from_the_worker_capabilities() {
    let state_directory = fresh_instance_state_directory("sentinel-max-output");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, Arc::clone(&stub_executor)).await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    let sentinel_request = DaemonRequest::ChatGenerate {
        model: "test/local-chatter".to_owned(),
        messages: vec![ChatMessage::User {
            content: "Say hello".to_owned(),
            images: vec![],
        }],
        settings: ChatGenerationSettings {
            max_output_tokens: 0,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
    };
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&sentinel_request),
    )
    .await
    .expect("the sentinel chat generate send should finish inside the test timeout")
    .expect("the sentinel chat generate request should transmit");
    for _frame_index in 0..3 {
        timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
            .await
            .expect("the sentinel stream frame should arrive inside the test timeout")
            .expect("the sentinel stream frame read should not fail at the transport layer");
    }

    let received_commands = stub_executor
        .received_commands
        .lock()
        .expect("the stub command log lock should not be poisoned");
    assert_eq!(received_commands.len(), 1);
    assert_eq!(
        received_commands[0].settings.max_output_tokens, 128,
        "a zero max_output_tokens sentinel must be filled from the worker-advertised capabilities"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}
