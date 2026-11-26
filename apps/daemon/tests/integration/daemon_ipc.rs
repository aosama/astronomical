//! Daemon IPC chat and status journeys: status probes report the resident
//! model identity, chat requests stream frames, and rejections carry reasons.

use std::path::PathBuf;
use std::sync::Arc;

use astronomical_daemon::{WorkerHealthSnapshot, WorkerHealthStatus};
use ipc_protocol::{
    ChatGenerationCompletionReason, ChatMessage, DaemonRequest, DaemonResponse, ModelLoadInFlight,
};

use crate::integration::daemon_ipc_exchange::{
    connect_and_send_request, connect_client, next_response_or_close, next_response_within_timeout,
    request_and_read_first_response,
};
use crate::integration::daemon_ipc_service_fixtures::{
    ready_idle_status_response, say_hello_chat_request, shut_down_and_remove_state_directory,
    start_ready_chat_service, start_service_with_installed_models,
    stub_executor_with_health_snapshot,
};
use crate::support::daemon_ipc::{
    fresh_instance_state_directory, ready_stub_executor, start_stub_daemon_ipc_service,
    text_only_chat_generate_request,
};

#[tokio::test]
async fn should_report_the_loaded_model_identity_for_status_requests() {
    let (daemon_ipc_service, _stub_executor, state_directory) =
        start_ready_chat_service("status").await;

    let handshake_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &DaemonRequest::Handshake,
    )
    .await;
    assert!(matches!(
        handshake_response,
        DaemonResponse::HandshakeAccepted { .. }
    ));

    // The wire contract serves one request per connection: the CLI performs
    // its handshake, then opens a fresh connection for the status probe.
    let status_response =
        request_and_read_first_response(daemon_ipc_service.socket_path(), &DaemonRequest::Status)
            .await;
    assert_eq!(
        status_response,
        ready_idle_status_response("test/local-chatter", "Qwen3.5-2B-4bit")
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

async fn start_status_test_service(
    test_name: &str,
) -> (astronomical_daemon::DaemonIpcService, PathBuf) {
    let (daemon_ipc_service, _stub_executor, state_directory) =
        start_ready_chat_service(test_name).await;
    (daemon_ipc_service, state_directory)
}

/// A connection that has not finished stands in for a generation holding its
/// connection through a long model load: `status` has to answer anyway, because that
/// is how a waiting `respond` learns the load is still moving.
#[tokio::test]
async fn should_answer_status_while_another_connection_is_still_open() {
    let (daemon_ipc_service, state_directory) = start_status_test_service("open-connection").await;
    let _unfinished_connection = connect_client(daemon_ipc_service.socket_path()).await;

    let status_response =
        request_and_read_first_response(daemon_ipc_service.socket_path(), &DaemonRequest::Status)
            .await;

    assert!(matches!(status_response, DaemonResponse::Status { .. }));
    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_report_a_load_in_flight_on_the_status_answer() {
    let state_directory = fresh_instance_state_directory("loadstatus");
    let mut stub_executor = ready_stub_executor("test/local-chatter");
    let load_report = ModelLoadInFlight::for_model("second-model");
    Arc::get_mut(&mut stub_executor)
        .expect("the stub executor should be uniquely owned here")
        .health_snapshot
        .model_load_in_flight = Some(load_report.clone());
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;
    let status_response =
        request_and_read_first_response(daemon_ipc_service.socket_path(), &DaemonRequest::Status)
            .await;

    match status_response {
        DaemonResponse::Status {
            model_load_in_flight,
            ..
        } => assert_eq!(model_load_in_flight, Some(load_report)),
        unexpected_response => panic!("expected a status answer: {unexpected_response:?}"),
    }
    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_report_the_request_progress_count_on_the_status_answer() {
    let state_directory = fresh_instance_state_directory("progresscount");
    let mut stub_executor = ready_stub_executor("test/local-chatter");
    Arc::get_mut(&mut stub_executor)
        .expect("the stub executor should be uniquely owned here")
        .health_snapshot
        .request_progress_count = 41;
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;
    let status_response =
        request_and_read_first_response(daemon_ipc_service.socket_path(), &DaemonRequest::Status)
            .await;

    match status_response {
        DaemonResponse::Status {
            request_progress_count,
            ..
        } => assert_eq!(request_progress_count, 41),
        unexpected_response => panic!("expected a status answer: {unexpected_response:?}"),
    }
    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

/// A client that sends nothing and hangs up must not cost the next client anything.
#[tokio::test]
async fn should_keep_serving_after_a_client_disconnects_without_a_request() {
    let (daemon_ipc_service, state_directory) =
        start_status_test_service("silent-disconnect").await;
    drop(connect_client(daemon_ipc_service.socket_path()).await);

    let status_response =
        request_and_read_first_response(daemon_ipc_service.socket_path(), &DaemonRequest::Status)
            .await;

    assert!(matches!(status_response, DaemonResponse::Status { .. }));
    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_stream_chat_generation_frames_to_the_cli_client() {
    let (daemon_ipc_service, stub_executor, state_directory) =
        start_ready_chat_service("stream-generate").await;

    let mut daemon_client = connect_and_send_request(
        daemon_ipc_service.socket_path(),
        &text_only_chat_generate_request("test/local-chatter"),
    )
    .await;

    let first_fragment = next_response_within_timeout(&mut daemon_client).await;
    let second_fragment = next_response_within_timeout(&mut daemon_client).await;
    let completion_frame = next_response_within_timeout(&mut daemon_client).await;
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
            astro_ssd_prompt_cache_restored_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        }
    );
    let connection_closed = next_response_or_close(&mut daemon_client).await;
    assert!(
        connection_closed.is_none(),
        "the daemon should close the connection after the terminal frame"
    );

    let received_commands = stub_executor.logged_commands();
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

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_admit_chat_generation_for_a_model_the_worker_has_not_loaded_yet() {
    // Core fix for CLI auto-load: a model the daemon knows but has not loaded
    // yet must be admitted, not rejected — the worker loop swaps or loads it
    // on demand. The stub's empty stream yields a failure terminal, which is
    // the proof the request passed the gate and reached the executor.
    let state_directory = fresh_instance_state_directory("admit-not-loaded");
    let stub_executor =
        stub_executor_with_health_snapshot(WorkerHealthSnapshot::ready_without_model(0));
    let daemon_ipc_service = start_service_with_installed_models(
        &state_directory,
        Arc::clone(&stub_executor),
        Vec::new(),
        &["test/local-chatter"],
    )
    .await;

    let terminal_frame = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &text_only_chat_generate_request("test/local-chatter"),
    )
    .await;
    assert!(
        matches!(&terminal_frame, DaemonResponse::ChatGenerationFailed { .. }),
        "a known model the worker has not loaded yet must be admitted to the \
         executor (the empty stub stream ends in failure, not rejection): {terminal_frame:?}"
    );
    let received_commands = stub_executor.logged_commands();
    assert_eq!(
        received_commands.len(),
        1,
        "the daemon should dispatch the generation to the worker"
    );
    assert_eq!(received_commands[0].model, "test/local-chatter");

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_reject_chat_generation_for_a_model_unknown_to_the_policy_catalog() {
    let state_directory = fresh_instance_state_directory("reject-unknown-model");
    let stub_executor =
        stub_executor_with_health_snapshot(WorkerHealthSnapshot::ready_without_model(0));
    let daemon_ipc_service = start_service_with_installed_models(
        &state_directory,
        Arc::clone(&stub_executor),
        Vec::new(),
        &["test/local-chatter"],
    )
    .await;

    let rejection_frame = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &text_only_chat_generate_request("nope/unknown-model"),
    )
    .await;
    assert!(
        matches!(
            &rejection_frame,
            DaemonResponse::GenerationRejected { reason }
                if reason.contains("nope/unknown-model is unknown")
        ),
        "a model unknown to the policy catalog must be rejected with its id named: {rejection_frame:?}"
    );
    let received_commands = stub_executor.logged_commands();
    assert!(
        received_commands.is_empty(),
        "a rejected request must not reach the worker"
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_reject_chat_generation_when_the_worker_is_unavailable() {
    let state_directory = fresh_instance_state_directory("reject-unavailable");
    let stub_executor = stub_executor_with_health_snapshot(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Unavailable,
    ));
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;

    let rejection_frame = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &text_only_chat_generate_request("test/local-chatter"),
    )
    .await;
    assert!(
        matches!(
            &rejection_frame,
            DaemonResponse::GenerationRejected { reason } if !reason.is_empty()
        ),
        "a chat request against an unavailable worker must be rejected: {rejection_frame:?}"
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_allocate_distinct_request_ids_to_consecutive_chat_requests() {
    let (daemon_ipc_service, stub_executor, state_directory) =
        start_ready_chat_service("distinct-request-ids").await;

    for connection_round in 0..2 {
        let mut daemon_client = connect_and_send_request(
            daemon_ipc_service.socket_path(),
            &text_only_chat_generate_request("test/local-chatter"),
        )
        .await;
        for _frame_index in 0..3 {
            next_response_or_close(&mut daemon_client)
                .await
                .unwrap_or_else(|| panic!("round {connection_round} should stream three frames"));
        }
    }

    let received_commands = stub_executor.logged_commands();
    assert_eq!(received_commands.len(), 2);
    let first_request_id = received_commands[0].request_id.value();
    let second_request_id = received_commands[1].request_id.value();
    assert!(
        first_request_id >= 1 && second_request_id >= 1 && first_request_id != second_request_id,
        "consecutive chat requests must receive distinct request identifiers: {first_request_id} and {second_request_id}"
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_fill_zero_max_output_tokens_from_the_worker_capabilities() {
    let (daemon_ipc_service, stub_executor, state_directory) =
        start_ready_chat_service("sentinel-max-output").await;

    let sentinel_request = say_hello_chat_request("test/local-chatter", 0, None);
    let mut daemon_client =
        connect_and_send_request(daemon_ipc_service.socket_path(), &sentinel_request).await;
    for _frame_index in 0..3 {
        next_response_or_close(&mut daemon_client).await;
    }

    let received_commands = stub_executor.logged_commands();
    assert_eq!(received_commands.len(), 1);
    assert_eq!(
        received_commands[0].settings.max_output_tokens, 128,
        "a zero max_output_tokens sentinel must be filled from the worker-advertised capabilities"
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}
