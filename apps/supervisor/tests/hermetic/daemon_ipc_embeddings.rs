//! Daemon IPC embeddings journeys: routing EmbedGenerate requests to the
//! worker executor and rejecting them when no model is resident.

use std::sync::{Arc, Mutex};

use astronomical_ipc_protocol::{DaemonIpcClient, DaemonRequest, DaemonResponse, RequestId};
use astronomical_supervisor::{EmbeddingsOutput, WorkerHealthSnapshot};
use tokio::time::timeout;

use crate::common::daemon_ipc::{
    HANDSHAKE_TEST_TIMEOUT, StubGenerationExecutor, fresh_instance_state_directory,
    ready_stub_executor_with_embeddings, start_stub_daemon_ipc_service,
};

#[tokio::test]
async fn should_route_embeddings_requests_to_the_executor_and_return_the_vectors() {
    let state_directory = fresh_instance_state_directory("embed-routed");
    let stub_executor = ready_stub_executor_with_embeddings(
        "test/local-embedder",
        Ok(EmbeddingsOutput {
            embeddings: vec![vec![0.25, -0.5]],
            input_token_counts: vec![3],
            elapsed_millis: 12,
        }),
    );
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, stub_executor.clone()).await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    let embeddings_request = DaemonRequest::EmbedGenerate {
        model: None,
        inputs: vec!["hello".to_owned()],
        dimensions: None,
    };
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&embeddings_request),
    )
    .await
    .expect("the embeddings request send should finish inside the test timeout")
    .expect("the embeddings request should transmit");
    let embeddings_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the embeddings response should arrive inside the test timeout")
        .expect("the embeddings response read should not fail at the transport layer")
        .expect("the daemon should answer the embeddings request");

    assert_eq!(
        embeddings_response,
        DaemonResponse::EmbeddingsCompleted {
            model: "test/local-embedder".to_owned(),
            vectors: vec![vec![0.25, -0.5]],
            input_token_counts: vec![3],
        },
        "an embeddings request without a model must resolve to the resident model and return the worker vectors"
    );

    let received_embeddings_commands = stub_executor
        .received_embeddings_commands
        .lock()
        .expect("the stub embeddings command log lock should not be poisoned");
    assert_eq!(received_embeddings_commands.len(), 1);
    assert_eq!(
        received_embeddings_commands[0].model, "test/local-embedder",
        "the daemon must resolve the resident model before dispatch"
    );
    assert_eq!(
        received_embeddings_commands[0].inputs,
        vec!["hello".to_owned()]
    );
    assert_eq!(
        received_embeddings_commands[0].request_id,
        RequestId::new(1)
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_reject_embeddings_when_no_model_is_resident_and_none_was_requested() {
    let state_directory = fresh_instance_state_directory("embed-no-model");
    let stub_executor = Arc::new(StubGenerationExecutor {
        health_snapshot: WorkerHealthSnapshot::ready_without_model(0),
        stream_events: vec![],
        received_commands: Mutex::new(Vec::new()),
        embeddings_output: None,
        received_embeddings_commands: Mutex::new(Vec::new()),
    });
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, stub_executor.clone()).await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    let embeddings_request = DaemonRequest::EmbedGenerate {
        model: None,
        inputs: vec!["hello".to_owned()],
        dimensions: None,
    };
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&embeddings_request),
    )
    .await
    .expect("the embeddings request send should finish inside the test timeout")
    .expect("the embeddings request should transmit");
    let embeddings_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the embeddings response should arrive inside the test timeout")
        .expect("the embeddings response read should not fail at the transport layer")
        .expect("the daemon should answer the embeddings request");

    let DaemonResponse::GenerationRejected { reason } = embeddings_response else {
        panic!("an embeddings request with no resident and no requested model must be rejected");
    };
    assert!(
        reason.contains("no model"),
        "the rejection should tell the caller no model is loaded: {reason}"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}
