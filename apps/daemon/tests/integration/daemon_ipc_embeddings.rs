//! Daemon IPC embeddings journeys: routing EmbedGenerate requests to the
//! worker executor and rejecting them when no model is resident.

use astronomical_daemon::{EmbeddingsOutput, WorkerHealthSnapshot};
use ipc_protocol::{DaemonRequest, DaemonResponse, RequestId};

use crate::integration::daemon_ipc_exchange::request_and_read_first_response;
use crate::integration::daemon_ipc_service_fixtures::{
    shut_down_and_remove_state_directory, stub_executor_with_health_snapshot,
};
use crate::support::daemon_ipc::{
    fresh_instance_state_directory, ready_stub_executor_with_embeddings,
    start_stub_daemon_ipc_service,
};

fn hello_embeddings_request() -> DaemonRequest {
    DaemonRequest::EmbedGenerate {
        model: None,
        inputs: vec!["hello".to_owned()],
        dimensions: None,
    }
}

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

    let embeddings_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &hello_embeddings_request(),
    )
    .await;

    assert_eq!(
        embeddings_response,
        DaemonResponse::EmbeddingsCompleted {
            model: "test/local-embedder".to_owned(),
            vectors: vec![vec![0.25, -0.5]],
            input_token_counts: vec![3],
        },
        "an embeddings request without a model must resolve to the resident model and return the worker vectors"
    );

    let received_embeddings_commands = stub_executor.logged_embeddings_commands();
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

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_reject_embeddings_when_no_model_is_resident_and_none_was_requested() {
    let state_directory = fresh_instance_state_directory("embed-no-model");
    let stub_executor =
        stub_executor_with_health_snapshot(WorkerHealthSnapshot::ready_without_model(0));
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, stub_executor.clone()).await;

    let embeddings_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &hello_embeddings_request(),
    )
    .await;

    let DaemonResponse::GenerationRejected { reason } = embeddings_response else {
        panic!("an embeddings request with no resident and no requested model must be rejected");
    };
    assert!(
        reason.contains("no model"),
        "the rejection should tell the caller no model is loaded: {reason}"
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}
