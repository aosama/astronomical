//! Schema journeys for the daemon IPC chat surface: a CLI `--schema` file
//! travels as a parsed, canonical worker constraint or is rejected at the
//! daemon trust boundary before the worker ever sees it.

use std::sync::Arc;

use astronomical_ipc_protocol::{
    ChatGenerationSettings, ChatMessage, DaemonIpcClient, DaemonRequest, DaemonResponse,
    StructuredGenerationConstraint,
};
use tokio::time::timeout;

use crate::common::daemon_ipc::{
    HANDSHAKE_TEST_TIMEOUT, fresh_instance_state_directory, ready_stub_executor,
    start_stub_daemon_ipc_service,
};

/// A CLI schema travels through the daemon as a parsed, canonical JSON-schema
/// constraint on the worker command: the daemon owns schema enforcement.
#[tokio::test]
async fn should_forward_a_valid_cli_schema_as_a_structured_generation_command() {
    let state_directory = fresh_instance_state_directory("schema-valid");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, Arc::clone(&stub_executor)).await;

    // Keys are already in the order serde_json serializes object keys, so the
    // canonical constraint text equals this input text.
    let canonical_schema_text = "{\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"type\":\"object\"}";
    let schema_request =
        chat_generate_with_schema("test/local-chatter", Some(canonical_schema_text.to_owned()));

    let mut request_client =
        connect_handshake_and_send(daemon_ipc_service.socket_path(), &schema_request).await;
    let completion_frame = read_final_frame(&mut request_client).await;
    assert!(
        matches!(
            completion_frame,
            Some(DaemonResponse::ChatGenerationCompleted { .. })
        ),
        "a valid schema should pass the daemon gate: {completion_frame:?}"
    );

    let received_commands = stub_executor
        .received_commands
        .lock()
        .expect("the stub command log lock should not be poisoned");
    assert_eq!(received_commands.len(), 1);
    assert_eq!(
        received_commands[0].structured_generation,
        Some(StructuredGenerationConstraint::JsonSchema {
            schema_json: canonical_schema_text.to_owned(),
        }),
        "the daemon must parse the schema text and send the worker a canonical constraint"
    );
    // Enforcement pairs with the JSON-output system instruction; otherwise
    // the masked model does not plan a JSON object answer.
    let Some(ChatMessage::System { content }) = received_commands[0].messages.first() else {
        panic!("the schema instruction must arrive as the first message");
    };
    assert!(
        content.contains(canonical_schema_text),
        "the schema instruction must carry the schema text: {content}"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

/// An empty object schema means "any JSON object": the daemon maps it to the
/// JSON-object constraint the workers already enforce, not a schema mask.
#[tokio::test]
async fn should_map_an_empty_object_schema_to_the_json_object_constraint() {
    let state_directory = fresh_instance_state_directory("schema-empty-object");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, Arc::clone(&stub_executor)).await;

    let schema_request = chat_generate_with_schema("test/local-chatter", Some("{}".to_owned()));

    let mut request_client =
        connect_handshake_and_send(daemon_ipc_service.socket_path(), &schema_request).await;
    let completion_frame = read_final_frame(&mut request_client).await;
    assert!(matches!(
        completion_frame,
        Some(DaemonResponse::ChatGenerationCompleted { .. })
    ));

    let received_commands = stub_executor
        .received_commands
        .lock()
        .expect("the stub command log lock should not be poisoned");
    assert_eq!(
        received_commands[0].structured_generation,
        Some(StructuredGenerationConstraint::JsonObject),
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

/// A schema the daemon cannot parse is a request-shape error: the daemon
/// rejects the connection with a user-safe reason and never reaches the
/// worker, regardless of worker health.
#[tokio::test]
async fn should_reject_a_chat_generate_request_with_an_unparseable_schema() {
    let state_directory = fresh_instance_state_directory("schema-unparseable");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, Arc::clone(&stub_executor)).await;

    let schema_request =
        chat_generate_with_schema("test/local-chatter", Some("{not json".to_owned()));

    let mut request_client =
        connect_handshake_and_send(daemon_ipc_service.socket_path(), &schema_request).await;
    let rejected_frame = read_final_frame(&mut request_client).await;
    match rejected_frame {
        Some(DaemonResponse::GenerationRejected { reason }) => {
            assert!(
                reason.contains("not valid JSON"),
                "the rejection reason should name the malformed schema: {reason}"
            );
        }
        other => panic!("expected a generation rejection, got {other:?}"),
    }

    let received_commands = stub_executor
        .received_commands
        .lock()
        .expect("the stub command log lock should not be poisoned");
    assert!(
        received_commands.is_empty(),
        "an unparseable schema must never reach the worker executor"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

/// Direct checks of the daemon-side schema validator for the narrow risks
/// the journeys above do not cover: the byte bound, non-object schemas, and
/// canonical re-serialization.
#[test]
fn should_validate_chat_schema_constraints_directly() {
    use astronomical_supervisor::daemon_ipc_chat_schema::validated_chat_schema_constraint;

    let oversize_schema_text = format!("{{\"values\":[\"{}\"]}}", "x".repeat(70_000));
    let oversize_error = validated_chat_schema_constraint(&oversize_schema_text)
        .expect_err("oversize should reject");
    assert!(
        oversize_error.contains("65536-byte limit"),
        "the oversize reason should state the bound: {oversize_error}"
    );

    let array_schema_error =
        validated_chat_schema_constraint("[1,2,3]").expect_err("a non-object schema should reject");
    assert!(
        array_schema_error.contains("one JSON object"),
        "the non-object reason should state the object rule: {array_schema_error}"
    );

    let spaced_schema_text = "{ \"type\": \"object\" }";
    let validated_constraint = validated_chat_schema_constraint(spaced_schema_text)
        .expect("a valid object schema should accept");
    assert_eq!(
        validated_constraint,
        StructuredGenerationConstraint::JsonSchema {
            schema_json: "{\"type\":\"object\"}".to_owned(),
        },
        "the daemon should re-serialize the schema canonically"
    );
}

/// Builds a chat generate request that carries the given schema text, for
/// the schema journeys; everything else matches the shared text-only builder.
fn chat_generate_with_schema(model: &str, schema_text: Option<String>) -> DaemonRequest {
    DaemonRequest::ChatGenerate {
        model: model.to_owned(),
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
        schema_json: schema_text,
    }
}

/// Performs the handshake on one connection, then sends the request on a
/// fresh connection per the wire contract, and returns the request client
/// for the caller to read the streamed frames.
async fn connect_handshake_and_send(
    socket_path: &std::path::Path,
    request: &DaemonRequest,
) -> DaemonIpcClient {
    let mut handshake_client = DaemonIpcClient::connect(socket_path.to_path_buf())
        .await
        .expect("the handshake client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        handshake_client.send_request(&DaemonRequest::Handshake),
    )
    .await
    .expect("the handshake send should finish inside the test timeout")
    .expect("the handshake request should transmit");
    let _handshake_accepted = timeout(HANDSHAKE_TEST_TIMEOUT, handshake_client.next_response())
        .await
        .expect("the handshake response should arrive inside the test timeout")
        .expect("the handshake response read should not fail at the transport layer")
        .expect("the daemon should answer the handshake");
    drop(handshake_client);

    let mut request_client = DaemonIpcClient::connect(socket_path.to_path_buf())
        .await
        .expect("the request client should connect to the running daemon IPC service");
    timeout(HANDSHAKE_TEST_TIMEOUT, request_client.send_request(request))
        .await
        .expect("the request send should finish inside the test timeout")
        .expect("the request should transmit");
    request_client
}

/// Reads frames until the daemon closes the connection and returns the final
/// frame: the completion for an accepted journey, or the single rejection.
async fn read_final_frame(daemon_client: &mut DaemonIpcClient) -> Option<DaemonResponse> {
    let mut final_frame = None;
    loop {
        let next_frame = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
            .await
            .expect("the response frame should arrive inside the test timeout")
            .expect("the response frame read should not fail at the transport layer");
        match next_frame {
            Some(frame) => final_frame = Some(frame),
            None => return final_frame,
        }
    }
}
