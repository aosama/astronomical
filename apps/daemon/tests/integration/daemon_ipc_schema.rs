//! Schema journeys for the daemon IPC chat surface: a CLI `--schema` file
//! travels as a parsed, canonical worker constraint or is rejected at the
//! daemon trust boundary before the worker ever sees it.

use ipc_protocol::{ChatMessage, DaemonRequest, DaemonResponse, StructuredGenerationConstraint};

use crate::integration::daemon_ipc_exchange::{handshake_then_send_request, read_final_frame};
use crate::integration::daemon_ipc_service_fixtures::{
    say_hello_chat_request, shut_down_and_remove_state_directory, start_ready_chat_service,
};

/// A CLI schema travels through the daemon as a parsed, canonical JSON-schema
/// constraint on the worker command: the daemon owns schema enforcement.
#[tokio::test]
async fn should_forward_a_valid_cli_schema_as_a_structured_generation_command() {
    let (daemon_ipc_service, stub_executor, state_directory) =
        start_ready_chat_service("schema-valid").await;

    // Keys are already in the order serde_json serializes object keys, so the
    // canonical constraint text equals this input text.
    let canonical_schema_text = "{\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"type\":\"object\"}";
    let schema_request =
        chat_generate_with_schema("test/local-chatter", Some(canonical_schema_text.to_owned()));

    let mut request_client =
        handshake_then_send_request(daemon_ipc_service.socket_path(), &schema_request).await;
    let completion_frame = read_final_frame(&mut request_client).await;
    assert!(
        matches!(
            completion_frame,
            Some(DaemonResponse::ChatGenerationCompleted { .. })
        ),
        "a valid schema should pass the daemon gate: {completion_frame:?}"
    );

    let received_commands = stub_executor.logged_commands();
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

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

/// An empty object schema means "any JSON object": the daemon maps it to the
/// JSON-object constraint the workers already enforce, not a schema mask.
#[tokio::test]
async fn should_map_an_empty_object_schema_to_the_json_object_constraint() {
    let (daemon_ipc_service, stub_executor, state_directory) =
        start_ready_chat_service("schema-empty-object").await;

    let schema_request = chat_generate_with_schema("test/local-chatter", Some("{}".to_owned()));

    let mut request_client =
        handshake_then_send_request(daemon_ipc_service.socket_path(), &schema_request).await;
    let completion_frame = read_final_frame(&mut request_client).await;
    assert!(matches!(
        completion_frame,
        Some(DaemonResponse::ChatGenerationCompleted { .. })
    ));

    let received_commands = stub_executor.logged_commands();
    assert_eq!(
        received_commands[0].structured_generation,
        Some(StructuredGenerationConstraint::JsonObject),
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

/// A schema the daemon cannot parse is a request-shape error: the daemon
/// rejects the connection with a user-safe reason and never reaches the
/// worker, regardless of worker health.
#[tokio::test]
async fn should_reject_a_chat_generate_request_with_an_unparseable_schema() {
    let (daemon_ipc_service, stub_executor, state_directory) =
        start_ready_chat_service("schema-unparseable").await;

    let schema_request =
        chat_generate_with_schema("test/local-chatter", Some("{not json".to_owned()));

    let mut request_client =
        handshake_then_send_request(daemon_ipc_service.socket_path(), &schema_request).await;
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

    let received_commands = stub_executor.logged_commands();
    assert!(
        received_commands.is_empty(),
        "an unparseable schema must never reach the worker executor"
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

/// Direct checks of the daemon-side schema validator for the narrow risks
/// the journeys above do not cover: the byte bound, non-object schemas, and
/// canonical re-serialization.
#[test]
fn should_validate_chat_schema_constraints_directly() {
    use astronomical_daemon::daemon_ipc_chat_schema;

    let oversize_schema_text = format!("{{\"values\":[\"{}\"]}}", "x".repeat(70_000));
    let oversize_error =
        daemon_ipc_chat_schema::validated_chat_schema_constraint(&oversize_schema_text)
            .expect_err("oversize should reject");
    assert!(
        oversize_error.contains("65536-byte limit"),
        "the oversize reason should state the bound: {oversize_error}"
    );

    let array_schema_error = daemon_ipc_chat_schema::validated_chat_schema_constraint("[1,2,3]")
        .expect_err("a non-object schema should reject");
    assert!(
        array_schema_error.contains("one JSON object"),
        "the non-object reason should state the object rule: {array_schema_error}"
    );

    let spaced_schema_text = "{ \"type\": \"object\" }";
    let validated_constraint =
        daemon_ipc_chat_schema::validated_chat_schema_constraint(spaced_schema_text)
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
    say_hello_chat_request(model, 128, schema_text)
}
