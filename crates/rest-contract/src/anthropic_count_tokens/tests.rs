use serde_json::json;

use crate::{
    AnthropicCountTokensRequest, AnthropicMessageParts, AnthropicMessagesRequestParts,
    AnthropicMessagesValidationError, AnthropicTokenCount,
};

fn parts_of(request_document: serde_json::Value) -> AnthropicMessagesRequestParts {
    serde_json::from_value::<AnthropicCountTokensRequest>(request_document)
        .expect("the count request should deserialize")
        .into_parts()
        .expect("the count request should validate")
}

fn rejection_of(request_document: serde_json::Value) -> AnthropicMessagesValidationError {
    serde_json::from_value::<AnthropicCountTokensRequest>(request_document)
        .expect("the count request should deserialize")
        .into_parts()
        .expect_err("the count request should be rejected")
}

fn minimal_count_request() -> serde_json::Value {
    json!({
        "model": "local-model",
        "messages": [{"role": "user", "content": "Shall I compare thee to a summer's day?"}]
    })
}

#[test]
fn should_accept_a_count_request_without_max_tokens() {
    let request_parts = parts_of(minimal_count_request());

    assert_eq!(request_parts.model, "local-model");
    assert_eq!(request_parts.maximum_output_tokens, 1);
    assert!(!request_parts.stream);
    assert!(matches!(
        &request_parts.messages[0],
        AnthropicMessageParts::User { .. }
    ));
}

#[test]
fn should_carry_the_system_prompt_tools_and_thinking_that_shape_the_prompt() {
    let mut request_document = minimal_count_request();
    request_document["system"] = json!("Speak as Prospero.");
    request_document["tools"] = json!([{"name": "read_play", "input_schema": {"type": "object"}}]);
    request_document["thinking"] = json!({"type": "disabled"});
    request_document["context_management"] = json!({"edits": []});

    let request_parts = parts_of(request_document);

    assert_eq!(request_parts.system.as_deref(), Some("Speak as Prospero."));
    assert_eq!(request_parts.tools[0].name, "read_play");
    assert_eq!(request_parts.thinking_budget, Some(0));
}

#[test]
fn should_reject_generation_controls_a_count_request_does_not_take() {
    for (field_name, field_value) in [
        ("max_tokens", json!(64)),
        ("stream", json!(true)),
        ("temperature", json!(0.5)),
    ] {
        let mut request_document = minimal_count_request();
        request_document[field_name] = field_value;

        assert_eq!(
            rejection_of(request_document),
            AnthropicMessagesValidationError::UnknownField {
                field_name: field_name.to_owned()
            }
        );
    }
}

#[test]
fn should_reject_empty_messages_and_an_empty_model() {
    let mut request_document = minimal_count_request();
    request_document["messages"] = json!([]);
    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::EmptyMessages
    );

    let mut request_document = minimal_count_request();
    request_document["model"] = json!("");
    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::EmptyModel
    );
}

#[test]
fn should_apply_the_messages_content_rules_to_a_count_request() {
    let mut request_document = minimal_count_request();
    request_document["messages"] = json!([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "opaque"}]}
    ]);

    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedReasoningReplay
    );
}

#[test]
fn should_serialize_the_count_as_input_tokens() {
    let count_document = serde_json::to_value(AnthropicTokenCount { input_tokens: 0 })
        .expect("the count should serialize");

    assert_eq!(count_document, json!({"input_tokens": 0}));
}
