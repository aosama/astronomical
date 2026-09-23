//! Contract coverage for replayed history that carries provider-specific fields.
//! Mainstream harnesses resend messages produced by other providers, so unknown
//! fields must degrade to ignored instead of rejecting the whole request (#772).

use super::*;

#[test]
fn should_ignore_unknown_fields_on_replayed_history_messages() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "system", "content": "Be helpful.", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "What is the weather?", "metadata": {"source": "web"}},
            {
                "role": "assistant",
                "content": "I cannot check the weather.",
                "finish_reason": "content_filter",
                "provider_specific": true
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "sunny",
                "executed_at": "2026-01-01T00:00:00Z"
            }
        ]
    }
    "#;

    let request_parts = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("history messages carrying unknown provider fields should decode")
        .into_parts()
        .expect("unknown history fields must be ignored, not rejected");

    assert!(matches!(
        request_parts.messages.first(),
        Some(OpenAiChatMessageParts::System { content }) if content == "Be helpful."
    ));
    assert!(matches!(
        request_parts.messages.last(),
        Some(OpenAiChatMessageParts::Tool { content, .. }) if content == "sunny"
    ));
}

#[test]
fn should_preserve_an_assistant_refusal_as_message_content() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Ask the upstream model something"},
            {"role": "assistant", "refusal": "I'm sorry, but I can't help with that.", "finish_reason": "content_filter"}
        ]
    }
    "#;

    let request_parts = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("an assistant refusal message should decode")
        .into_parts()
        .expect("a refusal-only assistant message must not be treated as empty");

    assert!(matches!(
        request_parts.messages.last(),
        Some(OpenAiChatMessageParts::Assistant { content, .. })
            if content.as_deref() == Some("I'm sorry, but I can't help with that.")
    ));
}

#[test]
fn should_prefer_explicit_content_over_refusal_on_an_assistant_message() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "the visible answer", "refusal": "a hidden refusal"}
        ]
    }
    "#;

    let request_parts = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("an assistant message with both content and refusal should decode")
        .into_parts()
        .expect("content plus refusal must validate");

    assert!(matches!(
        request_parts.messages.last(),
        Some(OpenAiChatMessageParts::Assistant { content, .. })
            if content.as_deref() == Some("the visible answer")
    ));
}

#[test]
fn should_preserve_a_refusal_content_part_as_message_text() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "before "},
                    {"type": "refusal", "refusal": "and a refusal"}
                ]
            }
        ]
    }
    "#;

    let request_parts = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("a refusal content part should decode")
        .into_parts()
        .expect("refusal content parts must fold into message text");

    assert!(matches!(
        request_parts.messages.last(),
        Some(OpenAiChatMessageParts::Assistant { content, .. })
            if content.as_deref() == Some("before and a refusal")
    ));
}

#[test]
fn should_ignore_unknown_fields_inside_stream_options_and_history_tool_calls() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"},
                    "provider_marker": true
                }]
            }
        ],
        "stream": true,
        "stream_options": {"include_usage": true, "verbose": false}
    }
    "#;

    let request_parts = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("stream options and history tool calls with unknown fields should decode")
        .into_parts()
        .expect("unknown stream-option and tool-call fields must be ignored");

    assert!(request_parts.includes_usage_in_stream);
    assert!(matches!(
        request_parts.messages.last(),
        Some(OpenAiChatMessageParts::Assistant { tool_calls, .. })
            if tool_calls.first().is_some_and(|tool_call| tool_call.name == "bash")
    ));
}

#[test]
fn should_ignore_an_unknown_field_on_an_image_url_object() {
    let image_data_uri = "data:image/png;base64,iVBORw0KGgo=";
    let request_json = format!(
        r#"{{
            "model": "astronomical/fake-mixture-of-experts",
            "messages": [{{
                "role": "user",
                "content": [
                    {{
                        "type": "image_url",
                        "image_url": {{"url": "{image_data_uri}", "detail": "auto"}}
                    }}
                ]
            }}]
        }}"#
    );

    let request_parts = serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
        .expect("an image_url object carrying provider-specific fields should decode")
        .into_parts()
        .expect("unknown image_url fields must be ignored");

    assert!(matches!(
        request_parts.messages.first(),
        Some(OpenAiChatMessageParts::User { images, .. }) if images.len() == 1
    ));
}

#[test]
fn should_reject_an_unknown_message_role() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "critic", "content": "still unknown"}
        ]
    }
    "#;

    let deserialization_error = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect_err("role is the message discriminator and must stay strict");

    assert!(
        deserialization_error
            .to_string()
            .contains("unknown variant `critic`"),
        "an unknown role must be reported through the tag error, got: {deserialization_error}"
    );
}

#[test]
fn should_accept_unknown_top_level_request_fields() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "logit_bias": {"13": -100},
        "service_tier": "auto"
    }
    "#;

    serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("unknown top-level fields should decode")
        .validate()
        .expect("unknown top-level request fields must be ignored, not rejected");
}
