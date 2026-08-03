use astronomical_rest_contract::{
    OpenAiAssistantMessage, OpenAiChatCompletionChunk, OpenAiChatCompletionResponse,
    OpenAiFinishReason, OpenAiTokenUsage,
};

#[test]
fn should_serialize_an_openai_compatible_streaming_text_delta() {
    let streaming_chunk = OpenAiChatCompletionChunk::text_delta(
        "chatcmpl-local-42",
        1_784_231_803,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
        "local fragment",
    );

    assert_eq!(
        serde_json::to_string(&streaming_chunk)
            .expect("the OpenAI-compatible chunk should serialize"),
        r#"{"id":"chatcmpl-local-42","object":"chat.completion.chunk","created":1784231803,"model":"mlx-community/Ornith-1.0-35B-OptiQ-4bit","choices":[{"index":0,"delta":{"content":"local fragment"},"finish_reason":null}]}"#
    );
}

#[test]
fn should_serialize_a_terminal_tool_call_chunk_with_requested_usage() {
    let token_usage = OpenAiTokenUsage::new(31, 7)
        .expect("small token counts should produce non-overflowing usage");
    let terminal_chunk = OpenAiChatCompletionChunk::finished(
        "chatcmpl-local-42",
        1_784_231_803,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
        OpenAiFinishReason::ToolCalls,
    )
    .with_usage(token_usage);

    assert_eq!(
        serde_json::to_string(&terminal_chunk)
            .expect("the terminal OpenAI-compatible chunk should serialize"),
        r#"{"id":"chatcmpl-local-42","object":"chat.completion.chunk","created":1784231803,"model":"mlx-community/Ornith-1.0-35B-OptiQ-4bit","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":31,"completion_tokens":7,"total_tokens":38}}"#
    );
}

#[test]
fn should_serialize_a_non_streaming_response_with_reasoning_and_text() {
    let token_usage = OpenAiTokenUsage::new(31, 7)
        .expect("small token counts should produce non-overflowing usage");
    let response = OpenAiChatCompletionResponse::new(
        "chatcmpl-local-42",
        1_784_231_803,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
        OpenAiAssistantMessage::new(
            Some("The repository uses Rust.".to_owned()),
            Some("I inspected the source tree.".to_owned()),
            Vec::new(),
        ),
        OpenAiFinishReason::Stop,
        token_usage,
    );

    assert_eq!(
        serde_json::to_string(&response)
            .expect("the non-streaming OpenAI-compatible response should serialize"),
        r#"{"id":"chatcmpl-local-42","object":"chat.completion","created":1784231803,"model":"mlx-community/Ornith-1.0-35B-OptiQ-4bit","choices":[{"index":0,"message":{"role":"assistant","content":"The repository uses Rust.","reasoning_content":"I inspected the source tree."},"finish_reason":"stop"}],"usage":{"prompt_tokens":31,"completion_tokens":7,"total_tokens":38}}"#
    );
}

#[test]
fn should_serialize_cached_tokens_in_usage_when_nonzero() {
    let token_usage = OpenAiTokenUsage::new(4096, 100)
        .expect("small token counts should produce non-overflowing usage")
        .with_cached_tokens(2048);

    let serialized =
        serde_json::to_string(&token_usage).expect("the usage with cached tokens should serialize");

    assert!(
        serialized.contains(r#""prompt_tokens":4096"#),
        "expected prompt_tokens in serialized usage: {serialized}"
    );
    assert!(
        serialized.contains(r#""completion_tokens":100"#),
        "expected completion_tokens in serialized usage: {serialized}"
    );
    assert!(
        serialized.contains(r#""prompt_tokens_details":{"cached_tokens":2048}"#),
        "expected prompt_tokens_details.cached_tokens in serialized usage: {serialized}"
    );
}

#[test]
fn should_omit_cached_tokens_from_usage_when_zero() {
    let token_usage = OpenAiTokenUsage::new(31, 7)
        .expect("small token counts should produce non-overflowing usage");

    let serialized = serde_json::to_string(&token_usage)
        .expect("the usage without cached tokens should serialize");

    assert!(
        !serialized.contains("cached_tokens"),
        "expected no cached_tokens in serialized usage: {serialized}"
    );
    assert!(
        !serialized.contains("prompt_tokens_details"),
        "expected no prompt_tokens_details in serialized usage: {serialized}"
    );
}
