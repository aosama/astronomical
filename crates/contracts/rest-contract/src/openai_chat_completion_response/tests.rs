use crate::{
    OpenAiAssistantMessage, OpenAiChatCompletionChunk, OpenAiChatCompletionResponse,
    OpenAiFinishReason, OpenAiTokenUsage,
};

#[test]
fn should_serialize_an_openai_compatible_streaming_text_delta() {
    let streaming_chunk = OpenAiChatCompletionChunk::text_delta(
        "chatcmpl-local-42",
        1_784_231_803,
        "astronomical/fake-mixture-of-experts",
        "local fragment",
    );

    assert_eq!(
        serde_json::to_string(&streaming_chunk)
            .expect("the OpenAI-compatible chunk should serialize"),
        r#"{"id":"chatcmpl-local-42","object":"chat.completion.chunk","created":1784231803,"model":"astronomical/fake-mixture-of-experts","choices":[{"index":0,"delta":{"content":"local fragment"},"finish_reason":null}]}"#
    );
}

#[test]
fn should_serialize_a_terminal_tool_call_chunk_with_requested_usage() {
    let token_usage = OpenAiTokenUsage::new(31, 7)
        .expect("small token counts should produce non-overflowing usage");
    let terminal_chunk = OpenAiChatCompletionChunk::finished(
        "chatcmpl-local-42",
        1_784_231_803,
        "astronomical/fake-mixture-of-experts",
        OpenAiFinishReason::ToolCalls,
    )
    .with_usage(token_usage);

    assert_eq!(
        serde_json::to_string(&terminal_chunk)
            .expect("the terminal OpenAI-compatible chunk should serialize"),
        r#"{"id":"chatcmpl-local-42","object":"chat.completion.chunk","created":1784231803,"model":"astronomical/fake-mixture-of-experts","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":31,"completion_tokens":7,"total_tokens":38}}"#
    );
}

#[test]
fn should_serialize_a_non_streaming_response_with_reasoning_and_text() {
    let token_usage = OpenAiTokenUsage::new(31, 7)
        .expect("small token counts should produce non-overflowing usage");
    let response = OpenAiChatCompletionResponse::new(
        "chatcmpl-local-42",
        1_784_231_803,
        "astronomical/fake-mixture-of-experts",
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
        r#"{"id":"chatcmpl-local-42","object":"chat.completion","created":1784231803,"model":"astronomical/fake-mixture-of-experts","choices":[{"index":0,"message":{"role":"assistant","content":"The repository uses Rust.","reasoning_content":"I inspected the source tree."},"finish_reason":"stop"}],"usage":{"prompt_tokens":31,"completion_tokens":7,"total_tokens":38}}"#
    );
}

#[test]
fn should_serialize_cached_tokens_in_usage_when_nonzero() {
    let token_usage = OpenAiTokenUsage::new(4096, 100)
        .expect("small token counts should produce non-overflowing usage")
        .with_astro_ssd_prompt_cache_restored_tokens(2048);

    let serialized = serde_json::to_string(&token_usage)
        .expect("the usage with OpenAI's cached_tokens should serialize");

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
        .expect("the usage without OpenAI's cached_tokens should serialize");

    assert!(
        !serialized.contains("cached_tokens"),
        "expected no cached_tokens in serialized usage: {serialized}"
    );
    assert!(
        !serialized.contains("prompt_tokens_details"),
        "expected no prompt_tokens_details in serialized usage: {serialized}"
    );
}

#[test]
fn should_report_a_draft_models_kept_and_thrown_away_guesses_as_prediction_tokens() {
    let token_usage = OpenAiTokenUsage::new(28, 500)
        .expect("small token counts should produce non-overflowing usage")
        .with_drafted_tokens(310, 290);

    let serialized =
        serde_json::to_string(&token_usage).expect("the usage with drafting should serialize");

    assert!(
        serialized.contains(
            r#""completion_tokens_details":{"accepted_prediction_tokens":310,"rejected_prediction_tokens":290}"#
        ),
        "{serialized}"
    );
}

#[test]
fn should_omit_completion_token_details_for_a_reply_that_never_drafted() {
    let token_usage = OpenAiTokenUsage::new(28, 500)
        .expect("small token counts should produce non-overflowing usage");
    let serialized = serde_json::to_string(&token_usage).expect("the usage should serialize");
    assert!(
        !serialized.contains("completion_tokens_details"),
        "{serialized}"
    );
}
