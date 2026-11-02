use super::*;

async fn thinking_then_text_stream() -> Vec<ParsedMessagesSseEvent> {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::ReasoningFragment("Consider the summer.".to_owned()),
        ChatGenerationStreamEvent::TextFragment("Thou art ".to_owned()),
        ChatGenerationStreamEvent::TextFragment("more temperate".to_owned()),
        completed_event(ChatGenerationCompletionReason::EndOfSequence),
    ]));
    let http_response = post_messages_body(application, &messages_request_body(true)).await;
    assert_eq!(http_response.status(), StatusCode::OK);
    parsed_sse_events(http_response).await
}

#[tokio::test]
async fn should_stream_the_anthropic_event_sequence_with_matching_event_names() {
    let parsed_events = thinking_then_text_stream().await;

    assert_eq!(
        event_types(&parsed_events),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    for parsed_event in &parsed_events {
        assert_eq!(
            parsed_event.json_payload["type"],
            parsed_event.event_type.as_str()
        );
    }
}

#[tokio::test]
async fn should_stream_block_kinds_and_the_final_stop_reason_with_usage() {
    let parsed_events = thinking_then_text_stream().await;

    assert_eq!(
        parsed_events[0].json_payload["message"]["content"],
        serde_json::json!([])
    );
    assert_eq!(
        parsed_events[1].json_payload["content_block"]["type"],
        "thinking"
    );
    assert_eq!(parsed_events[5].json_payload["delta"]["type"], "text_delta");
    assert_eq!(
        parsed_events[8].json_payload["delta"]["stop_reason"],
        "end_turn"
    );
    assert_eq!(parsed_events[8].json_payload["usage"]["output_tokens"], 6);
}

#[tokio::test]
async fn should_stream_a_tool_use_block_with_its_whole_input_json() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "read_sonnet".to_owned(),
            arguments_json: r#"{"number":18}"#.to_owned(),
        },
        completed_event(ChatGenerationCompletionReason::ToolCalls),
    ]));
    let request_body = format!(
        r#"{{"model":"{MODEL_ID}","max_tokens":256,"stream":true,"tools":[{{"name":"read_sonnet","input_schema":{{"type":"object"}}}}],"messages":[{{"role":"user","content":"Read sonnet 18."}}]}}"#
    );

    let http_response = post_messages_body(application, &request_body).await;

    let parsed_events = parsed_sse_events(http_response).await;
    let tool_use_start = &parsed_events[1].json_payload["content_block"];
    assert_eq!(tool_use_start["type"], "tool_use");
    assert_eq!(tool_use_start["input"], serde_json::json!({}));
    assert_eq!(
        parsed_events[2].json_payload["delta"]["type"],
        "input_json_delta"
    );
    assert_eq!(
        parsed_events[2].json_payload["delta"]["partial_json"],
        r#"{"number":18}"#
    );
    assert_eq!(
        parsed_events[4].json_payload["delta"]["stop_reason"],
        "tool_use"
    );
}

#[tokio::test]
async fn should_end_an_interrupted_stream_with_an_error_event() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::TextFragment("Partial".to_owned()),
    ]));

    let http_response = post_messages_body(application, &messages_request_body(true)).await;

    let parsed_events = parsed_sse_events(http_response).await;
    let final_event = parsed_events
        .last()
        .expect("the stream should carry events");
    assert_eq!(final_event.event_type, "error");
    assert_eq!(
        final_event.json_payload["error"]["type"],
        "overloaded_error"
    );
    assert!(!event_types(&parsed_events).contains(&"message_stop"));
}

#[tokio::test]
async fn should_stream_context_overflow_as_prompt_too_long() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: 262_145,
                maximum_context_tokens: 262_144,
            },
        },
    ]));

    let http_response = post_messages_body(application, &messages_request_body(true)).await;

    let parsed_events = parsed_sse_events(http_response).await;
    assert_eq!(event_types(&parsed_events), vec!["message_start", "error"]);
    assert_eq!(
        parsed_events[1].json_payload["error"]["message"],
        "prompt is too long: 262145 tokens > 262144 maximum"
    );
}
