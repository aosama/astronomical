use super::*;

#[tokio::test]
async fn should_return_semantic_sse_events_without_a_done_sentinel() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::TextFragment("Done.".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 10,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let http_response = post_response(application, true).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let response_body = to_bytes(http_response.into_body(), 32 * 1024)
        .await
        .expect("the Responses stream should be readable");
    let response_text =
        String::from_utf8(response_body.to_vec()).expect("the stream should contain UTF-8");
    assert!(response_text.contains("event: response.created"));
    assert!(response_text.contains("event: response.output_text.delta"));
    assert!(response_text.contains(r#""delta":"Done.""#));
    assert!(response_text.contains("event: response.completed"));
    assert!(!response_text.contains("[DONE]"));
}

#[tokio::test]
async fn should_stream_responses_visible_text_separately_from_reasoning_for_opencode() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::ReasoningFragment("internal thought".to_owned()),
        ChatGenerationStreamEvent::TextFragment("Visible answer".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 10,
            generated_token_count: 2,
            reasoning_token_count: 1,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let http_response = post_response(application, true).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let response_body = to_bytes(http_response.into_body(), 64 * 1024)
        .await
        .expect("the Responses stream should be readable");
    let response_text =
        String::from_utf8(response_body.to_vec()).expect("the stream should contain UTF-8");
    let parsed_stream = ParsedResponsesSseStream::parse(&response_text);

    assert_eq!(
        parsed_stream.event_types(),
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_eq!(parsed_stream.visible_text_for_opencode(), "Visible answer");
    assert_eq!(parsed_stream.reasoning_summary_text(), "internal thought");
    assert!(
        !response_text.contains("[DONE]"),
        "Responses streams use semantic terminal events rather than a chat [DONE] sentinel: {response_text}"
    );
    let completed_response = parsed_stream
        .completed_response()
        .expect("the stream should include a completed response payload");
    assert_eq!(completed_response["status"], "completed");
    assert_eq!(completed_response["output_text"], "Visible answer");
    assert_eq!(completed_response["output"][0]["type"], "reasoning");
    assert_eq!(completed_response["output"][1]["type"], "message");
    assert_eq!(completed_response["usage"]["input_tokens"], 10);
    assert_eq!(completed_response["usage"]["output_tokens"], 2);
    assert_eq!(
        completed_response["usage"]["output_tokens_details"]["reasoning_tokens"],
        1
    );
}

#[tokio::test]
async fn should_stream_responses_parallel_function_call_lifecycle_for_opencode() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"filePath":"README.md"}"#.to_owned(),
        },
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 1,
            function_name: "glob".to_owned(),
            arguments_json: r#"{"pattern":"tests/**/*.rs"}"#.to_owned(),
        },
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 10,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::ToolCalls,
        },
    ]));

    let http_response = post_response(application, true).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let response_body = to_bytes(http_response.into_body(), 64 * 1024)
        .await
        .expect("the Responses stream should be readable");
    let response_text =
        String::from_utf8(response_body.to_vec()).expect("the stream should contain UTF-8");
    let parsed_stream = ParsedResponsesSseStream::parse(&response_text);

    assert_eq!(
        parsed_stream.event_types(),
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let function_call_item = parsed_stream
        .first_payload_for_event_type("response.output_item.added")
        .expect("the stream should add a function call item");
    assert_eq!(function_call_item["item"]["type"], "function_call");
    assert_eq!(function_call_item["item"]["name"], "read");
    assert!(
        function_call_item["item"]["call_id"]
            .as_str()
            .is_some_and(|function_call_id| function_call_id.starts_with("call_")),
        "function calls should expose an OpenAI-style call id: {response_text}"
    );
    assert_eq!(
        parsed_stream
            .first_payload_for_event_type("response.function_call_arguments.delta")
            .expect("the stream should include function arguments delta")["delta"],
        r#"{"filePath":"README.md"}"#
    );
    let completed_response = parsed_stream
        .completed_response()
        .expect("the stream should include a completed response payload");
    assert_eq!(completed_response["output_text"], "");
    assert_eq!(completed_response["output"][0]["type"], "function_call");
    assert_eq!(completed_response["output"][1]["type"], "function_call");
    assert_eq!(completed_response["output"][1]["name"], "glob");
}

#[tokio::test]
async fn should_stream_the_context_overflow_signal_without_assistant_output() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: 262_145,
                maximum_context_tokens: 262_144,
            },
        },
    ]));

    let http_response = post_response(application, true).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let response_body = to_bytes(http_response.into_body(), 32 * 1024)
        .await
        .expect("the context error stream should be readable");
    let response_text = String::from_utf8(response_body.to_vec())
        .expect("the context error stream should contain UTF-8");
    assert!(response_text.contains("event: response.failed"));
    assert!(response_text.contains(r#""code":"context_length_exceeded""#));
    assert!(response_text.contains(r#""status":"failed""#));
    assert!(!response_text.contains("response.output_text.delta"));
    assert!(!response_text.contains("[DONE]"));
}

#[tokio::test]
async fn should_end_a_partially_emitted_stream_with_an_error_if_the_worker_closes() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::TextFragment("Partial".to_owned()),
    ]));

    let http_response = post_response(application, true).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let response_body = to_bytes(http_response.into_body(), 32 * 1024)
        .await
        .expect("the interrupted stream should be readable");
    let response_text =
        String::from_utf8(response_body.to_vec()).expect("the stream should contain UTF-8");
    assert!(response_text.contains("event: response.output_text.delta"));
    assert!(response_text.contains("event: error"));
    assert!(response_text.contains(r#""code":"worker_unavailable""#));
    assert!(!response_text.contains("event: response.completed"));
}
