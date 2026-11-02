use super::*;

#[tokio::test]
async fn should_return_a_whole_message_with_thinking_text_and_split_usage() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::ReasoningFragment("A sonnet opens softly.".to_owned()),
        ChatGenerationStreamEvent::TextFragment("Thou art more lovely".to_owned()),
        completed_event(ChatGenerationCompletionReason::EndOfSequence),
    ]));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let message_document = response_json(http_response).await;
    assert_eq!(message_document["type"], "message");
    assert_eq!(message_document["role"], "assistant");
    assert_eq!(message_document["model"], MODEL_ID);
    assert!(
        message_document["id"]
            .as_str()
            .unwrap_or_default()
            .starts_with("msg_")
    );
    assert_eq!(message_document["content"][0]["type"], "thinking");
    assert_eq!(
        message_document["content"][1]["text"],
        "Thou art more lovely"
    );
    assert_eq!(message_document["stop_reason"], "end_turn");
    assert_eq!(message_document["stop_sequence"], Value::Null);
    assert_eq!(message_document["usage"]["input_tokens"], 8);
    assert_eq!(message_document["usage"]["cache_read_input_tokens"], 32);
    assert_eq!(message_document["usage"]["output_tokens"], 6);
}

#[tokio::test]
async fn should_return_tool_use_blocks_with_parsed_input() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "read_sonnet".to_owned(),
            arguments_json: r#"{"number":18}"#.to_owned(),
        },
        completed_event(ChatGenerationCompletionReason::ToolCalls),
    ]));
    let request_body = format!(
        r#"{{"model":"{MODEL_ID}","max_tokens":256,"tools":[{{"name":"read_sonnet","input_schema":{{"type":"object"}}}}],"messages":[{{"role":"user","content":"Read sonnet 18."}}]}}"#
    );

    let http_response = post_messages_body(application, &request_body).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let message_document = response_json(http_response).await;
    let tool_use_block = &message_document["content"][0];
    assert_eq!(tool_use_block["type"], "tool_use");
    assert_eq!(tool_use_block["name"], "read_sonnet");
    assert_eq!(tool_use_block["input"]["number"], 18);
    assert!(
        tool_use_block["id"]
            .as_str()
            .unwrap_or_default()
            .starts_with("toolu_")
    );
    assert_eq!(message_document["stop_reason"], "tool_use");
}

#[tokio::test]
async fn should_report_max_tokens_when_the_output_budget_runs_out() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::TextFragment("Rough winds do shake".to_owned()),
        completed_event(ChatGenerationCompletionReason::MaximumOutputTokens),
    ]));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    let message_document = response_json(http_response).await;
    assert_eq!(message_document["stop_reason"], "max_tokens");
}
