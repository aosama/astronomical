use super::*;

#[test]
fn should_accept_opencode_large_output_budget_without_a_public_coding_cap() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "inspect the repository"}],
        "stream": true,
        "max_tokens": 20000
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the OpenCode large-output request should deserialize");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("model context validation, not a public coding cap, should bound output tokens");

    assert_eq!(request_parts.maximum_output_tokens, 20_000);
}

#[test]
fn should_accept_large_opencode_chat_history_without_a_public_message_count_cap() {
    let request_messages = (0..250)
        .map(|message_number| {
            json!({
                "role": "user",
                "content": format!("short turn {message_number}"),
            })
        })
        .collect::<Vec<_>>();
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": request_messages,
        "stream": true,
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("the large OpenCode-style chat history should deserialize");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("transport bytes and model context should bound long histories, not message count");

    assert_eq!(request_parts.messages.len(), 250);
}

#[test]
fn should_accept_many_small_text_content_parts_without_a_public_part_count_cap() {
    let content_parts = (0..250)
        .map(|part_number| {
            json!({
                "type": "text",
                "text": format!("part {part_number} "),
            })
        })
        .collect::<Vec<_>>();
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": content_parts}],
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("many small text parts should deserialize");

    let request_parts = chat_completion_request.into_parts().expect(
        "transport bytes and text-only validation should bound content parts, not part count",
    );

    assert!(matches!(
        request_parts.messages.as_slice(),
        [OpenAiChatMessageParts::User { content, .. }] if content.contains("part 249")
    ));
}

#[test]
fn should_accept_many_small_tool_definitions_without_a_public_tool_count_cap() {
    let tools = (0..250)
        .map(|tool_number| {
            json!({
                "type": "function",
                "function": {
                    "name": format!("tool_{tool_number}"),
                    "parameters": {"type": "object"},
                }
            })
        })
        .collect::<Vec<_>>();
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "use a tool"}],
        "tools": tools,
        "tool_choice": "auto",
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("many small tool definitions should deserialize");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("transport bytes and schema bytes should bound tools, not tool count");

    assert_eq!(request_parts.tools.len(), 250);
}

#[test]
fn should_accept_large_assistant_tool_call_arguments_without_a_public_field_byte_cap() {
    let large_arguments_json = format!(r#"{{"payload":"{}"}}"#, "x".repeat(80 * 1024));
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_large_arguments",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": large_arguments_json,
                }
            }]
        }]
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("large assistant tool-call arguments should deserialize");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("request body bytes should bound assistant tool-call arguments, not a field cap");

    assert!(matches!(
        request_parts.messages.as_slice(),
        [OpenAiChatMessageParts::Assistant { tool_calls, .. }]
            if tool_calls[0].arguments_json.contains("payload")
    ));
}

#[test]
fn should_accept_a_single_text_message_larger_than_the_old_public_message_byte_limit() {
    let large_message_content = "x".repeat(128 * 1024);
    let request_json = format!(
        r#"{{
            "model": "astronomical/fake-mixture-of-experts",
            "messages": [{{"role": "user", "content": {}}}],
            "stream": true
        }}"#,
        serde_json::to_string(&large_message_content)
            .expect("the large message content should serialize")
    );
    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("the large text request should deserialize");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("large text prompts should be bounded by transport and model constraints");

    assert!(matches!(
        request_parts.messages.as_slice(),
        [OpenAiChatMessageParts::User { content, .. }] if content == &large_message_content
    ));
}

#[test]
fn should_accept_large_ignored_reasoning_effort_without_a_public_field_byte_cap() {
    let oversized_reasoning_effort = "x".repeat(8 * 1024);
    let request_json = format!(
        r#"{{
            "model": "astronomical/fake-mixture-of-experts",
            "messages": [{{"role": "user", "content": "Inspect the repository."}}],
            "reasoning_effort": {}
        }}"#,
        serde_json::to_string(&oversized_reasoning_effort)
            .expect("the oversized reasoning effort label should serialize")
    );
    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("reasoning_effort should decode before bounded validation");

    chat_completion_request
        .validate()
        .expect("request body bytes should bound ignored labels, not a field cap");
}
