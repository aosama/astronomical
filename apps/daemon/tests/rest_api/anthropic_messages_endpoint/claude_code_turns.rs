//! Request shapes a real Claude Code session sends, rebuilt without any of its
//! identifiers, so a regression in what the endpoint accepts from Claude Code
//! fails here instead of in a live session.

use super::*;

/// Claude Code's second turn after loading a deferred tool: the tool round
/// trip, then a system turn that defines the new tool and references it.
fn second_turn_with_a_deferred_tool_definition() -> String {
    let request_document = serde_json::json!({
        "model": MODEL_ID,
        "max_tokens": 32000,
        "stream": true,
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "low"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "metadata": {"user_id": "local-journey"},
        "system": [{"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "Read", "description": "Reads a file", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Read sonnet.txt."}]},
            {"role": "system", "content": [
                {"type": "text", "text": "The working directory is a scratch folder."},
                {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "Read"}}
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": ""},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "sonnet.txt"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "Shall I compare thee to a summer's day?"}
            ]},
            {"role": "system", "content": [
                {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": {
                    "name": "mcp__folio__search", "description": "Searches the folio",
                    "input_schema": {"type": "object", "properties": {"line": {"type": "string"}}}
                }}},
                {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "mcp__folio__search"}}
            ]}
        ]
    });
    request_document.to_string()
}

#[tokio::test]
async fn should_accept_the_second_turn_claude_code_sends_after_loading_a_deferred_tool() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::TextFragment("Thou art more lovely".to_owned()),
        completed_event(ChatGenerationCompletionReason::EndOfSequence),
    ]));

    let http_response =
        post_messages_body(application, &second_turn_with_a_deferred_tool_definition()).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let parsed_events = parsed_sse_events(http_response).await;
    assert_eq!(
        event_types(&parsed_events).last().copied(),
        Some("message_stop")
    );
}
