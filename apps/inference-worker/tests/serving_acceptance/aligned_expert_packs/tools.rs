//! Tool-calling round trips through the converted expert-streaming model.
//!
//! Two REST-surface contracts are pinned here. First, the model may fail-open
//! a malformed tool-call envelope with an arbitrary invented name; the
//! follow-up request replaying that history must be accepted instead of
//! stranding the conversation with invalid_request. Second, declaring tools
//! on the streaming identity must not degrade ordinary chat streaming.

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, STREAMING_MODEL_ID, assert_streaming_model_is_advertised, households_prompt,
    launch_streaming_model_rest_server, stop_streaming_model_rest_server, streamed_chat_text,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "streams tool-calling round trips through the converted expert-streaming model"]
async fn should_round_trip_tool_history_through_the_expert_streaming_model() {
    timeout(JOURNEY_TIMEOUT, run_tool_journey())
        .await
        .expect("the expert-streaming tool journey must finish within 115 seconds");
}

async fn run_tool_journey() {
    let rest_server = launch_streaming_model_rest_server().await;
    let server_address = rest_server.server_address;
    assert_streaming_model_is_advertised(server_address).await;

    // Ordinary streaming with a declared tool set.
    eprintln!("[aligned-expert-packs] phase=tools-declared model={STREAMING_MODEL_ID}");
    let declared_tools_turn = post_chat_completion(
        server_address,
        json!({
            "model": STREAMING_MODEL_ID,
            "messages": [{
                "role": "user",
                "content": households_prompt(),
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 512,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&declared_tools_turn);
    eprintln!(
        "[aligned-expert-packs] tools_declared_streamed_text={}",
        streamed_chat_text(&declared_tools_turn)
    );

    // The reported failure replayed end to end: history carrying a
    // model-invented tool name that fails the definition grammar must be
    // accepted as a conversation continuation, never rejected as
    // invalid_request.
    eprintln!("[aligned-expert-packs] phase=tools-history-replay model={STREAMING_MODEL_ID}");
    let history_replay_turn = post_chat_completion(
        server_address,
        json!({
            "model": STREAMING_MODEL_ID,
            "messages": [
                {"role": "user", "content": "Inspect the repository and then summarize."},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call-model-invented-1",
                    "type": "function",
                    "function": {"name": "r=bash", "arguments": "{\"command\": \"git status\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call-model-invented-1", "content": "clean working tree"},
                {"role": "user", "content": "Name the two households of Romeo and Juliet in one short sentence."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 512,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&history_replay_turn);
    eprintln!(
        "[aligned-expert-packs] history_replay_streamed_text={}",
        streamed_chat_text(&history_replay_turn)
    );

    stop_streaming_model_rest_server(rest_server).await;
}
