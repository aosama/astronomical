//! A coding-shaped tool catalog must produce a get_scene tool call.

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::post_chat_completion;

use super::support::{
    JOURNEY_TIMEOUT, assert_k2_is_advertised, compact_romeo_and_juliet_source,
    launch_k2_rest_server, public_model_id, stop_k2_rest_server, streamed_chat_text,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "honors get_scene through Chat Completions tools for K2 Horizon MoVA"]
async fn should_emit_get_scene_tool_call_for_romeo_and_juliet() {
    timeout(JOURNEY_TIMEOUT, run_tool_journey())
        .await
        .expect("the K2 Horizon MoVA tool journey must finish within 115 seconds");
}

async fn run_tool_journey() {
    let (_isolated_development_home, rest_server) = launch_k2_rest_server().await;
    let server_address = rest_server.server_address;
    assert_k2_is_advertised(server_address).await;
    let model_id = public_model_id();
    eprintln!("[k2-horizon-mova] phase=tools model={model_id}");
    let tool_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": format!(
                    "Call get_scene with title Romeo and Juliet. Do not answer in prose.\n\n{}",
                    compact_romeo_and_juliet_source()
                ),
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_scene",
                    "description": "Return a scene from the play",
                    "parameters": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"]
                    }
                }
            }],
            "tool_choice": "auto",
            "stream": true,
            "temperature": 1,
            "max_tokens": 512,
        })
        .to_string(),
    )
    .await;
    assert!(
        tool_response.starts_with("HTTP/1.1 200 OK"),
        "unexpected HTTP response: {tool_response}"
    );
    assert!(
        tool_response.contains("data: [DONE]"),
        "tool stream must finish cleanly: {tool_response}"
    );
    let streamed_text = streamed_chat_text(&tool_response);
    eprintln!("[k2-horizon-mova] tool_streamed_text={streamed_text}");
    assert!(
        has_structured_get_scene_tool_call(&tool_response),
        "Chat Completions must emit a structured get_scene tool call, got {streamed_text:?}"
    );
    stop_k2_rest_server(rest_server).await;
}

fn has_structured_get_scene_tool_call(tool_response: &str) -> bool {
    tool_response.lines().any(|line| {
        let Some(payload) = line.strip_prefix("data: ") else {
            return false;
        };
        let Ok(chunk) = serde_json::from_str::<serde_json::Value>(payload) else {
            return false;
        };
        chunk
            .pointer("/choices/0/delta/tool_calls")
            .or_else(|| chunk.pointer("/choices/0/message/tool_calls"))
            .and_then(serde_json::Value::as_array)
            .is_some_and(|tool_calls| {
                tool_calls.iter().any(|tool_call| {
                    tool_call
                        .pointer("/function/name")
                        .and_then(serde_json::Value::as_str)
                        == Some("get_scene")
                })
            })
    })
}
