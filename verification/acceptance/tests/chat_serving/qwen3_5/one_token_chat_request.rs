//! A one-token streamed chat request that must return visible model output,
//! shared by the journeys that only need to prove the served model answers.

use std::net::SocketAddr;

use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::get_json_endpoint;

/// Dumps the worker status on a failed start so a refused request explains
/// itself (for example a memory ceiling the engine could not honor).
pub(super) async fn complete_one_token_chat_request(
    server_address: SocketAddr,
    openai_client: &LocalOpenAiClient,
    model_id: &str,
    messages: Value,
    request_label: &str,
) -> String {
    let completion_request = json!({
        "model": model_id,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 1,
        "thinking_budget": 0,
        "max_tokens": 1
    });
    let mut streamed_completion = match openai_client
        .create_streaming_chat_completion(&completion_request)
        .await
    {
        Ok(streamed_completion) => streamed_completion,
        Err(request_error) => {
            let status_document = get_json_endpoint(server_address, "/v1/status").await;
            panic!(
                "{request_label} should start: {request_error}; worker_status={status_document}"
            );
        }
    };
    let mut assistant_response = String::new();
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.unwrap_or_else(|stream_error| {
            panic!("{request_label} should stream a valid response: {stream_error}")
        });
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(text_fragment) = choice["delta"]["content"].as_str() {
                assistant_response.push_str(text_fragment);
            }
        }
    }
    assert!(
        !assistant_response.trim().is_empty(),
        "{request_label} should return visible model output"
    );
    assistant_response
}
