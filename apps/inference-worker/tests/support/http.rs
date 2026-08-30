use std::net::SocketAddr;

use serde_json::Value;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

#[allow(dead_code)]
pub(crate) fn streamed_model_text_from_chat_response(chat_response: &str) -> String {
    let mut streamed_model_text = String::new();
    for response_line in chat_response.lines() {
        let Some(server_sent_event_payload) = response_line.strip_prefix("data: ") else {
            continue;
        };
        if server_sent_event_payload == "[DONE]" {
            continue;
        }
        let Ok(stream_chunk_json) = serde_json::from_str::<Value>(server_sent_event_payload) else {
            continue;
        };
        let Some(delta_json) = stream_chunk_json.pointer("/choices/0/delta") else {
            continue;
        };
        if let Some(content_delta) = delta_json.get("content").and_then(Value::as_str) {
            streamed_model_text.push_str(content_delta);
        }
        if let Some(reasoning_content_delta) =
            delta_json.get("reasoning_content").and_then(Value::as_str)
        {
            streamed_model_text.push_str(reasoning_content_delta);
        }
    }
    streamed_model_text
}

pub(crate) async fn send_http_request(server_address: SocketAddr, request_text: String) -> String {
    let mut server_connection = TcpStream::connect(server_address)
        .await
        .expect("the E2E server should accept a local connection");
    server_connection
        .write_all(request_text.as_bytes())
        .await
        .expect("the E2E HTTP request should be written");
    let mut response_text = String::new();
    server_connection
        .read_to_string(&mut response_text)
        .await
        .expect("the bounded E2E HTTP response should be readable");
    response_text
}
