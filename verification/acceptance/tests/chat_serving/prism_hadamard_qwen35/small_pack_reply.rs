//! One text reply and one picture reply for a configured small Bonsai pack.
//!
//! Thinking is off. The draft count is left unset so a dense prediction head
//! stays idle. The configured `model_directory` chooses which pack loads.

use std::time::Duration;

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, image_chat_request_body_for_model,
    post_chat_completion,
};
use crate::support::http;
use crate::support::romeo_and_juliet;

use super::support::{
    launch_configured_pack_rest_server, public_model_id_of_configured_pack, server_address,
    stop_prism_rest_server,
};

// A shared GPU can make one small-pack load take minutes. The deadline is
// long so a slow reply is still a reply.
const JOURNEY_TIMEOUT: Duration = Duration::from_secs(900);

/// Streams one greedy Shakespeare reply from the configured pack.
pub(super) async fn write_greedy_text_reply(pack_size: &str) {
    let deadline = format!("the {pack_size} text reply must finish within 15 minutes");
    timeout(JOURNEY_TIMEOUT, greedy_text_reply(pack_size))
        .await
        .expect(&deadline);
}

/// Answers the existing red-image request from the configured pack.
pub(super) async fn answer_red_picture(pack_size: &str) {
    let deadline = format!("the {pack_size} picture reply must finish within 15 minutes");
    timeout(JOURNEY_TIMEOUT, picture_reply(pack_size))
        .await
        .expect(&deadline);
}

async fn greedy_text_reply(pack_size: &str) {
    let (_home, rest_server) = launch_configured_pack_rest_server().await;
    let model_id = public_model_id_of_configured_pack();
    let chat_response = post_chat_completion(
        server_address(&rest_server),
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": romeo_and_juliet::summary_prompt(80),
            }],
            "stream": true,
            "thinking_budget": 0,
            "temperature": 0.0,
            "max_tokens": 32,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);
    let streamed_text = http::streamed_model_text_from_chat_response(&chat_response);
    eprintln!("[bonsai-2-{pack_size}] text={streamed_text}");
    assert!(
        !streamed_text.trim().is_empty(),
        "the {pack_size} must stream a text reply"
    );
    stop_prism_rest_server(rest_server).await;
}

async fn picture_reply(pack_size: &str) {
    let (_home, rest_server) = launch_configured_pack_rest_server().await;
    let model_id = public_model_id_of_configured_pack();
    let mut request_body: serde_json::Value =
        serde_json::from_str(&image_chat_request_body_for_model(&model_id))
            .expect("the picture request is json");
    request_body["thinking_budget"] = json!(0);
    request_body["temperature"] = json!(0.0);
    request_body["max_tokens"] = json!(32);
    let chat_response =
        post_chat_completion(server_address(&rest_server), request_body.to_string()).await;
    assert_successful_streaming_chat_response(&chat_response);
    let streamed_text = http::streamed_model_text_from_chat_response(&chat_response);
    eprintln!("[bonsai-2-{pack_size}] picture={streamed_text}");
    assert!(
        reply_names_red(&streamed_text),
        "the {pack_size} must name the red picture: {streamed_text:?}"
    );
    stop_prism_rest_server(rest_server).await;
}

fn reply_names_red(streamed_text: &str) -> bool {
    let normalized = streamed_text.to_lowercase();
    for color_name in ["red", "crimson", "scarlet"] {
        if normalized.contains(color_name) {
            return true;
        }
    }
    false
}
