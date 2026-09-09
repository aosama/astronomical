//! Chat Completions must answer a Romeo and Juliet household question in English.

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, assert_k2_is_advertised, households_prompt, launch_k2_rest_server,
    names_the_households, public_model_id, stop_k2_rest_server, streamed_chat_text,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "streams Romeo and Juliet households through Chat Completions for K2 Horizon MoVA"]
async fn should_name_romeo_and_juliet_households_over_chat_completions() {
    timeout(JOURNEY_TIMEOUT, run_chat_journey())
        .await
        .expect("the K2 Horizon MoVA Chat Completions journey must finish within 115 seconds");
}

async fn run_chat_journey() {
    let (_isolated_development_home, rest_server) = launch_k2_rest_server().await;
    let server_address = rest_server.server_address;
    assert_k2_is_advertised(server_address).await;
    let model_id = public_model_id();
    eprintln!("[k2-horizon-mova] phase=chat model={model_id}");
    let chat_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": households_prompt(),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 512,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);
    let streamed_text = streamed_chat_text(&chat_response);
    eprintln!("[k2-horizon-mova] chat_streamed_text={streamed_text}");
    assert!(
        names_the_households(&streamed_text),
        "Chat Completions must name Montague or Capulet, got {streamed_text:?}"
    );
    stop_k2_rest_server(rest_server).await;
}
