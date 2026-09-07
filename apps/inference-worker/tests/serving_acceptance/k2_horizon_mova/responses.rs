//! Responses must answer the same Romeo and Juliet household question.

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_responses_response, post_responses_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, assert_k2_is_advertised, households_prompt, launch_k2_rest_server,
    names_the_households, public_model_id, stop_k2_rest_server, streamed_responses_text,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "streams Romeo and Juliet households through Responses for K2 Horizon MoVA"]
async fn should_name_romeo_and_juliet_households_over_responses() {
    timeout(JOURNEY_TIMEOUT, run_responses_journey())
        .await
        .expect("the K2 Horizon MoVA Responses journey must finish within 115 seconds");
}

async fn run_responses_journey() {
    let (_isolated_development_home, rest_server) = launch_k2_rest_server().await;
    let server_address = rest_server.server_address;
    assert_k2_is_advertised(server_address).await;
    let model_id = public_model_id();
    eprintln!("[k2-horizon-mova] phase=responses model={model_id}");
    let responses_response = post_responses_completion(
        server_address,
        json!({
            "model": model_id,
            "input": households_prompt(),
            "stream": true,
            "max_output_tokens": 512,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_responses_response(&responses_response);
    let streamed_text = streamed_responses_text(&responses_response);
    eprintln!("[k2-horizon-mova] responses_streamed_text={streamed_text}");
    assert!(
        names_the_households(&streamed_text),
        "Responses must name Montague or Capulet, got {streamed_text:?}"
    );
    stop_k2_rest_server(rest_server).await;
}
