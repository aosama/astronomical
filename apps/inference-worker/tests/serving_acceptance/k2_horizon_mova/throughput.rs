//! Prompt-processing and decode rates must be measured, not guessed.

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, assert_k2_is_advertised, launch_k2_rest_server, public_model_id,
    status_document, stop_k2_rest_server, throughput_prompt,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "measures K2 Horizon MoVA prompt-processing and decode tokens per second"]
async fn should_measure_k2_horizon_mova_prompt_processing_and_decode_throughput() {
    timeout(JOURNEY_TIMEOUT, run_throughput_journey())
        .await
        .expect("the K2 Horizon MoVA throughput journey must finish within 115 seconds");
}

async fn run_throughput_journey() {
    let (_isolated_development_home, rest_server) = launch_k2_rest_server().await;
    let server_address = rest_server.server_address;
    assert_k2_is_advertised(server_address).await;
    let model_id = public_model_id();
    eprintln!("[k2-horizon-mova] phase=throughput-warmup model={model_id}");
    let warmup_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": throughput_prompt(),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 8,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&warmup_response);
    let warmup_session = status_document(server_address).await["serving_session"].clone();
    eprintln!(
        "[k2-horizon-mova] warmup_prefill_tok_per_second={:.2} warmup_generation_tok_per_second={:.2}",
        warmup_session["average_prefill_tok_per_second"]
            .as_f64()
            .unwrap_or(0.0),
        warmup_session["average_generation_tok_per_second"]
            .as_f64()
            .unwrap_or(0.0),
    );
    eprintln!("[k2-horizon-mova] phase=throughput-warm model={model_id}");
    let chat_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": format!("Second pass.\n\n{}", throughput_prompt()),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 48,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);
    let serving_session = status_document(server_address).await["serving_session"].clone();
    let completed_request_count = serving_session["completed_request_count"]
        .as_u64()
        .unwrap_or(0);
    let average_prefill_tokens_per_second = serving_session["average_prefill_tok_per_second"]
        .as_f64()
        .unwrap_or(0.0);
    let average_generation_tokens_per_second = serving_session["average_generation_tok_per_second"]
        .as_f64()
        .unwrap_or(0.0);
    eprintln!(
        "[k2-horizon-mova] completed_request_count={completed_request_count} prompt_tokens={} average_prefill_tok_per_second={average_prefill_tokens_per_second:.2} average_generation_tok_per_second={average_generation_tokens_per_second:.2}",
        serving_session["total_prompt_token_count"],
    );
    assert!(
        completed_request_count >= 1,
        "a completed K2 Horizon MoVA chat must increment the serving session: {serving_session}"
    );
    assert!(
        average_prefill_tokens_per_second.is_finite() && average_prefill_tokens_per_second > 0.0,
        "prompt-processing throughput must be a positive measurement: {serving_session}"
    );
    assert!(
        average_generation_tokens_per_second >= 40.0,
        "a 4B-active MoE must decode at 40 tok/s or faster; anything below is a defect, got {average_generation_tokens_per_second:.2}: {serving_session}"
    );
    stop_k2_rest_server(rest_server).await;
}
