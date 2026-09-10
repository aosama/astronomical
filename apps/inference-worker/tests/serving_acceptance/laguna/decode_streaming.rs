//! SSD decode streaming journey for Laguna: experts reuse across generated tokens.

use std::time::Duration;

use serde_json::json;
use tokio::time::timeout;

use super::http::assert_laguna_is_advertised;
use super::validate::{
    compact_romeo_and_juliet_source, laguna_xs_public_model_id, resolve_reference_model_directory,
};
use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, get_endpoint, launch_serving_rest_server_for_model,
    post_chat_completion, stop_serving_rest_server,
};

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const DECODE_OUTPUT_TOKEN_COUNT: u16 = 16;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "streams Romeo through Laguna decode under a paging memory ceiling"]
async fn should_stream_decode_experts_from_ssd_and_report_generation_throughput() {
    timeout(JOURNEY_TIMEOUT, run_decode_streaming_journey())
        .await
        .expect("the Laguna decode streaming journey must finish within 115 seconds");
}

async fn run_decode_streaming_journey() {
    let public_model_id = laguna_xs_public_model_id();
    let model_directory = resolve_reference_model_directory();
    let isolated_development_home = crate::support::isolated_development_home_from_user_config();
    eprintln!("[laguna-decode-streaming] phase=launch model={public_model_id}");
    let rest_server = launch_serving_rest_server_for_model(
        public_model_id,
        model_directory,
        Some(isolated_development_home.path()),
        None,
    )
    .await;
    let server_address = rest_server.server_address;
    assert_laguna_is_advertised(server_address, public_model_id).await;
    let source_excerpt = compact_romeo_and_juliet_source();

    eprintln!("[laguna-decode-streaming] phase=generate model={public_model_id}");
    let chat_response = post_chat_completion(
        server_address,
        json!({
            "model": public_model_id,
            "messages": [{
                "role": "user",
                "content": format!("Use the supplied Romeo and Juliet source. Name the two households.\n\n{source_excerpt}"),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": DECODE_OUTPUT_TOKEN_COUNT,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);

    let status_response = get_endpoint(server_address, "/v1/status").await;
    let status_body = status_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or(status_response.as_str());
    let status_document: serde_json::Value = serde_json::from_str(status_body)
        .unwrap_or_else(|_| panic!("GET /v1/status should return JSON: {status_response}"));
    let serving_session = &status_document["serving_session"];
    let average_prefill_tokens_per_second = serving_session["average_prefill_tok_per_second"]
        .as_f64()
        .expect("status should report prefill throughput");
    let average_generation_tokens_per_second = serving_session["average_generation_tok_per_second"]
        .as_f64()
        .expect("status should report generation throughput");
    assert!(
        average_prefill_tokens_per_second.is_finite() && average_prefill_tokens_per_second > 0.0,
        "Laguna SSD prefill must report positive throughput: {status_document}"
    );
    assert!(
        average_generation_tokens_per_second.is_finite()
            && average_generation_tokens_per_second > 0.0,
        "Laguna SSD decode must report positive throughput: {status_document}"
    );
    let resident_expert_count = status_document["expert_residency"]["resident_expert_count"]
        .as_u64()
        .unwrap_or(0);
    let resident_expert_payload_bytes =
        status_document["expert_residency"]["resident_expert_payload_bytes"]
            .as_u64()
            .unwrap_or(0);
    assert!(
        resident_expert_count > 0 || resident_expert_payload_bytes > 0,
        "decode must leave resident experts after SSD paging: {status_document}"
    );
    eprintln!(
        "[laguna-decode-streaming] status=success prefill_tok_per_second={average_prefill_tokens_per_second:.2} generation_tok_per_second={average_generation_tokens_per_second:.2} resident_expert_count={resident_expert_count} resident_expert_payload_bytes={resident_expert_payload_bytes}"
    );
    stop_serving_rest_server(rest_server).await;
}
