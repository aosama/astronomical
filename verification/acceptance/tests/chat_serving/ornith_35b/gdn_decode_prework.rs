//! Production-selected gated-delta decode diagnostic measurement.
//!
//! Production kernel capability probes select the execution path. This
//! journey measures that selection without changing it or claiming to compare
//! switch arms. Direct fused-versus-composed numeric tests remain separate.
//!
//! A 500-token Romeo and Juliet warmup requests 50 output tokens, then a
//! 10,000-token prompt requests 500 output tokens with persistent caching off.
//! Decode rates come from daemon status differencing, never attribution spans.
//! Run the optimized diagnostic with `cargo xtask test-run --package
//! astronomical-acceptance --test chat_serving --features serving-acceptance
//! --release --configuration PATH --filter
//! serving_acceptance::ornith_35b::gdn_decode_prework::should_measure_production_selected_gated_delta_decode_throughput
//! --exact --ignored`.

use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};
use crate::support::exact_model_prompt::build_exact_model_prompt_content;

use super::support::{
    JOURNEY_TIMEOUT, launch_resident_rest_server_for_decode_diagnostics, model_directory,
    resident_model_id, status_document, stop_resident_rest_server,
};

const WARMUP_INPUT_TOKEN_COUNT: usize = 500;
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 50;
const MEASURED_INPUT_TOKEN_COUNT: usize = 10_000;
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 500;
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_warmup_romeo_and_juliet.txt");
const MEASURED_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_10000_tokens_romeo_and_juliet.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident sparse-MoE model and measures the production capability-selected decode path"]
#[serial]
async fn should_measure_production_selected_gated_delta_decode_throughput() {
    if cfg!(debug_assertions) {
        panic!(
            "run the production-selected decode diagnostic through cargo xtask test-run --release"
        );
    }
    timeout(JOURNEY_TIMEOUT, measure_production_selected_decode())
        .await
        .expect("the production-selected decode diagnostic must finish within 115 seconds");
}

async fn measure_production_selected_decode() {
    let model_id = resident_model_id();
    let model_directory = model_directory();
    let warmup_prompt = build_exact_model_prompt_content(
        &model_directory,
        WARMUP_ROMEO_AND_JULIET_SOURCE,
        "Continue the supplied Romeo and Juliet story.",
        WARMUP_INPUT_TOKEN_COUNT,
    );
    let measured_prompt = build_exact_model_prompt_content(
        &model_directory,
        MEASURED_ROMEO_AND_JULIET_SOURCE,
        "Continue the supplied Romeo and Juliet story for approximately 500 tokens.",
        MEASURED_INPUT_TOKEN_COUNT,
    );
    let (_isolated_home, rest_server) = launch_resident_rest_server_for_decode_diagnostics().await;
    let server_address = rest_server.server_address;

    eprintln!("[ornith-35b-production-decode-diagnostic] phase=warmup model={model_id}");
    let warmup_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": warmup_prompt,
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": WARMUP_MAXIMUM_OUTPUT_TOKENS,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&warmup_response);
    let previous_session_sample = status_sample(server_address).await;
    assert!(
        previous_session_sample.completed_request_count >= 1,
        "the warmup completion must be attributed before measurement starts"
    );

    eprintln!("[ornith-35b-production-decode-diagnostic] phase=measured model={model_id}");
    let chat_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{"role": "user", "content": measured_prompt}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "temperature": 1,
            "max_tokens": MEASURED_MAXIMUM_OUTPUT_TOKENS,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);
    assert_measured_completion_shape(&chat_response);
    let current_session_sample = status_sample(server_address).await;
    let decode_tokens_per_second =
        current_session_sample.recover_generation_rate(&previous_session_sample);
    eprintln!(
        "[ornith-35b-production-decode-diagnostic] selection=production-capability-probes decode_tokens_per_second={decode_tokens_per_second:.2}"
    );

    stop_resident_rest_server(rest_server).await;
}

/// One `/v1/status` serving-session snapshot. The daemon folds each
/// request's rates into rolling averages, so the per-request measurement is
/// recovered by differencing two snapshots bracketing exactly one request.
struct ServingSessionSample {
    completed_request_count: u64,
    average_generation_tok_per_second: f64,
}

impl ServingSessionSample {
    async fn new(server_address: std::net::SocketAddr) -> Self {
        Self::from_status(&status_document(server_address).await["serving_session"])
    }

    fn from_status(serving_session: &serde_json::Value) -> Self {
        Self {
            completed_request_count: serving_session["completed_request_count"]
                .as_u64()
                .unwrap_or(0),
            average_generation_tok_per_second: serving_session["average_generation_tok_per_second"]
                .as_f64()
                .unwrap_or(0.0),
        }
    }

    fn recover_generation_rate(&self, previous: &ServingSessionSample) -> f64 {
        let request_count = self
            .completed_request_count
            .saturating_sub(previous.completed_request_count);
        assert_eq!(
            request_count, 1,
            "exactly one completion must land between two status samples"
        );
        let decode_tok_per_second = self.completed_request_count as f64
            * self.average_generation_tok_per_second
            - previous.completed_request_count as f64 * previous.average_generation_tok_per_second;
        assert!(
            decode_tok_per_second.is_finite() && decode_tok_per_second > 0.0,
            "the daemon must attribute a positive decode rate to every completion: {} -> {}",
            previous.average_generation_tok_per_second,
            self.average_generation_tok_per_second,
        );
        decode_tok_per_second
    }
}

async fn status_sample(server_address: std::net::SocketAddr) -> ServingSessionSample {
    ServingSessionSample::new(server_address).await
}

fn assert_measured_completion_shape(chat_response: &str) {
    let usage = chat_response
        .lines()
        .filter_map(|streamed_line| streamed_line.strip_prefix("data: "))
        .filter_map(|event_json| serde_json::from_str::<Value>(event_json).ok())
        .find_map(|streamed_event| {
            streamed_event
                .get("usage")
                .filter(|usage| usage.is_object())
                .cloned()
        })
        .expect("the measured diagnostic completion should report usage");
    let prompt_tokens = usage["prompt_tokens"].as_u64().expect("prompt_tokens");
    let completion_tokens = usage["completion_tokens"]
        .as_u64()
        .expect("completion_tokens");
    assert!(
        (9_000..=11_000).contains(&prompt_tokens),
        "the diagnostic should measure approximately 10,000 Romeo and Juliet prompt tokens"
    );
    assert!(
        (450..=550).contains(&completion_tokens),
        "the diagnostic should measure approximately 500 output tokens"
    );
}
