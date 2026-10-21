//! Serving throughput tier for the folded ternary pack: the reply rate the
//! resident REST stack actually delivers (issue #1366).
//!
//! The method mirrors the Ornith-35B throughput journey: one warmup
//! completion absorbs the model load and first-use just-in-time kernel
//! compilation, then cache-warm completions over the identical prompt feed
//! the medians. Rates are server-attributed, never client wall clock — the
//! daemon folds every request's prefill and decode rates into rolling
//! averages, and differencing two `/v1/status` snapshots that bracket
//! exactly one completion recovers the per-request measurement.

use std::time::Duration;

use serde_json::Value;
use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, get_endpoint, post_chat_completion,
};
use crate::support::romeo_and_juliet;

use super::support::{
    launch_prism_rest_server, public_model_id, server_address, stop_prism_rest_server,
};

/// The serving journey launches the full resident stack and completes eight
/// completions over a roughly 1,300-token prompt, so it needs a wider budget
/// than the chat journey's shared timeout.
const SERVING_THROUGHPUT_JOURNEY_TIMEOUT: Duration = Duration::from_secs(240);
const MEASURED_COMPLETION_COUNT: usize = 7;
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 128;
/// A thousand-word excerpt keeps every measured completion prefill-shaped
/// while fitting the journey budget at the folded pack's prefill rate.
const MEASURED_PROMPT_WORD_COUNT: usize = 1_000;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident folded ternary pack and measures serving prompt-processing and decode throughput"]
async fn should_measure_resident_folded_ternary_serving_decode_throughput() {
    timeout(
        SERVING_THROUGHPUT_JOURNEY_TIMEOUT,
        run_serving_throughput_journey(),
    )
    .await
    .expect("the folded-ternary serving throughput journey must finish within 240 seconds");
}

async fn run_serving_throughput_journey() {
    let (isolated_development_home, rest_server) = launch_prism_rest_server().await;
    let address = server_address(&rest_server);
    let model_id = public_model_id();
    let user_prompt = romeo_and_juliet::summary_prompt(MEASURED_PROMPT_WORD_COUNT);

    eprintln!("[prism-hadamard] phase=warmup model={model_id}");
    let warmup_response = post_chat_completion(
        address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": user_prompt.clone(),
            }],
            "stream": true,
            "thinking_budget": 0,
            "temperature": 0.0,
            "max_tokens": 8,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&warmup_response);
    let mut previous_session_sample =
        ServingSessionSample::from_status(&status_document(address).await["serving_session"]);
    assert!(
        previous_session_sample.completed_request_count >= 1,
        "the warmup completion must be attributed before measurement starts"
    );

    let mut prefill_tokens_per_second_samples = Vec::new();
    let mut decode_tokens_per_second_samples = Vec::new();
    for completion_index in 1..=MEASURED_COMPLETION_COUNT {
        eprintln!(
            "[prism-hadamard] phase=measured completion={completion_index}/{MEASURED_COMPLETION_COUNT} model={model_id}"
        );
        let chat_response = post_chat_completion(
            address,
            json!({
                "model": model_id,
                "messages": [{
                    "role": "user",
                    "content": user_prompt.clone(),
                }],
                "stream": true,
                "thinking_budget": 0,
                "temperature": 0.0,
                "max_tokens": MEASURED_MAXIMUM_OUTPUT_TOKENS,
            })
            .to_string(),
        )
        .await;
        assert_successful_streaming_chat_response(&chat_response);
        let current_session_sample =
            ServingSessionSample::from_status(&status_document(address).await["serving_session"]);
        let completion_measurement =
            current_session_sample.recover_request_measurement(&previous_session_sample);
        eprintln!(
            "[prism-hadamard] completion={completion_index} prefill_tok_per_second={:.2} decode_tok_per_second={:.2} prompt_tokens={} reused_prompt_tokens={}",
            completion_measurement.prefill_tok_per_second,
            completion_measurement.decode_tok_per_second,
            completion_measurement.prompt_token_count,
            completion_measurement.reused_prompt_token_count,
        );
        prefill_tokens_per_second_samples.push(completion_measurement.prefill_tok_per_second);
        decode_tokens_per_second_samples.push(completion_measurement.decode_tok_per_second);
        previous_session_sample = current_session_sample;
    }

    let median_prefill_tokens_per_second = median(&mut prefill_tokens_per_second_samples);
    let median_decode_tokens_per_second = median(&mut decode_tokens_per_second_samples);
    eprintln!(
        "[prism-hadamard-perf:tier=serving] median_prefill_tokens_per_second={median_prefill_tokens_per_second:.2} median_decode_tokens_per_second={median_decode_tokens_per_second:.2} measured_completions={MEASURED_COMPLETION_COUNT}"
    );
    assert!(
        median_prefill_tokens_per_second.is_finite() && median_prefill_tokens_per_second > 0.0,
        "serving prompt-processing throughput must be a positive measurement"
    );
    assert!(
        median_decode_tokens_per_second.is_finite() && median_decode_tokens_per_second > 0.0,
        "serving decode throughput must be a positive measurement"
    );

    stop_prism_rest_server(rest_server).await;
    drop(isolated_development_home);
}

async fn status_document(server_address: std::net::SocketAddr) -> Value {
    let status_response = get_endpoint(server_address, "/v1/status").await;
    let response_body = status_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or(status_response.as_str());
    serde_json::from_str(response_body)
        .unwrap_or_else(|_| panic!("response should be JSON: {status_response}"))
}

/// One `/v1/status` serving-session snapshot. The daemon folds each
/// request's prefill and generation rates into rolling averages, so the
/// per-request measurement is recovered by differencing two snapshots that
/// bracket exactly one request.
struct ServingSessionSample {
    completed_request_count: u64,
    average_prefill_tok_per_second: f64,
    average_generation_tok_per_second: f64,
    total_prompt_token_count: u64,
    total_reused_prompt_token_count: u64,
}

/// The per-request measurement recovered between two status snapshots.
struct CompletionMeasurement {
    prefill_tok_per_second: f64,
    decode_tok_per_second: f64,
    prompt_token_count: u64,
    reused_prompt_token_count: u64,
}

impl ServingSessionSample {
    fn from_status(serving_session: &Value) -> Self {
        Self {
            completed_request_count: serving_session["completed_request_count"]
                .as_u64()
                .unwrap_or(0),
            average_prefill_tok_per_second: serving_session["average_prefill_tok_per_second"]
                .as_f64()
                .unwrap_or(0.0),
            average_generation_tok_per_second: serving_session["average_generation_tok_per_second"]
                .as_f64()
                .unwrap_or(0.0),
            total_prompt_token_count: serving_session["total_prompt_token_count"]
                .as_u64()
                .unwrap_or(0),
            total_reused_prompt_token_count: serving_session["total_reused_prompt_token_count"]
                .as_u64()
                .unwrap_or(0),
        }
    }

    /// Undoes the rolling average across the requests completed between the
    /// previous sample and this one. A negative recovery means a request
    /// finished without contributing a rate measurement, which would silently
    /// poison every later recovery, so it fails the journey instead.
    fn recover_request_measurement(
        &self,
        previous: &ServingSessionSample,
    ) -> CompletionMeasurement {
        let request_count = self
            .completed_request_count
            .saturating_sub(previous.completed_request_count);
        assert_eq!(
            request_count, 1,
            "exactly one completion must land between two status samples"
        );
        let measurement_count = self.completed_request_count;
        let previous_measurement_count = previous.completed_request_count;
        let prefill_tok_per_second = measurement_count as f64 * self.average_prefill_tok_per_second
            - previous_measurement_count as f64 * previous.average_prefill_tok_per_second;
        let decode_tok_per_second = measurement_count as f64
            * self.average_generation_tok_per_second
            - previous_measurement_count as f64 * previous.average_generation_tok_per_second;
        assert!(
            prefill_tok_per_second.is_finite() && prefill_tok_per_second > 0.0,
            "the daemon must attribute a positive prefill rate to every completion: {} -> {}",
            previous.average_prefill_tok_per_second,
            self.average_prefill_tok_per_second,
        );
        assert!(
            decode_tok_per_second.is_finite() && decode_tok_per_second > 0.0,
            "the daemon must attribute a positive decode rate to every completion: {} -> {}",
            previous.average_generation_tok_per_second,
            self.average_generation_tok_per_second,
        );
        CompletionMeasurement {
            prefill_tok_per_second,
            decode_tok_per_second,
            prompt_token_count: self
                .total_prompt_token_count
                .saturating_sub(previous.total_prompt_token_count),
            reused_prompt_token_count: self
                .total_reused_prompt_token_count
                .saturating_sub(previous.total_reused_prompt_token_count),
        }
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|first, second| first.total_cmp(second));
    let middle = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}
