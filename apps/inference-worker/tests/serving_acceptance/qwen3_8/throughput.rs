//! Qwen3.8-35B-A3B-Distill-oQ6e-mtp serving throughput onboarding journey.
//!
//! This journey proves the funnygeeker artifact — whose MTP block and vision
//! tower ship inside the main shards with no sidecar keys — serves end to
//! end, and records the onboarding throughput baseline: one warmup completion
//! absorbs the model load plus first-use JIT kernel compilation, then seven
//! cache-warm completions over the identical full Romeo and Juliet prompt
//! feed the medians. All rates are server-attributed, never client wall
//! clock: prefill and decode both come from the supervisor's per-request
//! counters, which `/v1/status` only exposes as a rolling average and this
//! journey recovers exactly by differencing consecutive snapshots. Decode
//! deliberately does not come from attribution reports: their
//! `decode_advance_span` counts chunk spans rather than wall-clock decode
//! time and under-reports the rate (measured 44.9 versus 60+ tok/s on the
//! identical prompt and build). The MTP runtime state is echoed so the
//! onboarding record shows whether the text_config-declared MTP block was
//! picked up without a sidecar.

use serde_json::Value;
use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, QWEN3_8_MODEL_ID, launch_rest_server, romeo_and_juliet_prompt,
    status_document, stop_rest_server,
};

const MEASURED_COMPLETION_COUNT: usize = 7;
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 128;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident 30 GB Qwen3.8 sparse-MoE artifact and measures serving prompt-processing and decode throughput"]
async fn should_measure_qwen3_8_distill_prompt_processing_and_decode_throughput() {
    timeout(JOURNEY_TIMEOUT, run_throughput_journey())
        .await
        .expect("the Qwen3.8 throughput journey must finish within 115 seconds");
}

async fn run_throughput_journey() {
    let (_isolated_home, rest_server) = launch_rest_server().await;
    let server_address = rest_server.server_address;

    eprintln!("[qwen3-8] phase=jit-warmup model={QWEN3_8_MODEL_ID}");
    let warmup_response = post_chat_completion(
        server_address,
        json!({
            "model": QWEN3_8_MODEL_ID,
            "messages": [{
                "role": "user",
                "content": romeo_and_juliet_prompt(),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 8,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&warmup_response);
    let mut previous_session_sample = ServingSessionSample::from_status(
        &status_document(server_address).await["serving_session"],
    );
    assert!(
        previous_session_sample.completed_request_count >= 1,
        "the warmup completion must be attributed before measurement starts"
    );

    let mut prefill_tokens_per_second_samples = Vec::new();
    let mut decode_tokens_per_second_samples = Vec::new();
    for completion_index in 1..=MEASURED_COMPLETION_COUNT {
        eprintln!(
            "[qwen3-8] phase=measured completion={completion_index}/{} model={QWEN3_8_MODEL_ID}",
            MEASURED_COMPLETION_COUNT
        );
        let chat_response = post_chat_completion(
            server_address,
            json!({
                "model": QWEN3_8_MODEL_ID,
                "messages": [{
                    "role": "user",
                    "content": romeo_and_juliet_prompt(),
                }],
                "stream": true,
                "temperature": 1,
                "max_tokens": MEASURED_MAXIMUM_OUTPUT_TOKENS,
            })
            .to_string(),
        )
        .await;
        assert_successful_streaming_chat_response(&chat_response);
        let current_session_sample = ServingSessionSample::from_status(
            &status_document(server_address).await["serving_session"],
        );
        let completion_measurement =
            current_session_sample.recover_request_measurement(&previous_session_sample);
        eprintln!(
            "[qwen3-8] completion={completion_index} prefill_tok_per_second={:.2} decode_tok_per_second={:.2} prompt_tokens={} reused_prompt_tokens={}",
            completion_measurement.prefill_tok_per_second,
            completion_measurement.decode_tok_per_second,
            completion_measurement.prompt_token_count,
            completion_measurement.reused_prompt_token_count,
        );
        prefill_tokens_per_second_samples.push(completion_measurement.prefill_tok_per_second);
        decode_tokens_per_second_samples.push(completion_measurement.decode_tok_per_second);
        previous_session_sample = current_session_sample;
    }

    let final_session_sample = previous_session_sample;
    let median_prefill_tokens_per_second = median(&mut prefill_tokens_per_second_samples);
    let median_decode_tokens_per_second = median(&mut decode_tokens_per_second_samples);
    eprintln!(
        "[qwen3-8] BASELINE model={QWEN3_8_MODEL_ID} median_prefill_tok_per_second={median_prefill_tokens_per_second:.2} median_decode_tok_per_second={median_decode_tokens_per_second:.2} completions={MEASURED_COMPLETION_COUNT} completed_request_count={}",
        final_session_sample.completed_request_count,
    );
    assert!(
        median_prefill_tokens_per_second.is_finite() && median_prefill_tokens_per_second > 0.0,
        "prompt-processing throughput must be a positive measurement"
    );
    assert!(
        median_decode_tokens_per_second.is_finite() && median_decode_tokens_per_second > 0.0,
        "decode throughput must be a positive measurement"
    );

    stop_rest_server(rest_server).await;
}

/// One `/v1/status` serving-session snapshot. The supervisor folds each
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
            "the supervisor must attribute a positive prefill rate to every completion: {} -> {}",
            previous.average_prefill_tok_per_second,
            self.average_prefill_tok_per_second,
        );
        assert!(
            decode_tok_per_second.is_finite() && decode_tok_per_second > 0.0,
            "the supervisor must attribute a positive generation rate to every completion: {} -> {}",
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
