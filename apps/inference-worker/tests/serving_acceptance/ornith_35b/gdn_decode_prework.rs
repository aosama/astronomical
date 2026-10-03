//! Fused GDN decode prework A/B measurement, one phase per invocation.
//!
//! The fused prework kernel replaces roughly thirteen launch-bound dispatches
//! per gated-delta layer at decode shapes with one Metal launch. The hermetic
//! numerics test proves the kernel bit-matches the composed path; this journey
//! is the measurement half. The worker test binary forbids unsafe code, so the
//! disengage switch must come from the environment the journey is invoked
//! under (`ASTRONOMICAL_GDN_DECODE_PREWORK=0`), and this journey reads — never
//! sets — that value and labels every printed line with the resulting mode.
//!
//! Run the journey alternately with the variable unset (engaged) and set to
//! `0` (disengaged) so thermal state cannot order-confound the comparison;
//! aggregate the medians across the alternation. Decode rates come from
//! supervisor status differencing, never attribution spans. A warm-up
//! completion absorbs the model load and first-use JIT compilation.
//! Persistent prompt caching makes each measured completion after the warmup
//! a pure decode measurement over the identical Romeo and Juliet prompt.

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, launch_resident_rest_server, resident_model_id, romeo_and_juliet_prompt,
    status_document, stop_resident_rest_server,
};

const MEASURED_COMPLETION_COUNT: usize = 3;
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 128;
const PREWORK_DISENGAGE_ENVIRONMENT: &str = "ASTRONOMICAL_GDN_DECODE_PREWORK";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident 35B sparse-MoE model and measures decode for the ambient fused-prework mode"]
async fn should_measure_decode_for_the_ambient_prework_mode() {
    timeout(JOURNEY_TIMEOUT, measure_ambient_prework_decode())
        .await
        .expect("the prework decode measurement must finish within 115 seconds");
}

async fn measure_ambient_prework_decode() {
    let prework_disengaged = std::env::var(PREWORK_DISENGAGE_ENVIRONMENT)
        .map(|value| value == "0")
        .unwrap_or(false);
    let mode = if prework_disengaged {
        "disengaged"
    } else {
        "engaged"
    };
    let model_id = resident_model_id();
    let (_isolated_home, rest_server) = launch_resident_rest_server().await;
    let server_address = rest_server.server_address;

    eprintln!("[ornith-35b-prework] mode={mode} phase=jit-warmup model={model_id}");
    let warmup_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
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
    let mut previous_session_sample = status_sample(server_address).await;
    assert!(
        previous_session_sample.completed_request_count >= 1,
        "the warmup completion must be attributed before measurement starts"
    );

    let mut decode_tokens_per_second_samples = Vec::new();
    for completion_index in 1..=MEASURED_COMPLETION_COUNT {
        eprintln!(
            "[ornith-35b-prework] mode={mode} phase=measured completion={completion_index}/{MEASURED_COMPLETION_COUNT} model={model_id}"
        );
        let chat_response = post_chat_completion(
            server_address,
            json!({
                "model": model_id,
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
        let current_session_sample = status_sample(server_address).await;
        let decode_tok_per_second =
            current_session_sample.recover_generation_rate(&previous_session_sample);
        eprintln!(
            "[ornith-35b-prework] mode={mode} completion={completion_index} decode_tok_per_second={decode_tok_per_second:.2}"
        );
        decode_tokens_per_second_samples.push(decode_tok_per_second);
        previous_session_sample = current_session_sample;
    }

    let median_decode_tok_per_second = median(&mut decode_tokens_per_second_samples);
    eprintln!(
        "[ornith-35b-prework] MODE_MEDIAN mode={mode} median_decode_tok_per_second={median_decode_tok_per_second:.2}"
    );
    assert!(
        median_decode_tok_per_second.is_finite() && median_decode_tok_per_second > 0.0,
        "the {mode} prework phase must report a positive decode rate"
    );

    stop_resident_rest_server(rest_server).await;
}

/// One `/v1/status` serving-session snapshot. The supervisor folds each
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
            "the supervisor must attribute a positive decode rate to every completion: {} -> {}",
            previous.average_generation_tok_per_second,
            self.average_generation_tok_per_second,
        );
        decode_tok_per_second
    }
}

async fn status_sample(server_address: std::net::SocketAddr) -> ServingSessionSample {
    ServingSessionSample::new(server_address).await
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
