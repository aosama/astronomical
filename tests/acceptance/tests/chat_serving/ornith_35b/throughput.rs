//! Ornith-1.5-35B serving throughput baseline for the pinned MLX runtime.
//!
//! This journey is the measurement half of the paired A/B that gates a
//! pinned-MLX bump: run it unchanged on the old pin and the new pin, then
//! compare the printed medians. The method follows the documented previous
//! bump recipe: one warmup
//! completion absorbs the model load plus the first-use JIT kernel
//! compilation (its prefill sits an order of magnitude low and must never
//! enter the comparison), then seven cache-warm completions over the
//! identical full Romeo and Juliet prompt feed the medians. All rates are
//! server-attributed, never client wall clock: prefill and decode both come
//! from the daemon's per-request counters, which `/v1/status` only
//! exposes as a rolling average and this journey recovers exactly by
//! differencing consecutive snapshots. Decode deliberately does not come
//! from attribution reports: their `decode_advance_span` counts chunk spans
//! rather than wall-clock decode time and under-reports the rate.

use serde_json::Value;
use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, launch_resident_rest_server, resident_model_id, romeo_and_juliet_prompt,
    status_document, stop_resident_rest_server,
};

const MEASURED_COMPLETION_COUNT: usize = 7;
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 128;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident 35B sparse-MoE model and measures serving prompt-processing and decode throughput"]
async fn should_measure_resident_sparse_moe_prompt_processing_and_decode_throughput() {
    timeout(JOURNEY_TIMEOUT, run_throughput_journey())
        .await
        .expect("the Ornith-35B throughput journey must finish within 115 seconds");
}

async fn run_throughput_journey() {
    let model_id = resident_model_id();
    let (isolated_home, rest_server) = launch_resident_rest_server().await;
    let server_address = rest_server.server_address;

    eprintln!("[ornith-35b] phase=jit-warmup model={model_id}");
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
            "[ornith-35b] phase=measured completion={completion_index}/{} model={model_id}",
            MEASURED_COMPLETION_COUNT
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
        let current_session_sample = ServingSessionSample::from_status(
            &status_document(server_address).await["serving_session"],
        );
        let completion_measurement =
            current_session_sample.recover_request_measurement(&previous_session_sample);
        eprintln!(
            "[ornith-35b] completion={completion_index} prefill_tok_per_second={:.2} decode_tok_per_second={:.2} prompt_tokens={} reused_prompt_tokens={}",
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
        "[ornith-35b] BASELINE model={model_id} median_prefill_tok_per_second={median_prefill_tokens_per_second:.2} median_decode_tok_per_second={median_decode_tokens_per_second:.2} completions={MEASURED_COMPLETION_COUNT} completed_request_count={}",
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

    stop_resident_rest_server(rest_server).await;
    report_prefix_adoption_evidence(isolated_home.path());
}

/// Reads the journey's worker attribution log and prints the per-completion
/// restore evidence the prefix-adoption A/B compares: the restore-side span
/// occurrences and total (prefix lookup, KV-block reads, state
/// reconstruction, recurrent-snapshot read). Arm B (adoption off) shows the
/// restore spans on every cache-hit completion; arm A (adoption on) shows
/// zero restore-span occurrences — the live state was adopted instead.
fn report_prefix_adoption_evidence(isolated_home: &std::path::Path) {
    let adoption_arm_enabled = std::env::var("ORNITH_JOURNEY_PREFIX_ADOPTION")
        .map(|value| value == "1")
        .unwrap_or(false);
    let attribution_log_path = isolated_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let Some(generation_reports) = read_jsonl_lines(&attribution_log_path).map(|reports| {
        reports
            .into_iter()
            .filter(|report| {
                report.get("report_kind").and_then(Value::as_str) == Some("generation")
            })
            .collect::<Vec<_>>()
    }) else {
        eprintln!("[ornith-35b-adoption] no worker attribution log found");
        return;
    };

    eprintln!(
        "[ornith-35b-adoption] completion kv_block_read_occurrences restore_span_occurrences restore_spans_ms"
    );
    for (completion_index, generation_report) in generation_reports.iter().enumerate() {
        let (kv_block_read_occurrences, restore_span_occurrences, restore_spans_ms) =
            restore_span_evidence(generation_report);
        eprintln!(
            "[ornith-35b-adoption] completion={} kv_block_read_occurrences={kv_block_read_occurrences} restore_span_occurrences={restore_span_occurrences} restore_spans_ms={restore_spans_ms:.1}",
            completion_index + 1,
        );
        if adoption_arm_enabled && completion_index > 0 {
            assert_eq!(
                kv_block_read_occurrences, 0,
                "completion {completion_index} must adopt the retained live state and read no \
                 KV blocks from the disk store"
            );
        }
    }
}

/// Counts and sums the restore-side operation spans (prefix lookup, KV-block
/// reads, state reconstruction, recurrent-snapshot read) in one generation
/// report; zero occurrences mean the live state was adopted. The KV-block
/// read count is returned separately: it is the byte-moving part of the
/// restore that adoption must eliminate.
fn restore_span_evidence(generation_report: &Value) -> (u64, u64, f64) {
    const RESTORE_OPERATIONS: [&str; 4] = [
        "persistent_prompt_cache_prefix_lookup",
        "persistent_prompt_cache_kv_block_read",
        "persistent_prompt_cache_state_reconstruction",
        "persistent_prompt_cache_recurrent_snapshot_read",
    ];
    generation_report
        .get("operations")
        .and_then(Value::as_array)
        .map(|operations| {
            let mut restore_span_occurrences = 0_u64;
            let mut restore_spans_ms = 0.0;
            let mut kv_block_read_occurrences = 0_u64;
            for operation in operations {
                let operation_name = operation.get("operation").and_then(Value::as_str);
                if !RESTORE_OPERATIONS.contains(&operation_name.unwrap_or("")) {
                    continue;
                }
                let occurrence_count = operation
                    .get("occurrence_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                restore_span_occurrences += occurrence_count;
                if operation_name == Some("persistent_prompt_cache_kv_block_read") {
                    kv_block_read_occurrences += occurrence_count;
                }
                restore_spans_ms += operation
                    .get("total_elapsed_nanoseconds")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
                    / 1_000_000.0;
            }
            (
                kv_block_read_occurrences,
                restore_span_occurrences,
                restore_spans_ms,
            )
        })
        .unwrap_or((0, 0, 0.0))
}

fn read_jsonl_lines(path: &std::path::Path) -> Option<Vec<Value>> {
    let log_bytes = std::fs::read(path).ok()?;
    Some(
        log_bytes
            .split(|byte| *byte == b'\n')
            .filter_map(|line_bytes| serde_json::from_slice(line_bytes).ok())
            .collect(),
    )
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
            "the daemon must attribute a positive generation rate to every completion: {} -> {}",
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
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}
