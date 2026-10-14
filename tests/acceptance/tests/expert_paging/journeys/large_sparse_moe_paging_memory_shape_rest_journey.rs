//! Experiment journey: the large sparse MoE is larger than its 32 GB MLX
//! ceiling, so prefill and decode page routed experts from SSD. The journey
//! measures the memory shape across the whole request and the serving rates
//! the user actually experiences.
//!
//! Measurement contract (what a future run reads from this journey):
//!
//! 1. The completion finishes (non-empty text, finish reason "stop" or
//!    "length") and usage reports the requested ~5,000-token prompt.
//! 2. The model pages: expert payload is non-zero and expert storage reads
//!    happened, proving the cell exercises SSD streaming.
//! 3. Active and peak MLX memory stay within the 32 GB ceiling (plus a small
//!    tolerance for allocator rounding that settles before finalization).
//! 4. Prompt-prefill and decode tokens-per-second are reported from the
//!    server-side progress counters (not client wall clock), corroborated by
//!    client-observed stream spans.
//! 5. The sampled memory timeline is printed and preserved as evidence.
//! 6. The worker's performance-attribution reports (model loading and
//!    generation) are printed as complete segment tables and preserved, so
//!    every run attributes elapsed time to specific code segments.

mod memory_observation;
mod support;

use std::{fs, path::Path};

use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::{Duration, Instant as TokioInstant, timeout};

use crate::support::exact_model_prompt::build_exact_model_prompt_content;
use memory_observation::{
    consume_stream_with_timing, observe_memory_shape, print_memory_shape_timeline,
    progress_tokens_per_second,
};

use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::{launch_real_model_rest_server, stop_real_model_rest_server};

fn model_id() -> &'static str {
    crate::support::large_sparse_moe_model_id()
}

// The 8-bit artifact is ~36.8 decimal GB on disk; the default ceiling is ~87%
// of that, so complete residency is impossible and routed experts must page
// from SSD. `PAGING_EXPERIMENT_MLX_MEMORY_GB` sweeps the ceiling so one
// journey binary can probe the complete-residency boundary too.
const DEFAULT_MAXIMUM_MLX_MEMORY_GB: u64 = 32;

fn experiment_maximum_mlx_memory_bytes() -> u64 {
    std::env::var("PAGING_EXPERIMENT_MLX_MEMORY_GB")
        .ok()
        .and_then(|ceiling_gb| ceiling_gb.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAXIMUM_MLX_MEMORY_GB)
        * 1_000_000_000
}

const PROMPT_TOKEN_COUNT: usize = 5_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 500;
const MINIMUM_MEANINGFUL_OUTPUT_TOKEN_COUNT: u64 = 50;
pub(super) const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_millis(500);
pub(super) const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
// The hard SSD-journey timeout rejects slow behavior early so its call stack
// can be isolated with lower-level instrumentation instead of grinding on.
const EXPERIMENT_TIMEOUT: Duration = Duration::from_secs(60);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker to measure paging memory shape and serving rates for the large sparse MoE under the configured ceiling (default 32 GB)"]
#[serial]
async fn should_measure_paging_memory_shape_and_serving_rates_under_the_configured_ceiling() {
    timeout(EXPERIMENT_TIMEOUT, run_paging_memory_shape_experiment())
        .await
        .expect("the paging memory-shape experiment must finish within 60 seconds");
}

async fn run_paging_memory_shape_experiment() {
    let maximum_mlx_memory_bytes = experiment_maximum_mlx_memory_bytes();
    // One percent covers allocator rounding and transient peaks that settle
    // before the finalized snapshot.
    let maximum_mlx_memory_tolerance_bytes = maximum_mlx_memory_bytes / 100;
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id());
    let isolated_worker_home =
        tempfile::tempdir().expect("the paging-experiment worker home should be created");
    write_acceptance_config(
        isolated_worker_home.path(),
        &model_directory,
        maximum_mlx_memory_bytes / 1_000_000_000,
    );
    let user_message = build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Summarize Romeo and Juliet in one concise paragraph. Include the central conflict, major decisions, and tragic outcome.",
        PROMPT_TOKEN_COUNT,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id(),
        model_directory,
        isolated_worker_home.path(),
        maximum_mlx_memory_bytes,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = LocalOpenAiClient::new(server_address, "local-acceptance-client");
    let completion_request = json!({
        "model": model_id(),
        "messages": [{"role": "user", "content": user_message}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    });
    let streamed_completion = openai_client
        .create_streaming_chat_completion(&completion_request)
        .await
        .expect("the public REST summary request should start");
    let request_started_at = TokioInstant::now();
    let (stream_measurement, memory_evidence) = tokio::join!(
        consume_stream_with_timing(streamed_completion, request_started_at),
        observe_memory_shape(server_address),
    );
    let wall_decode_seconds = stream_measurement
        .last_token_elapsed
        .saturating_sub(stream_measurement.first_token_elapsed)
        .as_secs_f64();

    // --- Structural assertion 1: generation completes with real output ---
    assert!(
        !stream_measurement.model_text.is_empty(),
        "the paged model should produce a non-empty completion"
    );
    assert!(
        matches!(
            stream_measurement.finish_reason.as_deref(),
            Some("stop" | "length")
        ),
        "the completion should finish cleanly; finish_reason={:?}",
        stream_measurement.finish_reason
    );

    // --- Structural assertion 2: usage reports the requested prompt size ---
    let usage = stream_measurement
        .usage
        .as_ref()
        .expect("include_usage should report final usage");
    let prompt_token_total = usage["prompt_tokens"]
        .as_u64()
        .expect("usage prompt_tokens");
    let completion_token_total = usage["completion_tokens"]
        .as_u64()
        .expect("usage completion_tokens");
    assert!(
        (PROMPT_TOKEN_COUNT as u64..PROMPT_TOKEN_COUNT as u64 + 1_000)
            .contains(&prompt_token_total),
        "the served prompt should carry the requested {PROMPT_TOKEN_COUNT} input tokens plus template overhead; prompt_tokens={prompt_token_total}"
    );
    assert!(
        completion_token_total >= MINIMUM_MEANINGFUL_OUTPUT_TOKEN_COUNT,
        "the completion should generate a meaningful answer; completion_tokens={completion_token_total}"
    );

    // --- Structural assertion 3: the model paged experts from SSD ---
    let maximum_observed_expert_payload_bytes = memory_evidence
        .samples
        .iter()
        .map(|sample| sample.expert_payload_bytes)
        .max()
        .unwrap_or(0);
    assert!(
        maximum_observed_expert_payload_bytes > 0,
        "the 36.8 GB artifact under a 32 GB ceiling must retain expert payload in RAM while paging"
    );
    stop_real_model_rest_server(real_model_rest_server).await;
    let expert_source_read_byte_count = support::generation_attribution_counter(
        isolated_worker_home.path(),
        "positional_file_read_byte_count",
    );
    let expert_source_read_call_count = support::generation_attribution_counter(
        isolated_worker_home.path(),
        "positional_file_read_call_count",
    );
    // A ceiling that fits the whole artifact legitimately completes with zero
    // request-time expert reads (mode stays resident); a ceiling that cannot
    // hold the artifact must stream expert bytes from storage.
    let complete_residency_cell = memory_evidence.samples.iter().all(|sample| {
        sample.expert_memory_mode != "paged" && sample.expert_memory_mode != "hybrid"
    });
    if !complete_residency_cell {
        assert!(
            expert_source_read_byte_count > 0,
            "a model larger than its ceiling must stream expert bytes from storage during the request"
        );
    }

    // --- Full attribution visibility: every measured segment of the paging
    // call stack, printed and preserved, so slow-downs attribute to code ---
    let attribution_reports = support::attribution_reports(isolated_worker_home.path());
    assert!(
        attribution_reports
            .iter()
            .any(|report| report["report_kind"] == "generation"),
        "the paged generation must flush a generation attribution report"
    );
    assert!(
        attribution_reports
            .iter()
            .any(|report| report["report_kind"] == "model_loading"),
        "the paged worker must flush a model-loading attribution report"
    );
    for attribution_report in &attribution_reports {
        support::print_attribution_report(attribution_report);
    }

    // --- Structural assertion 4: memory stays within the ceiling ---
    let final_snapshot = &memory_evidence.final_status["mlx_memory_snapshot"];
    let final_active_memory_bytes = final_snapshot["active_memory_bytes"]
        .as_u64()
        .expect("the finalized status should report active MLX memory");
    let peak_memory_bytes = final_snapshot["peak_memory_bytes"]
        .as_u64()
        .expect("the finalized status should report peak MLX memory");
    assert!(
        final_active_memory_bytes <= maximum_mlx_memory_bytes,
        "final active memory {final_active_memory_bytes} must stay within ceiling {maximum_mlx_memory_bytes}"
    );
    assert!(
        peak_memory_bytes
            <= maximum_mlx_memory_bytes.saturating_add(maximum_mlx_memory_tolerance_bytes),
        "peak memory {peak_memory_bytes} must stay within ceiling plus tolerance {maximum_mlx_memory_tolerance_bytes}"
    );

    // --- Measured assertion 5: serving rates are positive and finite ---
    // The prompt phase publishes processed_tokens at chunk boundaries while
    // elapsed_ms ticks live, so the prefill rate divides the authoritative
    // usage prompt total by the final observed phase elapsed.
    let prefill_progress = memory_evidence
        .last_prompt_processing_progress
        .expect("the request should observe prompt-processing progress");
    let decode_progress = memory_evidence
        .last_generation_progress
        .expect("the request should observe generation progress");
    let prefill_tokens_per_second = prompt_token_total as f64 / prefill_progress.elapsed_seconds();
    let decode_tokens_per_second = progress_tokens_per_second(&decode_progress);
    assert!(
        prefill_tokens_per_second.is_finite() && prefill_tokens_per_second > 0.0,
        "prompt prefill must report a positive tokens-per-second rate; {prefill_tokens_per_second}"
    );
    assert!(
        decode_tokens_per_second.is_finite() && decode_tokens_per_second > 0.0,
        "decode must report a positive tokens-per-second rate; {decode_tokens_per_second}"
    );

    // --- Evidence: timeline, rates, and attribution survive the run ---
    print_memory_shape_timeline(&memory_evidence.samples);
    eprintln!(
        "[paging-memory-shape] status=summary \
prompt_tokens={prompt_token_total} completion_tokens={completion_token_total} \
prefill_tokens_per_second={prefill_tokens_per_second:.2} \
decode_tokens_per_second={decode_tokens_per_second:.2} \
decode_wall_seconds={wall_decode_seconds:.3} \
peak_memory_gb={:.2} final_active_memory_gb={:.2} \
maximum_expert_payload_gb={:.2} \
expert_source_read_gb={:.2} expert_source_read_calls={expert_source_read_call_count}",
        peak_memory_bytes as f64 / 1_000_000_000.0,
        final_active_memory_bytes as f64 / 1_000_000_000.0,
        maximum_observed_expert_payload_bytes as f64 / 1_000_000_000.0,
        expert_source_read_byte_count as f64 / 1_000_000_000.0,
    );
    let evidence_directory = support::preserve_experiment_evidence(
        isolated_worker_home.path(),
        &json!({
            "model_id": model_id(),
            "maximum_mlx_memory_bytes": maximum_mlx_memory_bytes,
            "prompt_token_target": PROMPT_TOKEN_COUNT,
            "maximum_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
            "usage": usage,
            "serving_rates": {
                "prefill_tokens_per_second": prefill_tokens_per_second,
                "prefill_phase_seconds": prefill_progress.elapsed_seconds(),
                "decode_progress_tokens_per_second": decode_tokens_per_second,
                "decode_progress_tokens": decode_progress.processed_tokens,
                "decode_progress_seconds": decode_progress.elapsed_seconds(),
                "decode_wall_seconds": wall_decode_seconds,
            },
            "memory_shape": {
                "peak_memory_bytes": peak_memory_bytes,
                "final_active_memory_bytes": final_active_memory_bytes,
                "maximum_observed_expert_payload_bytes": maximum_observed_expert_payload_bytes,
                "expert_source_read_byte_count": expert_source_read_byte_count,
                "expert_source_read_call_count": expert_source_read_call_count,
            },
            "timeline": memory_evidence
                .samples
                .iter()
                .map(Value::from)
                .collect::<Vec<_>>(),
            "attribution_reports": attribution_reports,
            "final_mlx_memory_snapshot": final_snapshot,
        }),
    );
    eprintln!(
        "[paging-memory-shape] status=evidence path={}",
        evidence_directory.display()
    );
}

fn write_acceptance_config(
    isolated_worker_home: &Path,
    model_directory: &Path,
    maximum_mlx_memory_gb: u64,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the paging-experiment configuration directory should be created");
    // Experiment overrides let one journey binary sweep paging-cell variables
    // one at a time; the defaults reproduce the production cell, whose
    // SSD-streaming chunk default is 2,048 tokens.
    let ssd_streaming_chunk_tokens: usize =
        std::env::var("PAGING_EXPERIMENT_SSD_STREAMING_CHUNK_TOKENS")
            .ok()
            .and_then(|chunk_tokens| chunk_tokens.parse().ok())
            .unwrap_or(2_048);
    let prefill_graph_submission_layer_interval: u32 =
        std::env::var("PAGING_EXPERIMENT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL")
            .ok()
            .and_then(|layer_interval| layer_interval.parse().ok())
            .unwrap_or(1);
    let generation_graph_submission_layer_interval: u32 =
        std::env::var("PAGING_EXPERIMENT_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL")
            .ok()
            .and_then(|layer_interval| layer_interval.parse().ok())
            .unwrap_or(1);
    eprintln!(
        "[paging-memory-shape] status=cell maximum_mlx_memory_gb={maximum_mlx_memory_gb} ssd_streaming_chunk_tokens={ssd_streaming_chunk_tokens} \
prefill_graph_submission_layer_interval={prefill_graph_submission_layer_interval} \
generation_graph_submission_layer_interval={generation_graph_submission_layer_interval} \
positional_read_parallelism={}",
        std::env::var("ASTRONOMICAL_POSITIONAL_READ_PARALLELISM")
            .unwrap_or_else(|_| "default".to_owned()),
    );
    let configuration_document = json!({
        "model_directories": [crate::support::artifact_bytes::acceptance_model_scan_root(model_directory)],
        "maximum_mlx_memory_gb": maximum_mlx_memory_gb,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        // This cell streams experts. The graph submission intervals control how
        // many decoder layers accumulate before one evaluation submission, which
        // trades submission back-pressure against lazy-tape residency.
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": ssd_streaming_chunk_tokens,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": ssd_streaming_chunk_tokens,
            "prefill_graph_submission_layer_interval": 0,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": prefill_graph_submission_layer_interval,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": generation_graph_submission_layer_interval,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the paging-experiment configuration should serialize"),
    )
    .expect("the paging-experiment configuration should be written");
}
