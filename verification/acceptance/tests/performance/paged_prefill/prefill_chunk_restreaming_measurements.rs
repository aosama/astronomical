//! **Performance measurement — run on demand, not in regression runs.**
//! Run with `cargo measure-paged-prefill` when comparing prefill or residency
//! strategies; the regression lane (`cargo test-acceptance`) does not build
//! this target's measurements. Results: issue #1717, PR #1730.
//!
//! Prefill chunk size versus cold-layer re-streaming (issue #1717).
//!
//! A 2,048-token prefill chunk routes essentially every expert of a layer, so
//! each chunk streams every nonresident layer in full and prefill storage
//! traffic grows as (prompt tokens ÷ chunk size) × cold bytes. These journeys
//! measure the second of two identical requests on the large sparse model
//! (after warm-up, so steady-state re-streaming is not buried under the first
//! request's expert seating) per ceiling, chunk size and prompt size, and
//! report requested and physical storage bytes, chunk count, time and peak.
//! Each asserts prefill never used larger chunks than configured and that the
//! request stayed under the ceiling.
//!
//! Requested and physical bytes are reported separately on purpose: at a
//! high ceiling the few cold layers stay in the operating system file cache,
//! so re-streaming costs almost no SSD traffic, and a chunk-size win there is
//! the page-cache warmth artifact that once misled the default chunk choice.

use std::{fs, path::Path};

use astronomical_test_harness::{sparse_decode_full_fit, test_run::TestRunContext};
use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::{Duration, timeout};

use crate::support::{
    openai_client::{ChatCompletionStream, LocalOpenAiClient},
    serving_rest::{launch_real_model_rest_server, stop_real_model_rest_server},
};

const JOURNEY_BUDGET: Duration = Duration::from_secs(115);
const SHORT_PROMPT_TOKENS: usize = 10_240;
const LONG_PROMPT_TOKENS: usize = 26_552;
const EXTRA_LONG_PROMPT_TOKENS: usize = 40_960;
const MAXIMUM_OUTPUT_TOKENS: u64 = 16;
const RESIDENT_CHUNK_TOKENS: u32 = 2_048;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 37 GB and measures prefill re-streaming at 2,048-token chunks, 10k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_37_gigabytes_with_2048_token_chunks_short_prompt() {
    measure_within_budget(37, 2_048, SHORT_PROMPT_TOKENS).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 37 GB and measures prefill re-streaming at 4,096-token chunks, 10k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_37_gigabytes_with_4096_token_chunks_short_prompt() {
    measure_within_budget(37, 4_096, SHORT_PROMPT_TOKENS).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 37 GB and measures prefill re-streaming at 2,048-token chunks, 26k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_37_gigabytes_with_2048_token_chunks_long_prompt() {
    measure_within_budget(37, 2_048, LONG_PROMPT_TOKENS).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 37 GB and measures prefill re-streaming at 4,096-token chunks, 26k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_37_gigabytes_with_4096_token_chunks_long_prompt() {
    measure_within_budget(37, 4_096, LONG_PROMPT_TOKENS).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 24 GB and measures prefill re-streaming at 2,048-token chunks, 10k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_24_gigabytes_with_2048_token_chunks_short_prompt() {
    measure_within_budget(24, 2_048, SHORT_PROMPT_TOKENS).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 24 GB and measures prefill re-streaming at 4,096-token chunks, 10k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_24_gigabytes_with_4096_token_chunks_short_prompt() {
    measure_within_budget(24, 4_096, SHORT_PROMPT_TOKENS).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 24 GB and measures prefill re-streaming at 2,048-token chunks, 26k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_24_gigabytes_with_2048_token_chunks_long_prompt() {
    measure_within_budget(24, 2_048, LONG_PROMPT_TOKENS).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 24 GB and measures prefill re-streaming at 4,096-token chunks, 26k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_24_gigabytes_with_4096_token_chunks_long_prompt() {
    measure_within_budget(24, 4_096, LONG_PROMPT_TOKENS).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 37 GB and checks 4,096-token SSD-streaming chunks admit a 41k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_37_gigabytes_with_4096_token_chunks_extra_long_prompt()
 {
    measure_within_budget(37, 4_096, EXTRA_LONG_PROMPT_TOKENS).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 24 GB and checks 4,096-token SSD-streaming chunks admit a 41k prompt"]
#[serial]
async fn should_measure_prefill_restreaming_at_24_gigabytes_with_4096_token_chunks_extra_long_prompt()
 {
    measure_within_budget(24, 4_096, EXTRA_LONG_PROMPT_TOKENS).await;
}

async fn measure_within_budget(ceiling_gibibytes: u64, chunk_tokens: u32, prompt_tokens: usize) {
    let journey = measure_prefill_restreaming(ceiling_gibibytes, chunk_tokens, prompt_tokens);
    timeout(JOURNEY_BUDGET, journey)
        .await
        .expect("the prefill re-streaming journey must finish inside the 120-second runner budget");
}

async fn measure_prefill_restreaming(
    ceiling_gibibytes: u64,
    chunk_tokens: u32,
    prompt_tokens: usize,
) {
    let model_id = crate::support::large_sparse_moe_streaming_model_id();
    let model_directory = TestRunContext::current()
        .configuration
        .model_directory
        .unwrap_or_else(|| crate::support::configured_installed_model_directory_by_id(model_id));
    let worker_home = isolated_worker_home();
    write_configuration(
        &worker_home,
        &model_directory,
        ceiling_gibibytes,
        chunk_tokens,
    );
    let prompt = crate::support::exact_model_prompt::build_model_prompt_content_at_most(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Continue the supplied Romeo and Juliet story.",
        prompt_tokens,
    );
    let server = launch_real_model_rest_server(
        model_id,
        model_directory,
        &worker_home,
        ceiling_gibibytes << 30,
    )
    .await;
    let client = LocalOpenAiClient::new(server.server_address, "prefill-chunk-restreaming");
    // The first request after a model load seats the expert library and
    // would bury steady-state re-streaming under warm-up reads; the second
    // request is the one measured.
    send_request(&client, model_id, &prompt).await;
    send_request(&client, model_id, &prompt).await;
    stop_real_model_rest_server(server).await;
    report_and_check(&worker_home, ceiling_gibibytes, chunk_tokens);
}

fn isolated_worker_home() -> std::path::PathBuf {
    let evidence_directory =
        crate::support::acceptance_evidence::acceptance_evidence_root("prefill-chunk-restreaming");
    fs::create_dir_all(&evidence_directory).expect("the evidence root should exist");
    tempfile::Builder::new()
        .prefix("worker-")
        .tempdir_in(&evidence_directory)
        .expect("the isolated worker home should exist")
        .keep()
}

async fn send_request(client: &LocalOpenAiClient, model_id: &str, prompt: &str) {
    let request = json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "max_tokens": MAXIMUM_OUTPUT_TOKENS,
        "temperature": 0,
    });
    let stream = client
        .create_streaming_chat_completion(&request)
        .await
        .expect("the request should start");
    consume_completion(stream).await;
}

async fn consume_completion(mut stream: ChatCompletionStream) {
    let mut finished = false;
    while let Some(fragment) = stream.next().await {
        let fragment = fragment.expect("the completion stream must not fail");
        assert!(fragment["error"].is_null(), "generation failed: {fragment}");
        for choice in fragment["choices"].as_array().into_iter().flatten() {
            finished |= matches!(choice["finish_reason"].as_str(), Some("stop" | "length"));
        }
    }
    assert!(finished, "the completion stream must carry a finish reason");
}

fn report_and_check(worker_home: &Path, ceiling_gibibytes: u64, chunk_tokens: u32) {
    let report = steady_state_generation_report(worker_home);
    let prompt_tokens = counter(&report, "prompt_token_count");
    let chunk_count = counter(&report, "prefill_chunk_count");
    let peak_bytes = report["mlx_peak_memory_bytes"].as_u64().unwrap_or(u64::MAX);
    println!(
        "[prefill-chunk-restreaming] ceiling_gib={ceiling_gibibytes} chunk_tokens={chunk_tokens} prompt_tokens={prompt_tokens} chunks={chunk_count} prefill_requested_gb={:.2} physical_gb={:.2} elapsed_s={:.1} peak_gb={:.2}",
        prefill_streamed_bytes(&report) as f64 / 1e9,
        report["process_physical_disk_read_bytes"]
            .as_u64()
            .unwrap_or(0) as f64
            / 1e9,
        report["report_elapsed_nanoseconds"].as_u64().unwrap_or(0) as f64 / 1e9,
        peak_bytes as f64 / 1e9,
    );
    // Memory admission may split a chunk the budget cannot hold, so the
    // engine can use more chunks than configured, never fewer.
    assert!(
        chunk_count >= prompt_tokens.div_ceil(u64::from(chunk_tokens)),
        "prefill used larger chunks than configured"
    );
    assert!(
        peak_bytes <= ceiling_gibibytes << 30,
        "the request exceeded the MLX memory ceiling"
    );
}

/// The second successful generation: the steady-state request.
fn steady_state_generation_report(worker_home: &Path) -> Value {
    let attribution_path = worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_text = fs::read_to_string(&attribution_path).unwrap_or_default();
    let mut successful_report_count = 0_usize;
    for attribution_line in attribution_text.lines() {
        if let Ok(report) = serde_json::from_str::<Value>(attribution_line)
            && report["report_kind"] == "generation"
            && report["outcome"] == "success"
        {
            successful_report_count += 1;
            if successful_report_count == 2 {
                return report;
            }
        }
    }
    panic!(
        "no steady-state generation report in {}",
        attribution_path.display()
    );
}

fn counter(report: &Value, counter_name: &str) -> u64 {
    sparse_decode_full_fit::attribution_counter(report, counter_name).unwrap_or(0)
}

fn prefill_streamed_bytes(report: &Value) -> u64 {
    let mut streamed_bytes = 0_u64;
    for source_summary in report["expert_streaming_source_summaries"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if source_summary["phase"] == "prefill" {
            streamed_bytes += source_summary["payload_byte_count"].as_u64().unwrap_or(0);
        }
    }
    streamed_bytes
}

fn write_configuration(
    worker_home: &Path,
    model_directory: &Path,
    ceiling_gibibytes: u64,
    chunk_tokens: u32,
) {
    let mut configuration_document =
        sparse_decode_full_fit::configuration_document(model_directory);
    configuration_document["runtime"]["maximum_mlx_memory_gb"] = json!(ceiling_gibibytes);
    // Only the SSD-streaming chunk varies; the resident chunk stays at its
    // default so a result cannot be a resident-chunk effect.
    configuration_document["chunking"]["fixed_prompt_processing_chunk_size_tokens"] =
        json!(RESIDENT_CHUNK_TOKENS);
    configuration_document["chunking"]["fixed_ssd_streaming_prompt_processing_chunk_size_tokens"] =
        json!(chunk_tokens);
    let state_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&state_directory).expect("the state directory should exist");
    fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the configuration should serialize"),
    )
    .expect("the configuration should be written");
}
