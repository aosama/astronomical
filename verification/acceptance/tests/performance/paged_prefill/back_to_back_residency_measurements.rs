//! **Performance measurement — run on demand, not in regression runs.**
//! Run with `cargo measure-paged-prefill` when comparing prefill or residency
//! strategies; the regression lane (`cargo test-acceptance`) does not build
//! this target's measurements. Results: issue #1716, PR #1721.
//!
//! Back-to-back requests on the large sparse model (issue #1716).
//!
//! Decode seats every layer at these ceilings while prefill plans against a
//! smaller expert budget, so the second request's prefill can release layers
//! the first request's decode warmed and then stream them again on every
//! chunk. Each journey reports that cost — layers re-streamed, bytes, and
//! each request's memory peak — and pins the invariant any residency change
//! must keep: no request may exceed the configured ceiling.
//!
//! Measured at 37 GiB on a machine whose wired budget caps the effective
//! ceiling at about 39.2 GB: two layers (8.39 GB) are re-streamed per
//! 8,192-token request. A two-layer prefill activation floor removed that
//! churn for 8,192 tokens but caused a rejected resident attempt at 26,549
//! tokens, so the floor needs a measured, context-dependent curve rather than
//! a lower constant (issue #1716).

use std::{collections::BTreeSet, fs, path::Path};

use astronomical_test_harness::{sparse_decode_full_fit, test_run::TestRunContext};
use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::{Duration, timeout};

use crate::support::{
    openai_client::{ChatCompletionStream, LocalOpenAiClient},
    serving_rest::{get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server},
};

const JOURNEY_BUDGET: Duration = Duration::from_secs(115);
const PROMPT_TOKEN_TARGET: usize = 8_192;
const MAXIMUM_OUTPUT_TOKENS: u64 = 32;
const STATS_POLL_ATTEMPTS: usize = 50;
const STATS_POLL_INTERVAL: Duration = Duration::from_millis(100);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 37 GB and measures residency churn between back-to-back requests"]
#[serial]
async fn should_measure_layers_restreamed_between_back_to_back_requests() {
    timeout(JOURNEY_BUDGET, run_back_to_back_journey(37))
        .await
        .expect("the back-to-back journey must finish inside the 120-second runner budget");
}

/// Runs two identical requests and returns how many layers complete after
/// the first were streamed again by the second request's prefill.
async fn run_back_to_back_journey(ceiling_gibibytes: u64) -> usize {
    let model_id = crate::support::large_sparse_moe_streaming_model_id();
    let model_directory = TestRunContext::current()
        .configuration
        .model_directory
        .unwrap_or_else(|| crate::support::configured_installed_model_directory_by_id(model_id));
    let evidence_directory =
        crate::support::acceptance_evidence::acceptance_evidence_root("back-to-back-residency");
    fs::create_dir_all(&evidence_directory).expect("the evidence root should exist");
    let worker_home = tempfile::Builder::new()
        .prefix("worker-")
        .tempdir_in(&evidence_directory)
        .expect("the isolated worker home should exist")
        .keep();
    write_configuration(&worker_home, &model_directory, ceiling_gibibytes);
    let prompt = crate::support::exact_model_prompt::build_model_prompt_content_at_most(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Continue the supplied Romeo and Juliet story.",
        PROMPT_TOKEN_TARGET,
    );
    let server = launch_real_model_rest_server(
        model_id,
        model_directory,
        &worker_home,
        ceiling_gibibytes << 30,
    )
    .await;
    let client = LocalOpenAiClient::new(server.server_address, "back-to-back-residency");
    send_request(&client, model_id, &prompt).await;
    let layers_complete_after_first =
        complete_layer_indices(&poll_until_stats_available(server.server_address).await);
    send_request(&client, model_id, &prompt).await;
    stop_real_model_rest_server(server).await;
    report_and_check(
        &worker_home,
        &layers_complete_after_first,
        ceiling_gibibytes << 30,
    )
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

async fn poll_until_stats_available(server_address: std::net::SocketAddr) -> Value {
    for _poll_attempt in 0..STATS_POLL_ATTEMPTS {
        let stats_document = get_json_endpoint(server_address, "/v1/experts/stats").await;
        if stats_document["available"] == json!(true) {
            return stats_document;
        }
        tokio::time::sleep(STATS_POLL_INTERVAL).await;
    }
    panic!("the expert stats must become available after the first reply finishes");
}

/// Layers whose every expert was seated when the stats were published.
fn complete_layer_indices(stats_document: &Value) -> BTreeSet<u64> {
    let mut complete_layers = BTreeSet::new();
    for layer_entry in stats_document["layer_topology"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if layer_entry["is_complete_layer"] == json!(true)
            && let Some(layer_index) = layer_entry["layer_index"].as_u64()
        {
            complete_layers.insert(layer_index);
        }
    }
    complete_layers
}

fn report_and_check(
    worker_home: &Path,
    layers_complete_after_first: &BTreeSet<u64>,
    ceiling_bytes: u64,
) -> usize {
    let generation_reports = generation_reports(worker_home);
    assert_eq!(
        generation_reports.len(),
        2,
        "both requests must publish a token-producing generation report; found request ids {:?} in {}",
        report_request_ids(&generation_reports),
        worker_home.display()
    );
    let second_request = &generation_reports[1];
    let restreamed_layers =
        prefill_streamed_layers_among(second_request, layers_complete_after_first);
    let restreamed_bytes =
        prefill_streamed_bytes_among(second_request, layers_complete_after_first);
    println!(
        "[back-to-back-residency] complete_after_first={} restreamed_layers={} restreamed_gb={:.2} second_prefill_gb={:.2} peak_first_gb={:.2} peak_second_gb={:.2} ceiling_gb={:.2}",
        layers_complete_after_first.len(),
        restreamed_layers.len(),
        restreamed_bytes as f64 / 1e9,
        prefill_streamed_bytes_among(second_request, &all_layers()) as f64 / 1e9,
        peak_memory_bytes(&generation_reports[0]) as f64 / 1e9,
        peak_memory_bytes(second_request) as f64 / 1e9,
        ceiling_bytes as f64 / 1e9,
    );
    for generation_report in &generation_reports {
        assert!(
            peak_memory_bytes(generation_report) <= ceiling_bytes,
            "a request exceeded the MLX memory ceiling"
        );
    }
    restreamed_layers.len()
}

fn report_request_ids(generation_reports: &[Value]) -> Vec<u64> {
    let mut request_ids = Vec::new();
    for generation_report in generation_reports {
        request_ids.push(generation_report["request_id"].as_u64().unwrap_or(0));
    }
    request_ids
}

fn all_layers() -> BTreeSet<u64> {
    let mut every_layer = BTreeSet::new();
    for layer_index in 0..sparse_decode_full_fit::SPARSE_LAYER_COUNT {
        every_layer.insert(layer_index);
    }
    every_layer
}

fn generation_reports(worker_home: &Path) -> Vec<Value> {
    let attribution_path = worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_text = fs::read_to_string(attribution_path).unwrap_or_default();
    let mut reports = Vec::new();
    for attribution_line in attribution_text.lines() {
        if let Ok(report) = serde_json::from_str::<Value>(attribution_line)
            && report["report_kind"] == "generation"
            && sparse_decode_full_fit::attribution_counter(&report, "generated_token_count")
                .unwrap_or(0)
                > 0
        {
            reports.push(report);
        }
    }
    reports
}

fn prefill_summaries(generation_report: &Value) -> Vec<Value> {
    let mut summaries = Vec::new();
    for source_summary in generation_report["expert_streaming_source_summaries"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if source_summary["phase"] == "prefill" {
            summaries.push(source_summary.clone());
        }
    }
    summaries
}

fn prefill_streamed_layers_among(
    generation_report: &Value,
    candidate_layers: &BTreeSet<u64>,
) -> BTreeSet<u64> {
    let mut streamed_layers = BTreeSet::new();
    for source_summary in prefill_summaries(generation_report) {
        if let Some(layer_index) = source_summary["layer_index"].as_u64()
            && candidate_layers.contains(&layer_index)
        {
            streamed_layers.insert(layer_index);
        }
    }
    streamed_layers
}

fn prefill_streamed_bytes_among(
    generation_report: &Value,
    candidate_layers: &BTreeSet<u64>,
) -> u64 {
    let mut streamed_bytes = 0_u64;
    for source_summary in prefill_summaries(generation_report) {
        if let Some(layer_index) = source_summary["layer_index"].as_u64()
            && candidate_layers.contains(&layer_index)
        {
            streamed_bytes += source_summary["payload_byte_count"].as_u64().unwrap_or(0);
        }
    }
    streamed_bytes
}

fn peak_memory_bytes(generation_report: &Value) -> u64 {
    generation_report["mlx_peak_memory_bytes"]
        .as_u64()
        .expect("a generation report should carry its MLX peak")
}

/// The 37-gigabyte cell's configuration at the requested ceiling.
fn write_configuration(worker_home: &Path, model_directory: &Path, ceiling_gibibytes: u64) {
    let mut configuration_document =
        sparse_decode_full_fit::configuration_document(model_directory);
    configuration_document["runtime"]["maximum_mlx_memory_gb"] = json!(ceiling_gibibytes);
    let state_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&state_directory).expect("the state directory should exist");
    fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the configuration should serialize"),
    )
    .expect("the configuration should be written");
}
