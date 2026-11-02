//! **Performance measurement — run on demand, not in regression runs.**
//! Run with `cargo measure-paged-prefill` when comparing prefill or residency
//! strategies; the regression lane (`cargo test-acceptance`) does not build
//! this target's measurements. Results: issue #1716, PR #1721.
//!
//! Back-to-back requests on the large sparse model (issue #1716).
//!
//! Decode seats every layer at these ceilings while prefill plans against a
//! smaller expert budget, so each later request's prefill can release layers
//! the previous decode warmed and then stream them again on every chunk. The
//! journey sends four identical requests so the report shows the steady state
//! a chat session reaches, not only the first hand-over. For every request
//! after the first it reports layers re-streamed and bytes, and it pins the
//! invariant any residency change must keep: no request may exceed the
//! configured ceiling.
//!
//! Measured at 37 GiB (effective ceiling about 39.2 GB, 4,096-token chunks):
//! requests 2-4 each re-stream two layers (5.09 GB). The layers leave through
//! generation-context admission, which reserves a three-layer activation
//! floor and one expert page while every layer is seated. Halving the
//! re-streamed bytes with measured activation evidence did not make prefill
//! faster (3.9 s against 4.0 s), because those layers are served from the
//! file cache while compute runs; see issue #1716 before changing residency
//! for speed.
//!
//! Prefer the smaller-model cell for timing comparisons: the 37 GiB cell can
//! drive the operating system into swap on a machine of about 50 GB, while
//! the 20 GiB cell measured no swap-outs. There, requests 2-4 stream 12.18 GB
//! in about 3.35 s of prefill. The same activation-evidence change streamed
//! 60 GB and was 50% slower in that cell: its forwards measured no
//! transient, the reserve fell to one layer, and the planner seated layers
//! the per-chunk guard then had to evict, so the static three-layer floor is
//! load-bearing there.

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
const REQUEST_COUNT: usize = 4;
const STATS_POLL_ATTEMPTS: usize = 50;
const STATS_POLL_INTERVAL: Duration = Duration::from_millis(100);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 37 GB and measures residency churn between back-to-back requests"]
#[serial]
async fn should_measure_layers_restreamed_between_back_to_back_requests() {
    timeout(
        JOURNEY_BUDGET,
        run_back_to_back_journey(crate::support::large_sparse_moe_streaming_model_id(), 37),
    )
    .await
    .expect("the back-to-back journey must finish inside the 120-second runner budget");
}

/// The same journey on the smaller 4-bit sparse model under a lowered ceiling.
/// It forces several layers to stream while leaving most of the machine's RAM
/// to the operating system, so prefill timings are not distorted by system
/// swapping the way the 37 GiB cell on the 8-bit model can be.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the 4-bit sparse model at 20 GB and measures residency churn between back-to-back requests"]
#[serial]
async fn should_measure_layers_restreamed_for_the_smaller_sparse_model_under_a_lowered_ceiling() {
    timeout(
        JOURNEY_BUDGET,
        run_back_to_back_journey(crate::support::resident_sparse_moe_model_id(), 20),
    )
    .await
    .expect("the back-to-back journey must finish inside the 120-second runner budget");
}

/// Runs identical requests back to back and returns, for each request after
/// the first, how many layers complete before it were streamed again by its
/// prefill. Several requests show the steady state a chat session reaches,
/// not only the first hand-over from decode to prefill.
async fn run_back_to_back_journey(model_id: &str, ceiling_gibibytes: u64) -> Vec<usize> {
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
    let mut layers_complete_before_each_request = Vec::new();
    for _request_index in 0..REQUEST_COUNT {
        send_request(&client, model_id, &prompt).await;
        layers_complete_before_each_request.push(complete_layer_indices(
            &poll_until_stats_available(server.server_address).await,
        ));
    }
    stop_real_model_rest_server(server).await;
    report_and_check(
        &worker_home,
        &layers_complete_before_each_request,
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
    layers_complete_after_each_request: &[BTreeSet<u64>],
    ceiling_bytes: u64,
) -> Vec<usize> {
    let generation_reports = generation_reports(worker_home);
    assert_eq!(
        generation_reports.len(),
        REQUEST_COUNT,
        "every request must publish a token-producing generation report; found request ids {:?} in {}",
        report_request_ids(&generation_reports),
        worker_home.display()
    );
    let mut restreamed_layer_counts = Vec::new();
    for request_index in 1..REQUEST_COUNT {
        let layers_complete_before_request = &layers_complete_after_each_request[request_index - 1];
        let generation_report = &generation_reports[request_index];
        let restreamed_layers =
            prefill_streamed_layers_among(generation_report, layers_complete_before_request);
        println!(
            "[back-to-back-residency] request={} complete_before={} restreamed_layers={} restreamed_gb={:.2} prefill_streamed_gb={:.2} peak_gb={:.2} ceiling_gb={:.2}",
            request_index + 1,
            layers_complete_before_request.len(),
            restreamed_layers.len(),
            prefill_streamed_bytes_among(generation_report, layers_complete_before_request) as f64
                / 1e9,
            prefill_streamed_bytes_among(generation_report, &all_layers()) as f64 / 1e9,
            peak_memory_bytes(generation_report) as f64 / 1e9,
            ceiling_bytes as f64 / 1e9,
        );
        restreamed_layer_counts.push(restreamed_layers.len());
    }
    for generation_report in &generation_reports {
        assert!(
            peak_memory_bytes(generation_report) <= ceiling_bytes,
            "a request exceeded the MLX memory ceiling"
        );
    }
    restreamed_layer_counts
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
