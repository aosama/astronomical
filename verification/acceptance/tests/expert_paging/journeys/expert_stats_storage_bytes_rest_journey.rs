//! Real-model acceptance journey for the observatory storage-byte numbers.
//!
//! The Experts page is a measuring instrument, so its storage figures must
//! reconcile with the per-request attribution evidence the worker writes
//! (issue #1715): the requested bytes equal the streamed source plans, the
//! physical disk figure never exceeds what was requested, and the avoided
//! bytes never exceed what a no-cache server would have read.
//!
//! Runs the large sparse mixture-of-experts model under a constrained ceiling
//! so expert paging actually streams from storage, with the Shakespeare
//! fixture its neighbouring journeys use.

use std::path::Path;

use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use std::fs;
use tokio::time::{Duration, timeout};

use crate::support::openai_client::{ChatCompletionStream, LocalOpenAiClient};
use crate::support::serving_rest::{
    SSD_JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server,
    stop_real_model_rest_server,
};

fn model_id() -> &'static str {
    crate::support::large_sparse_moe_streaming_model_id()
}

// This ceiling defines a reproducible acceptance cell only, matching the
// neighbouring SSD-paging journeys; production code must not hardwire it.
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 23_000_000_000;
const PROMPT_TOKEN_COUNT: usize = 2_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 256;
const STATS_POLL_ATTEMPTS: usize = 20;
const STATS_POLL_INTERVAL: Duration = Duration::from_millis(250);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

/// Top-k experts per token for this model family. The avoided-bytes baseline
/// scales with it; the constant is model geometry, not machine configuration.
const EXPERTS_PER_TOKEN: u64 = 8;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker against the large sparse MoE streaming model to accept the observatory storage-byte accounting"]
#[serial]
async fn should_report_storage_bytes_that_match_the_source_summaries() {
    timeout(
        SSD_JOURNEY_TIMEOUT,
        run_expert_stats_storage_bytes_journey(),
    )
    .await
    .expect("the expert-stats storage-byte journey must finish within 120 seconds");
}

async fn run_expert_stats_storage_bytes_journey() {
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id());
    let isolated_worker_home =
        tempfile::tempdir().expect("the expert-stats worker home should be created");
    write_expert_stats_acceptance_config(isolated_worker_home.path(), &model_directory);
    let user_message = crate::support::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Summarize Romeo and Juliet in one concise paragraph.",
        PROMPT_TOKEN_COUNT,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id(),
        model_directory,
        isolated_worker_home.path(),
        MAXIMUM_MLX_MEMORY_BYTES,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = LocalOpenAiClient::new(server_address, "local-acceptance-client");
    let completion_request = json!({
        "model": model_id(),
        "messages": [{"role": "user", "content": user_message}],
        "stream": true,
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    });
    let streamed_completion: ChatCompletionStream = openai_client
        .create_streaming_chat_completion(&completion_request)
        .await
        .expect("the public REST summary request should start");
    let observed_output_characters = consume_stream_output_characters(streamed_completion).await;
    assert!(
        observed_output_characters > 0,
        "the journey's request must produce output before its statistics are read"
    );

    let stats_document = poll_until_expert_stats_available(server_address).await;
    stop_real_model_rest_server(real_model_rest_server).await;

    assert_requested_bytes_match_source_summaries(&stats_document, isolated_worker_home.path());
    assert_physical_bytes_never_exceed_requested(&stats_document);
    assert_avoided_bytes_never_exceed_no_cache_baseline(&stats_document);
    eprintln!(
        "[expert-stats-storage-bytes] requested_gb={:.2} physical_gb={:.2} avoided_gb={:.2} summaries_gb={:.2}",
        requested_bytes(&stats_document) as f64 / 1e9,
        stats_document["value"]["expert_source_physical_disk_read_bytes"]
            .as_u64()
            .unwrap_or(0) as f64
            / 1e9,
        avoided_bytes(&stats_document) as f64 / 1e9,
        streamed_source_plan_bytes(isolated_worker_home.path()) as f64 / 1e9,
    );
}

/// Counts the streamed completion's output characters so the journey fails
/// loudly if the request never produced evidence to account for.
async fn consume_stream_output_characters(mut streamed_completion: ChatCompletionStream) -> usize {
    let mut observed_output_characters = 0_usize;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.expect("the public REST stream should remain healthy");
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                observed_output_characters += content_fragment.len();
            }
        }
    }
    observed_output_characters
}

/// Mirrors the neighbouring SSD-paging cell: attribution stays enabled so the
/// per-request source summaries exist to reconcile against, and paging
/// streams experts under the constrained ceiling.
fn write_expert_stats_acceptance_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the expert-stats configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": 23,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2_048,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 2_048,
            "prefill_graph_submission_layer_interval": 0,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 1,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the expert-stats configuration should serialize"),
    )
    .expect("the expert-stats configuration should be written");
}

/// Polls the observatory stats endpoint until the finalized generation's
/// event has reached the daemon; the document must be available before the
/// server stops or its evidence is lost.
async fn poll_until_expert_stats_available(server_address: std::net::SocketAddr) -> Value {
    for _poll_attempt in 0..STATS_POLL_ATTEMPTS {
        let stats_document = get_json_endpoint(server_address, "/v1/experts/stats").await;
        if stats_document["available"] == serde_json::json!(true) {
            return stats_document;
        }
        tokio::time::sleep(STATS_POLL_INTERVAL).await;
    }
    panic!("the observatory expert-stats document must become available after the reply finishes");
}

fn requested_bytes(stats_document: &Value) -> u64 {
    stats_document["value"]["expert_source_read_byte_count"]
        .as_u64()
        .expect("the stats document should report requested expert source bytes")
}

fn avoided_bytes(stats_document: &Value) -> u64 {
    stats_document["value"]["avoided_complete_layer_source_bytes"]
        .as_u64()
        .unwrap_or(0)
}

/// Sums every generation report's per-layer, per-phase source summaries.
fn streamed_source_plan_bytes(attribution_home: &Path) -> u64 {
    attribution_report_documents(attribution_home)
        .into_iter()
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .flat_map(|attribution_report| {
            attribution_report["expert_streaming_source_summaries"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|source_summary| source_summary["payload_byte_count"].as_u64())
        .sum()
}

/// Sums the streamed-expert payload counter over every non-generation report:
/// streams the model-loading phase issued (for example planned complete-layer
/// seating) reach the session counters but never a generation summary, so the
/// requested-vs-summaries equality must add them back.
fn model_loading_streamed_payload_bytes(attribution_home: &Path) -> u64 {
    attribution_report_documents(attribution_home)
        .into_iter()
        .filter(|attribution_report| attribution_report["report_kind"] != "generation")
        .flat_map(|attribution_report| {
            attribution_report["counters"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter(|counter_report| {
            counter_report["counter"] == "rust_expert_streaming_payload_byte_count"
        })
        .filter_map(|counter_report| counter_report["amount"].as_u64())
        .sum()
}

fn attribution_report_documents(attribution_home: &Path) -> Vec<Value> {
    let attribution_log_path = attribution_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_log_text = std::fs::read_to_string(attribution_log_path).unwrap_or_default();
    attribution_log_text
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .collect()
}

/// Requested bytes are exactly what the streamed source plans asked storage
/// for: the generation's per-layer summaries plus any model-loading streams.
fn assert_requested_bytes_match_source_summaries(stats_document: &Value, attribution_home: &Path) {
    let expected_requested_bytes = streamed_source_plan_bytes(attribution_home)
        + model_loading_streamed_payload_bytes(attribution_home);
    assert_eq!(
        requested_bytes(stats_document),
        expected_requested_bytes,
        "the stats document's requested bytes must equal the source-plan summaries the attribution reports recorded"
    );
}

/// The operating system file cache can only remove physical reads, never add
/// them, so the physical figure cannot honestly exceed the requested figure.
fn assert_physical_bytes_never_exceed_requested(stats_document: &Value) {
    let physical_disk_read_bytes =
        stats_document["value"]["expert_source_physical_disk_read_bytes"]
            .as_u64()
            .expect("macOS acceptance runs must sample process physical disk reads");
    assert!(
        physical_disk_read_bytes <= requested_bytes(stats_document),
        "physical reads {physical_disk_read_bytes} must never exceed requested bytes {}",
        requested_bytes(stats_document)
    );
}

/// Pure: the storage bytes a no-cache server would have read for this
/// session. Decode layer-tokens each read at most their routed assignments'
/// payload; prefill forwards read at most every prompt token's top-k per
/// sparse layer. Avoided bytes above this baseline are fabrication.
fn no_cache_session_baseline_bytes(stats_document: &Value) -> u64 {
    let library_bytes = stats_document["library"]["complete_expert_payload_byte_count"]
        .as_u64()
        .unwrap_or(0);
    let expert_capacity = stats_document["library"]["expert_capacity"]
        .as_u64()
        .unwrap_or(1);
    let total_layer_count = stats_document["residency"]["total_layer_count"]
        .as_u64()
        .unwrap_or(0);
    let retained_assignments = stats_document["value"]["retained_route_assignment_count"]
        .as_u64()
        .unwrap_or(0);
    let session_token_count = stats_document["value"]["session_prompt_token_count"]
        .as_u64()
        .unwrap_or(0)
        + stats_document["value"]["session_generated_token_count"]
            .as_u64()
            .unwrap_or(0);
    let bytes_per_expert = library_bytes / expert_capacity.max(1);
    let decode_baseline_assignments = retained_assignments;
    let prefill_baseline_assignments = session_token_count
        .saturating_mul(EXPERTS_PER_TOKEN)
        .saturating_mul(total_layer_count);
    bytes_per_expert
        .saturating_mul(decode_baseline_assignments.saturating_add(prefill_baseline_assignments))
}

fn assert_avoided_bytes_never_exceed_no_cache_baseline(stats_document: &Value) {
    let no_cache_baseline_bytes = no_cache_session_baseline_bytes(stats_document);
    assert!(
        avoided_bytes(stats_document) <= no_cache_baseline_bytes,
        "avoided bytes {} must never exceed the no-cache baseline {no_cache_baseline_bytes}",
        avoided_bytes(stats_document),
    );
}
