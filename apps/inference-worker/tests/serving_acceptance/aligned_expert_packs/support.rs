//! Shared launch helpers for the experimental streaming-model journeys.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use crate::serving_acceptance::chat::openai_rest::{
    get_endpoint, launch_serving_rest_server_for_model,
    launch_serving_rest_server_for_model_with_memory_limit, stop_serving_rest_server,
};
use crate::support::serving_rest::ServingRestServer;

pub(super) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const MAXIMUM_COMPACT_SOURCE_CHARACTERS: usize = 800;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

pub(super) const SOURCE_MODEL_ID: &str = "Ornith-1.5-35B-A3B-OptiQ-4bit";
pub(super) const STREAMING_MODEL_ID: &str = "Ornith-1.5-35B-A3B-OptiQ-4bit-expert-streaming";

pub(super) fn streaming_model_directory() -> PathBuf {
    crate::support::configured_installed_model_directory_by_id(STREAMING_MODEL_ID)
}

pub(super) fn households_prompt() -> String {
    format!(
        "Use the supplied Romeo and Juliet source. Name the two households in one short sentence.\n\n{}",
        ROMEO_AND_JULIET_SOURCE
            .chars()
            .take(MAXIMUM_COMPACT_SOURCE_CHARACTERS)
            .collect::<String>(),
    )
}

pub(super) async fn launch_streaming_model_rest_server() -> ServingRestServer {
    eprintln!(
        "[aligned-expert-packs] launching Development REST for {STREAMING_MODEL_ID} (source {SOURCE_MODEL_ID})"
    );
    launch_serving_rest_server_for_model(
        STREAMING_MODEL_ID,
        streaming_model_directory(),
        None,
        None,
    )
    .await
}

/// Launches the streaming model in an isolated Development home with worker
/// attribution enabled and an expert-paging memory cap, so the journey can
/// prove from the worker's own records which storage path served the routed
/// expert pages. The cap sits at roughly half the model's 22 GB footprint:
/// without it a fresh instance admits the full expert set resident and never
/// streams, which is correct policy behavior but proves nothing about paging.
pub(super) async fn launch_streaming_model_rest_server_with_attribution()
-> (tempfile::TempDir, ServingRestServer) {
    const STREAMING_JOURNEY_MLX_MEMORY_CAP_BYTES: u64 = 18_000_000_000;
    eprintln!(
        "[aligned-expert-packs] launching isolated Development REST for {STREAMING_MODEL_ID} with attribution and memory cap"
    );
    let isolated_development_home = attribution_enabled_isolated_home();
    let rest_server = launch_serving_rest_server_for_model_with_memory_limit(
        STREAMING_MODEL_ID,
        streaming_model_directory(),
        Some(isolated_development_home.path()),
        None,
        Some(STREAMING_JOURNEY_MLX_MEMORY_CAP_BYTES),
    )
    .await;
    (isolated_development_home, rest_server)
}

fn attribution_enabled_isolated_home() -> tempfile::TempDir {
    let development_config =
        astronomical_config::AstronomicalConfig::load_from_development_location()
            .expect("Development configuration should load");
    let mut isolated_home_builder = tempfile::Builder::new();
    isolated_home_builder.prefix("aligned-expert-packs-journey");
    // Forensic mode: keep the isolated home so the worker log can be inspected
    // after a failing journey instead of being deleted with the TempDir.
    let keep_isolated_home = std::env::var("STREAMING_JOURNEY_KEEP_HOME")
        .map(|value| value != "0")
        .unwrap_or(false);
    isolated_home_builder.disable_cleanup(keep_isolated_home);
    let isolated_development_home = isolated_home_builder
        .tempdir()
        .expect("isolated Development home should be created");
    eprintln!(
        "[aligned-expert-packs] isolated_home={}",
        isolated_development_home.path().display()
    );
    let isolated_state_directory = isolated_development_home.path().join(".astronomical-dev");
    std::fs::create_dir_all(&isolated_state_directory)
        .expect("isolated Development state should be created");
    let config_bytes = std::fs::read(development_config.instance_paths().config_file_path())
        .expect("Development config should be readable");
    let mut config_document: Value =
        serde_json::from_slice(&config_bytes).expect("Development config should parse");
    config_document["runtime"]["model_directories"] =
        serde_json::json!([streaming_model_directory().to_string_lossy()]);
    config_document["diagnostics"]["performance_attribution_enabled"] = serde_json::json!(true);
    config_document["chunking"]["experimental_decode_stage_attribution_enabled"] =
        serde_json::json!(false);
    config_document["chunking"]["experimental_quantized_kv_cache_enabled"] =
        serde_json::json!(false);
    config_document["chunking"]["experimental_fused_moe_decode_enabled"] = serde_json::json!(false);
    std::fs::write(
        isolated_state_directory.join("config.json"),
        config_document.to_string(),
    )
    .expect("isolated config should be written");
    isolated_development_home
}

/// Reads generation attribution reports from the isolated home's worker log.
pub(super) fn read_generation_attribution_reports(isolated_home: &Path) -> Vec<Value> {
    let attribution_log_path = isolated_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_log_bytes = std::fs::read(attribution_log_path)
        .expect("worker attribution log should be readable after the journey");
    attribution_log_bytes
        .split(|byte| *byte == b'\n')
        .filter_map(|line_bytes| {
            let line = std::str::from_utf8(line_bytes).ok()?;
            let report: Value = serde_json::from_str(line).ok()?;
            (report.get("report_kind")? == "generation").then_some(report)
        })
        .collect()
}

/// Returns the expert streaming source summaries that streamed through
/// per-expert pack files, from any of the supplied reports.
pub(super) fn pack_streamed_source_summaries(reports: &[Value]) -> Vec<Value> {
    reports
        .iter()
        .flat_map(|report| {
            report
                .get("expert_streaming_source_summaries")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .filter(|summary| {
            summary
                .get("streamed_through_expert_packs")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && summary
                    .get("source_plan_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    > 0
        })
        .collect()
}

pub(super) async fn stop_streaming_model_rest_server(rest_server: ServingRestServer) {
    stop_serving_rest_server(rest_server).await;
}

pub(super) async fn assert_streaming_model_is_advertised(server_address: SocketAddr) {
    let models_response = get_endpoint(server_address, "/v1/models").await;
    let models_document = http_json_body(&models_response);
    let advertised_models = models_document
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let advertised_model = advertised_models.iter().find(|model| {
        model
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|model_id| model_id == STREAMING_MODEL_ID)
    });
    assert!(
        advertised_model.is_some(),
        "{STREAMING_MODEL_ID} must be advertised independently of {SOURCE_MODEL_ID}: {models_document}"
    );
}

pub(super) fn http_json_body(http_response: &str) -> Value {
    let response_body = http_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or(http_response);
    serde_json::from_str(response_body)
        .unwrap_or_else(|_| panic!("response should be JSON: {http_response}"))
}

pub(super) fn streamed_chat_text(chat_response: &str) -> String {
    let mut streamed_text = String::new();
    for line in chat_response.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        if let Some(content) = chunk
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            streamed_text.push_str(content);
        }
        if let Some(content) = chunk
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
        {
            streamed_text.push_str(content);
        }
    }
    streamed_text
}

pub(super) fn names_the_households(streamed_text: &str) -> bool {
    let lowered = streamed_text.to_ascii_lowercase();
    lowered.contains("montague") || lowered.contains("capulet")
}
