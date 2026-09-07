//! Shared launch and observation helpers for K2 Horizon MoVA REST journeys.
//!
//! Each journey owns its own isolated Development home so a duplicate leaf id
//! under another scan root cannot hide this family from `/v1/models`.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use crate::serving_acceptance::chat::openai_rest::{
    get_endpoint, launch_serving_rest_server_for_model, stop_serving_rest_server,
};
use crate::support::serving_rest::ServingRestServer;

pub(super) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const MAXIMUM_COMPACT_SOURCE_CHARACTERS: usize = 800;
const MAXIMUM_THROUGHPUT_SOURCE_CHARACTERS: usize = 12_000;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

/// Reads the journey's fused-MoE-decode switch; on by default because the
/// long-context journey owns the fused path's throughput claim.
fn fused_moe_decode_enabled() -> bool {
    std::env::var("K2_JOURNEY_FUSED_MOE")
        .map(|value| value != "0")
        .unwrap_or(true)
}

pub(super) fn public_model_id() -> &'static str {
    crate::support::k2_horizon_mova_model_id()
}

pub(super) fn model_directory() -> PathBuf {
    crate::support::configured_installed_model_directory_by_id(public_model_id())
}

pub(super) fn compact_romeo_and_juliet_source() -> String {
    ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(MAXIMUM_COMPACT_SOURCE_CHARACTERS)
        .collect()
}

pub(super) fn households_prompt() -> String {
    format!(
        "Use the supplied Romeo and Juliet source. Name the two households in one short sentence.\n\n{}",
        compact_romeo_and_juliet_source()
    )
}

pub(super) fn throughput_prompt() -> String {
    format!(
        "Use the supplied Romeo and Juliet source. Name the two households in one short sentence.\n\n{}",
        ROMEO_AND_JULIET_SOURCE
            .chars()
            .take(MAXIMUM_THROUGHPUT_SOURCE_CHARACTERS)
            .collect::<String>(),
    )
}

pub(super) fn full_romeo_and_juliet_source() -> &'static str {
    ROMEO_AND_JULIET_SOURCE
}

/// Reads per-request generation attribution reports from an isolated home's
/// worker attribution log. Each report carries the engine-measured decode
/// span statistics, so decode tokens per second come from the owner of the
/// work rather than a client wall clock.
pub(super) fn read_generation_attribution_reports(isolated_home: &Path) -> Vec<serde_json::Value> {
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
            let report: serde_json::Value = serde_json::from_str(line).ok()?;
            (report.get("report_kind")? == "generation").then_some(report)
        })
        .collect()
}

/// Summarizes one generation report's engine-side decode and prefill spans.
pub(super) fn decode_throughput_from_report(report: &serde_json::Value) -> (u64, f64, u64, f64) {
    let operations = report
        .get("operations")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let decode_operation = operations
        .iter()
        .find(|operation| operation["operation"] == "decode_advance_span")
        .cloned()
        .unwrap_or(serde_json::json!({}));
    let prefill_operation = operations
        .iter()
        .find(|operation| operation["operation"] == "prompt_prefill_advance_span")
        .cloned()
        .unwrap_or(serde_json::json!({}));
    let decode_token_count = decode_operation
        .get("occurrence_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let decode_elapsed_seconds = decode_operation
        .get("total_elapsed_nanoseconds")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0)
        / 1_000_000_000.0;
    let prefill_token_span_count = prefill_operation
        .get("occurrence_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let prefill_elapsed_seconds = prefill_operation
        .get("total_elapsed_nanoseconds")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0)
        / 1_000_000_000.0;
    (
        decode_token_count,
        decode_elapsed_seconds,
        prefill_token_span_count,
        prefill_elapsed_seconds,
    )
}

/// Builds an isolated Development home. `attribution_enabled` trades the
/// engine's stage-split decode attribution for clean throughput numbers:
/// attribution-gated stage evaluations roughly double decode time.
pub(super) fn isolated_home_with_overrides(
    model_directory: &Path,
    attribution_enabled: bool,
    quantized_kv_cache_enabled: bool,
) -> tempfile::TempDir {
    let development_config =
        astronomical_config::AstronomicalConfig::load_from_development_location()
            .expect("Development configuration should load");
    let isolated_development_home =
        tempfile::tempdir().expect("isolated Development home should be created");
    let isolated_state_directory = isolated_development_home.path().join(".astronomical-dev");
    std::fs::create_dir_all(&isolated_state_directory)
        .expect("isolated Development state should be created");
    let config_bytes = std::fs::read(development_config.instance_paths().config_file_path())
        .expect("Development config should be readable");
    let mut config_document: Value =
        serde_json::from_slice(&config_bytes).expect("Development config should parse");
    config_document["runtime"]["model_directories"] =
        serde_json::json!([model_directory.to_string_lossy()]);
    config_document["diagnostics"]["performance_attribution_enabled"] =
        serde_json::json!(attribution_enabled);
    config_document["chunking"]["experimental_decode_stage_attribution_enabled"] =
        serde_json::json!(false);
    config_document["chunking"]["experimental_quantized_kv_cache_enabled"] =
        serde_json::json!(quantized_kv_cache_enabled);
    config_document["chunking"]["experimental_fused_moe_decode_enabled"] =
        serde_json::json!(fused_moe_decode_enabled());
    std::fs::write(
        isolated_state_directory.join("config.json"),
        config_document.to_string(),
    )
    .expect("isolated config should be written");
    isolated_development_home
}

pub(super) async fn launch_k2_rest_server() -> (tempfile::TempDir, ServingRestServer) {
    launch_k2_rest_server_with_attribution(false).await
}

pub(super) async fn launch_k2_rest_server_with_attribution(
    attribution_enabled: bool,
) -> (tempfile::TempDir, ServingRestServer) {
    let model_id = public_model_id();
    let model_directory = model_directory();
    eprintln!("[k2-horizon-mova] phase=launch model={model_id}");
    let isolated_development_home =
        isolated_home_with_overrides(&model_directory, attribution_enabled, false);
    let rest_server = launch_serving_rest_server_for_model(
        model_id,
        model_directory,
        Some(isolated_development_home.path()),
        None,
    )
    .await;
    (isolated_development_home, rest_server)
}

/// Launches the K2 journey server with the quantized KV slab enabled, the
/// quality-gated stage-3 mode whose decode bandwidth claim this journey owns.
pub(super) async fn launch_k2_rest_server_with_quantized_kv()
-> (tempfile::TempDir, ServingRestServer) {
    let quantized_kv_cache_enabled = std::env::var("K2_JOURNEY_QUANTIZED_KV")
        .map(|value| value != "0")
        .unwrap_or(true);
    eprintln!(
        "[k2-long-context] quantized_kv_cache_enabled={quantized_kv_cache_enabled} fused_moe_decode_enabled={}",
        fused_moe_decode_enabled(),
    );
    let model_id = public_model_id();
    let model_directory = model_directory();
    eprintln!("[k2-long-context] phase=launch model={model_id}");
    let isolated_development_home =
        isolated_home_with_overrides(&model_directory, true, quantized_kv_cache_enabled);
    let rest_server = launch_serving_rest_server_for_model(
        model_id,
        model_directory,
        Some(isolated_development_home.path()),
        None,
    )
    .await;
    (isolated_development_home, rest_server)
}

pub(super) async fn stop_k2_rest_server(rest_server: ServingRestServer) {
    stop_serving_rest_server(rest_server).await;
    eprintln!("[k2-horizon-mova] phase=done");
}

pub(super) async fn assert_k2_is_advertised(server_address: SocketAddr) {
    let models_response = get_endpoint(server_address, "/v1/models").await;
    let models_document = http_json_body(&models_response);
    let public_model_id = public_model_id();
    let advertised_model = models_document["data"].as_array().and_then(|models| {
        models
            .iter()
            .find(|model| model["id"] == public_model_id)
            .cloned()
    });
    let Some(advertised_model) = advertised_model else {
        panic!("K2 Horizon MoVA {public_model_id} must be advertised: {models_document}");
    };
    assert_eq!(
        advertised_model["supports_reasoning"], true,
        "K2 Horizon MoVA must advertise reasoning: {advertised_model}"
    );
    assert_eq!(
        advertised_model["supports_tool_calls"], true,
        "K2 Horizon MoVA must advertise tool calls: {advertised_model}"
    );
}

pub(super) async fn status_document(server_address: SocketAddr) -> Value {
    http_json_body(&get_endpoint(server_address, "/v1/status").await)
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
        let Some(delta) = chunk
            .pointer("/choices/0/delta")
            .or_else(|| chunk.pointer("/choices/0/message"))
        else {
            continue;
        };
        append_optional_text(&mut streamed_text, delta.get("content"));
        append_optional_text(&mut streamed_text, delta.get("reasoning_content"));
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                append_optional_text(&mut streamed_text, tool_call.pointer("/function/name"));
                append_optional_text(&mut streamed_text, tool_call.pointer("/function/arguments"));
            }
        }
    }
    streamed_text
}

pub(super) fn streamed_responses_text(responses_response: &str) -> String {
    let mut streamed_text = String::new();
    for line in responses_response.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(event) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        append_optional_text(&mut streamed_text, event.get("delta"));
        append_optional_text(&mut streamed_text, event.pointer("/item/content"));
        append_optional_text(&mut streamed_text, event.get("text"));
    }
    streamed_text
}

pub(super) fn names_the_households(streamed_text: &str) -> bool {
    let lowered = streamed_text.to_ascii_lowercase();
    lowered.contains("montague") || lowered.contains("capulet")
}

fn append_optional_text(streamed_text: &mut String, value: Option<&Value>) {
    if let Some(text) = value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        streamed_text.push_str(text);
    }
}
