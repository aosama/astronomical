//! Launch and observation helpers for the Ornith-1.5-35B throughput journey.
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
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

pub(super) fn resident_model_id() -> &'static str {
    crate::support::resident_sparse_moe_model_id()
}

pub(super) fn model_directory() -> PathBuf {
    crate::support::configured_installed_model_directory_by_id(resident_model_id())
}

pub(super) fn romeo_and_juliet_prompt() -> String {
    format!(
        "Use the supplied Romeo and Juliet source. Name the two households in one short sentence.\n\n{}",
        ROMEO_AND_JULIET_SOURCE
    )
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

/// Summarizes one generation report's engine-side decode span. Prefill rates
/// deliberately do not come from here: `prompt_prefill_advance_span` counts
/// chunk spans, not tokens, so its derived rate is meaningless and prefill is
/// measured from the supervisor's per-request counters instead.
pub(super) fn decode_span_from_report(report: &serde_json::Value) -> (u64, f64) {
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
    let decode_token_count = decode_operation
        .get("occurrence_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let decode_elapsed_seconds = decode_operation
        .get("total_elapsed_nanoseconds")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0)
        / 1_000_000_000.0;
    (decode_token_count, decode_elapsed_seconds)
}

/// Builds an isolated Development home with per-request attribution enabled
/// and stage-split decode attribution disabled: the stage attribution's
/// engine work roughly doubles decode time and would poison the baseline.
pub(super) fn isolated_home_with_attribution(model_directory: &Path) -> tempfile::TempDir {
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
    config_document["diagnostics"]["performance_attribution_enabled"] = serde_json::json!(true);
    config_document["chunking"]["experimental_decode_stage_attribution_enabled"] =
        serde_json::json!(false);
    std::fs::write(
        isolated_state_directory.join("config.json"),
        config_document.to_string(),
    )
    .expect("isolated config should be written");
    isolated_development_home
}

pub(super) async fn launch_resident_rest_server() -> (tempfile::TempDir, ServingRestServer) {
    let model_id = resident_model_id();
    let model_directory = model_directory();
    eprintln!("[ornith-35b] phase=launch model={model_id}");
    let isolated_development_home = isolated_home_with_attribution(&model_directory);
    let rest_server = launch_serving_rest_server_for_model(
        model_id,
        model_directory,
        Some(isolated_development_home.path()),
        None,
    )
    .await;
    (isolated_development_home, rest_server)
}

pub(super) async fn stop_resident_rest_server(rest_server: ServingRestServer) {
    stop_serving_rest_server(rest_server).await;
    eprintln!("[ornith-35b] phase=done");
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
