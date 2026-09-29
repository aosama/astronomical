//! Launch and observation helpers for the Qwen3.8-35B-A3B-Distill journey.
//!
//! The journey owns an isolated Development home so a duplicate leaf id under
//! another scan root cannot hide this family from `/v1/models`, and so the
//! attribution log it reads belongs to this run alone.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use crate::serving_acceptance::chat::openai_rest::{
    get_endpoint, launch_serving_rest_server_for_model, stop_serving_rest_server,
};
use crate::support::serving_rest::ServingRestServer;

pub(super) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
pub(super) const QWEN3_8_MODEL_ID: &str = "Qwen3.8-35B-A3B-Distill-oQ6e-mtp";
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

pub(super) fn model_directory() -> PathBuf {
    crate::support::configured_installed_model_directory_by_id(QWEN3_8_MODEL_ID)
}

pub(super) fn romeo_and_juliet_prompt() -> String {
    format!(
        "Use the supplied Romeo and Juliet source. Name the two households in one short sentence.\n\n{}",
        ROMEO_AND_JULIET_SOURCE
    )
}

/// Builds an isolated Development home pinned to the production measurement
/// conditions: performance attribution and stage-split decode attribution are
/// both explicitly disabled, so the baseline measures the path users run and
/// a developer's local diagnostics settings cannot leak into the numbers.
///
/// The model directory given to discovery is the `models--org--repo` cache
/// entry, not the snapshot inside it: discovery decodes that directory name
/// into the leaf model ID the worker reports and requests must address, while
/// a snapshot directory would register the model under its commit hash.
pub(super) fn isolated_home_for_measurement(model_directory: &Path) -> tempfile::TempDir {
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
    let discovery_root_directory = model_directory
        .ancestors()
        .find(|ancestor| {
            ancestor
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("models--"))
        })
        .unwrap_or(model_directory);
    config_document["runtime"]["model_directories"] =
        serde_json::json!([discovery_root_directory.to_string_lossy()]);
    config_document["diagnostics"]["performance_attribution_enabled"] = serde_json::json!(false);
    config_document["chunking"]["experimental_decode_stage_attribution_enabled"] =
        serde_json::json!(false);
    std::fs::write(
        isolated_state_directory.join("config.json"),
        config_document.to_string(),
    )
    .expect("isolated config should be written");
    isolated_development_home
}

pub(super) async fn launch_rest_server() -> (tempfile::TempDir, ServingRestServer) {
    eprintln!("[qwen3-8] phase=launch model={QWEN3_8_MODEL_ID}");
    let isolated_development_home = isolated_home_for_measurement(&model_directory());
    let rest_server = launch_serving_rest_server_for_model(
        QWEN3_8_MODEL_ID,
        model_directory(),
        Some(isolated_development_home.path()),
        None,
    )
    .await;
    (isolated_development_home, rest_server)
}

pub(super) async fn stop_rest_server(rest_server: ServingRestServer) {
    stop_serving_rest_server(rest_server).await;
    eprintln!("[qwen3-8] phase=done");
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
