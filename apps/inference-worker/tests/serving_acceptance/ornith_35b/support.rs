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

/// Four repetitions of the full source (~31k tokens), for long-context
/// attribution where per-token attention and KV growth costs dominate.
pub(super) fn romeo_and_juliet_long_context_prompt() -> String {
    format!(
        "Use the supplied Romeo and Juliet source. Name the two households in one short sentence.\n\n{}{}{}{}",
        ROMEO_AND_JULIET_SOURCE,
        ROMEO_AND_JULIET_SOURCE,
        ROMEO_AND_JULIET_SOURCE,
        ROMEO_AND_JULIET_SOURCE
    )
}

/// A short Romeo and Juliet excerpt that absorbs the model load and first-use
/// JIT kernel compilation without caching the full measurement prompt, so the
/// measured completion's prefill runs against a cold prompt cache.
pub(super) fn prefill_attribution_warmup_prompt() -> String {
    let excerpt: String = ROMEO_AND_JULIET_SOURCE.chars().take(400).collect();
    format!(
        "Use the supplied Romeo and Juliet excerpt. Name the two households in one short sentence.\n\n{excerpt}"
    )
}

/// Builds an isolated Development home pinned to the production measurement
/// conditions: performance attribution and stage-split decode attribution are
/// both explicitly disabled, so the baseline measures the path users run and
/// a developer's local diagnostics settings cannot leak into the numbers.
pub(super) fn isolated_home_for_measurement(model_directory: &Path) -> tempfile::TempDir {
    isolated_home(model_directory, "ornith-35b-throughput-journey", false)
}

/// Builds the isolated Development home for the prefill attribution journey:
/// identical to the measurement home except performance attribution is
/// enabled, so the per-family prefill graphics-processor completion waits are
/// recorded for the measured completion.
pub(super) fn isolated_home_for_prefill_attribution(model_directory: &Path) -> tempfile::TempDir {
    isolated_home(
        model_directory,
        "ornith-35b-prefill-attribution-journey",
        true,
    )
}

fn isolated_home(
    model_directory: &Path,
    home_prefix: &str,
    performance_attribution_enabled: bool,
) -> tempfile::TempDir {
    let development_config =
        astronomical_config::AstronomicalConfig::load_from_development_location()
            .expect("Development configuration should load");
    // Forensic mode: keep the isolated home so the worker log can be
    // inspected after a failing journey instead of being deleted on unwind.
    let keep_isolated_home = std::env::var("STREAMING_JOURNEY_KEEP_HOME")
        .map(|value| value != "0")
        .unwrap_or(false);
    let mut isolated_home_builder = tempfile::Builder::new();
    isolated_home_builder.prefix(home_prefix);
    isolated_home_builder.disable_cleanup(keep_isolated_home);
    let isolated_development_home = isolated_home_builder
        .tempdir()
        .expect("isolated Development home should be created");
    eprintln!(
        "[ornith-35b] isolated_home={} performance_attribution_enabled={performance_attribution_enabled}",
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
        serde_json::json!([model_directory.to_string_lossy()]);
    config_document["diagnostics"]["performance_attribution_enabled"] =
        serde_json::json!(performance_attribution_enabled);
    config_document["chunking"]["experimental_decode_stage_attribution_enabled"] =
        serde_json::json!(false);
    if let Ok(chunk_override) = std::env::var("STREAMING_JOURNEY_CHUNK_TOKENS") {
        config_document["chunking"]["fixed_prompt_processing_chunk_size_tokens"] = serde_json::json!(
            chunk_override
                .parse::<u32>()
                .expect("valid chunk token count")
        );
    }
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
    let isolated_development_home = isolated_home_for_measurement(&model_directory);
    launch_rest_server_with_home(model_id, model_directory, isolated_development_home).await
}

pub(super) async fn launch_resident_rest_server_with_prefill_attribution()
-> (tempfile::TempDir, ServingRestServer) {
    let model_id = resident_model_id();
    let model_directory = model_directory();
    eprintln!("[ornith-35b-attribution] phase=launch model={model_id}");
    let isolated_development_home = isolated_home_for_prefill_attribution(&model_directory);
    launch_rest_server_with_home(model_id, model_directory, isolated_development_home).await
}

async fn launch_rest_server_with_home(
    model_id: &str,
    model_directory: PathBuf,
    isolated_development_home: tempfile::TempDir,
) -> (tempfile::TempDir, ServingRestServer) {
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
