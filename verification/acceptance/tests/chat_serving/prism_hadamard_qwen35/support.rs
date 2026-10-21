//! Shared launch and observation helpers for folded-ternary REST journeys.
//!
//! Each journey owns its own isolated Development home whose model roots
//! point at the installed pack's Hugging Face cache directory, so discovery
//! decodes the real `models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit`
//! identity instead of trusting a fabricated one.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_config::{
    ChatModelCapabilities, DiscoveredModel, ModelCapabilities, ModelFamily, ModelLicense,
};

use crate::support::serving_rest::ServingRestServer;
use crate::support::serving_rest::launch_serving_rest_server_for_artifact_with_memory_limit;
use crate::support::serving_rest::stop_serving_rest_server;

pub(super) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);

/// The environment variable pointing at the installed folded-pack snapshot
/// (the directory containing config.json, hadamard.json, tokenizer.json, and
/// model.safetensors).
const PACK_SNAPSHOT_ENV: &str = "PRISM_HADAMARD_QWEN35_PACK_DIR";

/// The decoded Hugging Face identity of the installed folded pack.
pub(super) fn public_model_id() -> &'static str {
    "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
}

/// Resolves the installed snapshot directory, refusing to guess a path.
fn pack_snapshot_directory() -> PathBuf {
    match std::env::var_os(PACK_SNAPSHOT_ENV) {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => panic!(
            "set {PACK_SNAPSHOT_ENV} to the installed folded-pack snapshot directory \
             (containing config.json, hadamard.json, tokenizer.json, and model.safetensors)"
        ),
    }
}

/// The Hugging Face cache directory holding the pack's `models--` tree: the
/// discovery root whose `models--org--repo/snapshots/<revision>` layout the
/// scanner decodes into the real provider identity and revision.
fn pack_hub_directory() -> PathBuf {
    let snapshot = pack_snapshot_directory();
    snapshot
        .parent()
        .and_then(|snapshots_directory| snapshots_directory.parent())
        .map(Path::to_path_buf)
        .expect("the pack snapshot must live under a models-- repository tree")
}

/// The DiscoveredModel projection of the installed pack, matching what disk
/// discovery derives from the real documents.
fn prism_discovered_model_artifact() -> DiscoveredModel {
    let snapshot = pack_snapshot_directory();
    let weights_bytes = std::fs::metadata(snapshot.join("model.safetensors"))
        .expect("the pack weights should exist")
        .len();
    DiscoveredModel {
        model_id: public_model_id().to_owned(),
        provider_model_id: Some(public_model_id().to_owned()),
        model_family: ModelFamily::PrismHadamardQwen35,
        revision: "fcba37d2117a7077eac6b613b2668d14d9779edd".to_owned(),
        model_directory: snapshot,
        capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: 262_144,
            max_input_tokens: 262_143,
            max_output_tokens: 4096,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: Some(ModelLicense::Apache20),
        model_size_bytes: weights_bytes,
    }
}

/// Builds one isolated Development home whose model roots hold only the
/// folded pack's repository tree.
fn isolated_home_for_pack() -> tempfile::TempDir {
    let isolated_development_home = crate::support::isolated_development_home_from_user_config();
    let isolated_state_directory = isolated_development_home.path().join(".astronomical-dev");
    let config_path = isolated_state_directory.join("config.json");
    let mut config_document: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path).expect("the isolated config should be readable"),
    )
    .expect("the isolated config should parse");
    config_document["runtime"]["model_directories"] =
        serde_json::json!([pack_hub_directory().to_string_lossy()]);
    config_document["diagnostics"]["performance_attribution_enabled"] = serde_json::json!(true);
    std::fs::write(&config_path, config_document.to_string())
        .expect("the isolated config should be rewritten");
    isolated_development_home
}

/// Launches the serving REST surface for the folded pack.
pub(super) async fn launch_prism_rest_server() -> (tempfile::TempDir, ServingRestServer) {
    let isolated_development_home = isolated_home_for_pack();
    eprintln!("[prism-hadamard] phase=launch model={}", public_model_id());
    let rest_server = launch_serving_rest_server_for_artifact_with_memory_limit(
        prism_discovered_model_artifact(),
        Some(isolated_development_home.path()),
        None,
        None,
    )
    .await;
    (isolated_development_home, rest_server)
}

/// Stops the serving REST surface.
pub(super) async fn stop_prism_rest_server(rest_server: ServingRestServer) {
    stop_serving_rest_server(rest_server).await;
    eprintln!("[prism-hadamard] phase=done");
}

/// The serving address of a launched REST server.
pub(super) fn server_address(rest_server: &ServingRestServer) -> SocketAddr {
    rest_server.server_address
}
