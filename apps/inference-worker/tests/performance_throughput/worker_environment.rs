//! The isolated Development home a throughput journey launches its worker into.
//!
//! A throughput measurement is only production-faithful if the worker it
//! measures is the real worker binary, started from the real configuration
//! resolver, with the measured model and nothing else. This module owns that
//! setup and nothing else; driving a completion and reading its numbers back
//! live in `completion`, and the journey itself lives in `support`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use astronomical_ipc_protocol::WorkerStartupConfiguration;
use astronomical_supervisor::{ResolvedRuntimeConfigResolver, RuntimeModelPolicy};

/// Resolves the directory to advertise as the configuration's
/// `model_directories` entry for `model_directory`.
///
/// For a HuggingFace-cache snapshot (`.../models--org--repo/snapshots/<hash>`),
/// returns the `models--org--repo` entry root so discovery derives the decoded
/// `org/repo` identity. For any other layout the directory is already named by
/// its model id, so it is returned unchanged.
fn discovery_root_for_model_directory(model_directory: &Path) -> &Path {
    model_directory
        .ancestors()
        .find(|ancestor| {
            ancestor
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("models--"))
        })
        .unwrap_or(model_directory)
}

/// Builds an isolated Development home pinned to the measured model and resolves
/// the supervisor-owned bootstrap settings for the worker. The worker's logging
/// directory is pointed at a persistent per-model directory so its log lines
/// survive the isolated home's panic-unwind cleanup and stay readable after a
/// failed journey.
///
/// The returned tuple keeps four lifetimes the caller must understand: the
/// `TempDir` must be held for the whole journey or the worker loses its
/// configuration home; the executable path and logging directory are consumed
/// by the launch; the model-policy catalog is moved into the worker handle; and
/// the startup configuration is consumed by the launch.
pub(crate) fn perf_worker_environment(
    model_id: &str,
    model_directory: &Path,
    attribution_enabled: bool,
) -> (
    tempfile::TempDir,
    PathBuf,
    PathBuf,
    Arc<HashMap<String, RuntimeModelPolicy>>,
    WorkerStartupConfiguration,
) {
    let production_worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production inference-worker executable path"),
    );
    let isolated_worker_home =
        tempfile::tempdir().expect("the throughput worker home should be created");
    let configuration_directory = isolated_worker_home.path().join(".astronomical-dev");
    fs::create_dir_all(&configuration_directory)
        .expect("the throughput worker configuration directory should be created");
    // A HuggingFace-cache model must be advertised through its `models--org--repo`
    // entry root, not the raw `snapshots/<hash>` leaf. Discovery only decodes the
    // `org/repo` identity from a `models--` directory; pointed at the snapshot
    // leaf it names the model by the hash, so the requested model id never
    // resolves and the worker rejects the request as unavailable.
    let discovery_root = discovery_root_for_model_directory(model_directory);
    let mut configuration_document = serde_json::json!({
        "model_directories": [discovery_root],
        "persistent_prompt_cache_enabled": false,
    });
    if diagnostics_enabled() || attribution_enabled {
        // Attribution adds host synchronization to every forward and info
        // logging adds I/O, so a diagnostic or evidence run explains where
        // time goes but its throughput is distorted. Production-faithful
        // measured runs leave both off.
        configuration_document["logging"] = serde_json::json!({ "level": "info" });
        configuration_document["performance_attribution_enabled"] = serde_json::Value::Bool(true);
    }
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the throughput worker configuration should serialize"),
    )
    .expect("the throughput worker configuration should be written");
    let resolved_configuration = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_worker_home.path().to_path_buf(),
        PathBuf::from(&production_worker_executable_path),
    )
    .load()
    .expect("the throughput worker configuration should resolve");
    let persistent_logging_directory = std::env::temp_dir()
        .join("astronomical-throughput-logs")
        .join(model_id);
    if persistent_logging_directory.exists() {
        fs::remove_dir_all(&persistent_logging_directory)
            .expect("the stale throughput logging directory should be removed");
    }
    let mut worker_startup_configuration = resolved_configuration.worker_startup_configuration();
    worker_startup_configuration.logging_directory = persistent_logging_directory.clone();
    (
        isolated_worker_home,
        production_worker_executable_path,
        persistent_logging_directory,
        resolved_configuration.model_policy_catalog.clone(),
        worker_startup_configuration,
    )
}

const DIAGNOSTICS_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_THROUGHPUT_DIAGNOSTICS";

/// Diagnostic runs turn on worker attribution and info logging to explain where
/// time goes; their throughput is not production-faithful and is never recorded.
pub(crate) fn diagnostics_enabled() -> bool {
    std::env::var(DIAGNOSTICS_ENVIRONMENT_VARIABLE).is_ok_and(|value| value == "1")
}
