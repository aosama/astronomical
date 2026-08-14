use std::{fs, path::PathBuf};

use astronomical_ipc_protocol::WorkerStartupConfiguration;
use astronomical_supervisor::ResolvedRuntimeConfigResolver;

pub(super) fn create_cache_stats_worker_configuration() -> (
    tempfile::TempDir,
    PathBuf,
    PathBuf,
    PathBuf,
    WorkerStartupConfiguration,
) {
    let configured_model_directory = crate::common::configured_model_artifact_directory_by_id(
        super::persistent_prompt_cache_stats_e2e::MODEL_ID,
    );
    let configured_prompt_cache_maximum_size_gb = configured_prompt_cache_maximum_size_gb();
    let production_worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production inference-worker executable path"),
    );
    let isolated_worker_home =
        tempfile::tempdir().expect("the cache-stats worker home should be created");
    let isolated_configuration_directory = isolated_worker_home.path().join(".astronomical-dev");
    fs::create_dir(&isolated_configuration_directory)
        .expect("the cache-stats worker configuration directory should be created");
    let persistent_prompt_cache_directory_path = isolated_configuration_directory.join("cache");
    fs::create_dir(&persistent_prompt_cache_directory_path)
        .expect("the cache-stats prompt-cache directory should be created");
    let performance_log_directory = isolated_worker_home.path().join("logs");
    fs::create_dir(&performance_log_directory)
        .expect("the cache-stats performance log directory should be created");
    let worker_configuration_document = serde_json::json!({
        "model_directories": [configured_model_directory],
        "prompt_cache_max_size_gb": configured_prompt_cache_maximum_size_gb,
    });
    fs::write(
        isolated_configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&worker_configuration_document)
            .expect("the cache-stats worker configuration should serialize"),
    )
    .expect("the cache-stats worker configuration should be written");

    let worker_runtime_config = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_worker_home.path().to_path_buf(),
        production_worker_executable_path.clone(),
    )
    .load()
    .expect("the cache-stats worker configuration should resolve");

    (
        isolated_worker_home,
        persistent_prompt_cache_directory_path,
        production_worker_executable_path,
        configured_model_directory,
        worker_runtime_config.worker_startup_configuration(),
    )
}

fn configured_prompt_cache_maximum_size_gb() -> u64 {
    let home_directory = std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("HOME should be set for the cache-stats E2E test");
    let config_file_path = home_directory.join(".astronomical-dev/config.json");
    let config_file_bytes = fs::read(&config_file_path)
        .expect("Development config should be readable for the cache-stats E2E test");
    let config_document: serde_json::Value = serde_json::from_slice(&config_file_bytes)
        .expect("Development config should be valid JSON");
    config_document
        .get("prompt_cache_max_size_gb")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(astronomical_config::DEFAULT_PROMPT_CACHE_MAXIMUM_SIZE_GB)
}
