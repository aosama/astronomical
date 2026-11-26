//! Launching the idle fixture worker and resolving the configuration a
//! daemon application starts from, shared by the live-reload journeys.

use std::collections::HashMap;

use super::*;

pub(super) fn idle_worker_executable_path() -> PathBuf {
    PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-daemon-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    )
}

/// Launches the idle worker with a performance log under the config home.
/// Callers wait for readiness themselves because each journey waits for a
/// different worker signal.
pub(super) async fn launch_idle_worker(
    worker_executable_path: &std::path::Path,
    config_home_directory: &std::path::Path,
    model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
) -> WorkerHandle {
    let performance_log_directory = config_home_directory.join("performance");
    std::fs::create_dir_all(&performance_log_directory)
        .expect("the performance log directory should be created");
    WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(2),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the performance log should open"),
        model_policy_catalog,
    )
    .await
    .expect("the idle worker should launch")
}

/// Resolves the starting configuration from the config file on disk and
/// points it at the idle worker so a reload restarts the same fixture.
pub(super) fn resolve_initial_config(
    config_home_directory: &std::path::Path,
    worker_executable_path: &std::path::Path,
) -> (ResolvedRuntimeConfigResolver, ResolvedRuntimeConfig) {
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.to_path_buf(),
        worker_executable_path.to_path_buf(),
    );
    let mut initial_resolved_config = runtime_config_resolver
        .load()
        .expect("initial config should resolve");
    initial_resolved_config.worker_executable_path = worker_executable_path.to_path_buf();
    (runtime_config_resolver, initial_resolved_config)
}

pub(super) async fn shut_down_worker(worker_handle: WorkerHandle) {
    worker_handle
        .shutdown()
        .await
        .expect("the worker should shut down");
}
