//! Pure config-reload diff logic tests.
//!
//! These tests verify the classification of changed config fields into
//! reload, worker-restart, and rest-api-restart categories without
//! starting a worker process or HTTP listener.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use astronomical_config::{LogLevel, LoggingConfig, PrefillChunckSizingPolicy, PromptCacheConfig};
use astronomical_supervisor::{
    ChatGenerationExecutor, ConfigReloadDecision, ConfigReloadDiff, GenerationPerformanceLog,
    ResolvedRuntimeConfig, ResolvedRuntimeConfigResolver, WorkerHandle, WorkerHealthStatus,
};
use tokio::time::{Instant, sleep};

#[test]
fn should_mark_bind_address_changes_as_rest_api_restart_required() {
    let current = resolved_config_with_bind_address("127.0.0.1:6732");
    let candidate = resolved_config_with_bind_address("127.0.0.1:6733");

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    match decision {
        ConfigReloadDecision::RestApiRestartRequired {
            ref restart_required_fields,
            ..
        } if restart_required_fields.contains(&"supervisor.bind_address".to_owned()) => {}
        unexpected => {
            panic!("a bind address change must require a full REST API restart, got {unexpected:?}")
        }
    }
}

#[test]
fn should_mark_logging_changes_as_rest_api_restart_required() {
    let current = resolved_config_with_logging_directory("/tmp/astronomical-logs-a");
    let candidate = resolved_config_with_logging_directory("/tmp/astronomical-logs-b");

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    match decision {
        ConfigReloadDecision::RestApiRestartRequired {
            ref restart_required_fields,
            ..
        } if restart_required_fields.contains(&"logging".to_owned()) => {}
        unexpected => panic!(
            "a logging directory change must require a full REST API restart, got {unexpected:?}"
        ),
    }
}

#[test]
fn should_classify_no_changes_as_no_worker_restart() {
    let current = sample_resolved_config();
    let candidate = sample_resolved_config();

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::NoWorkerRestart { .. }),
        "identical configs must not trigger a worker restart, got {decision:?}"
    );
}

#[test]
fn should_classify_config_warning_only_change_as_no_worker_restart() {
    let mut current = sample_resolved_config();
    current.config_warning = Some("old warning".to_owned());
    let mut candidate = sample_resolved_config();
    candidate.config_warning = Some("new warning".to_owned());

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::NoWorkerRestart { ref reloaded_fields, .. } if reloaded_fields.contains(&"config_warning".to_owned())),
        "a config-warning text change must be a no-worker-restart reload, got {decision:?}"
    );
}

#[test]
fn should_classify_max_output_tokens_change_as_worker_restart() {
    let mut current = sample_resolved_config();
    current.max_output_tokens = 20_480;
    let mut candidate = sample_resolved_config();
    candidate.max_output_tokens = 8_192;

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. } if reloaded_fields.contains(&"max_output_tokens".to_owned())),
        "a max_output_tokens change must trigger a worker restart, got {decision:?}"
    );
}

#[test]
fn should_classify_mtp_config_as_worker_restart() {
    let mut current = sample_resolved_config();
    current.mtp_enabled = false;
    let mut candidate = sample_resolved_config();
    candidate.mtp_enabled = true;

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(
            decision,
            ConfigReloadDecision::RestartWorker {
                ref reloaded_fields,
                ..
            } if reloaded_fields == &["mtp_enabled".to_owned()]
        ),
        "an mtp_enabled change must trigger a worker restart, got {decision:?}"
    );
}

#[test]
fn should_classify_performance_attribution_change_as_worker_restart() {
    let mut current = sample_resolved_config();
    current.performance_attribution_enabled = false;
    let mut candidate = sample_resolved_config();
    candidate.performance_attribution_enabled = true;

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. } if reloaded_fields == &["performance_attribution_enabled".to_owned()]),
        "a performance attribution change must restart the worker, got {decision:?}"
    );
}

#[test]
fn should_classify_model_directories_change_as_worker_restart() {
    let mut current = sample_resolved_config();
    current.model_directories = Arc::new(HashMap::from([(
        "default".to_owned(),
        PathBuf::from("/tmp/models-a"),
    )]));
    let mut candidate = sample_resolved_config();
    candidate.model_directories = Arc::new(HashMap::from([(
        "default".to_owned(),
        PathBuf::from("/tmp/models-b"),
    )]));

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. } if reloaded_fields.contains(&"model_directories".to_owned())),
        "a model_directories change must trigger a worker restart, got {decision:?}"
    );
}

#[test]
fn should_restart_worker_when_empty_configured_model_root_changes() {
    let mut current = sample_resolved_config();
    current.configured_model_directories = vec![PathBuf::from("/tmp/empty-model-root-a")];
    let mut candidate = sample_resolved_config();
    candidate.configured_model_directories = vec![PathBuf::from("/tmp/empty-model-root-b")];

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. } if reloaded_fields.contains(&"model_directories".to_owned())),
        "configured model roots must participate in the worker-restart diff, got {decision:?}"
    );
}

#[test]
fn should_classify_prompt_cache_capacity_change_as_worker_restart() {
    let mut current = sample_resolved_config();
    current.prompt_cache_config =
        PromptCacheConfig::new(PathBuf::from("/tmp/prompt-cache"), 1_000_000_000);
    let mut candidate = sample_resolved_config();
    candidate.prompt_cache_config =
        PromptCacheConfig::new(PathBuf::from("/tmp/prompt-cache"), 2_000_000_000);

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. } if reloaded_fields.contains(&"prompt_cache".to_owned())),
        "a prompt-cache capacity change must restart the worker, got {decision:?}"
    );
}

#[test]
fn should_mark_logging_level_change_as_rest_api_restart_required() {
    let mut current = sample_resolved_config();
    current.logging_config = LoggingConfig::new(PathBuf::from("/tmp/logs"), LogLevel::Warn, 7);
    let mut candidate = sample_resolved_config();
    candidate.logging_config = LoggingConfig::new(PathBuf::from("/tmp/logs"), LogLevel::Info, 7);

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestApiRestartRequired { ref restart_required_fields, .. } if restart_required_fields.contains(&"logging".to_owned())),
        "a logging level change must require a REST restart, got {decision:?}"
    );
}

fn sample_resolved_config() -> ResolvedRuntimeConfig {
    ResolvedRuntimeConfig {
        worker_executable_path: PathBuf::from("/tmp/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        configured_model_directories: Vec::new(),
        model_directories: Arc::new(HashMap::new()),
        max_output_tokens: 20_480,
        maximum_mlx_memory_bytes: None,
        config_warning: None,
        prefill_chunck_sizing_policy: PrefillChunckSizingPolicy::Optimized,
        optimizer_state_directory: PathBuf::from("/tmp/astronomical-optimizer"),
        performance_attribution_enabled: false,
        mtp_enabled: false,
        prompt_cache_config: PromptCacheConfig::new(
            PathBuf::from("/tmp/prompt-cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6732".to_owned(),
        logging_config: LoggingConfig::new(
            PathBuf::from("/tmp/astronomical-logs"),
            LogLevel::Warn,
            7,
        ),
    }
}

fn resolved_config_with_bind_address(bind_address: &str) -> ResolvedRuntimeConfig {
    let mut config = sample_resolved_config();
    config.bind_address = bind_address.to_owned();
    config
}

fn resolved_config_with_logging_directory(directory: &str) -> ResolvedRuntimeConfig {
    let mut config = sample_resolved_config();
    config.logging_config = LoggingConfig::new(PathBuf::from(directory), LogLevel::Warn, 7);
    config
}

#[tokio::test]
async fn should_replace_the_worker_process_for_worker_startup_config_changes() {
    let initial_worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
            .expect("Cargo should provide the initial worker fixture path");
    let replacement_worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-replacement-ready-worker")
            .expect("Cargo should provide the replacement worker fixture path");
    let performance_log_directory =
        tempfile::tempdir().expect("the performance-log directory should be created");
    let performance_log = GenerationPerformanceLog::open(performance_log_directory.path())
        .expect("the performance log should open");
    let worker_handle = WorkerHandle::launch(
        initial_worker_executable_path,
        Duration::from_secs(2),
        performance_log,
        Arc::new(HashMap::new()),
        20_480,
    )
    .await
    .expect("the initial worker should launch");
    wait_for_ready_model(&worker_handle, "astronomical/test-worker").await;

    worker_handle
        .restart_worker(
            PathBuf::from(replacement_worker_executable_path),
            Arc::new(HashMap::new()),
            20_480,
        )
        .await
        .expect("the replacement worker should launch");

    wait_for_ready_model(&worker_handle, "astronomical/replacement-worker").await;
    worker_handle
        .shutdown()
        .await
        .expect("the replacement worker should shut down");
}

async fn wait_for_ready_model(worker_handle: &WorkerHandle, expected_model_id: &str) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready
            && worker_health_snapshot.ready_model_id.as_deref() == Some(expected_model_id)
        {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "worker did not become ready with model {expected_model_id}; latest snapshot: {worker_health_snapshot:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

#[test]
fn should_resolve_reload_config_from_the_config_file() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    let config_file_path = config_home_directory
        .join(".astronomical")
        .join("config.json");
    std::fs::create_dir_all(
        config_file_path
            .parent()
            .expect("the config path should have a parent"),
    )
    .expect("the config directory should be created");
    std::fs::write(
        &config_file_path,
        r#"{
            "prefill_chunck_size_optimizer_enabled": true,
            "supervisor": {
                "bind_address": "127.0.0.1:6733"
            },
            "prompt_cache_max_size_gb": 1
        }"#,
    )
    .expect("the config file should be written");
    let resolver = ResolvedRuntimeConfigResolver::new(
        config_home_directory.clone(),
        PathBuf::from("/fallback/worker"),
    );

    let resolved_config = resolver.load().expect("the reload config should resolve");

    assert_eq!(resolved_config.bind_address, "127.0.0.1:6733");
    assert_eq!(
        resolved_config.worker_executable_path,
        PathBuf::from("/fallback/worker")
    );
    assert_eq!(
        resolved_config
            .prompt_cache_config
            .global_prompt_cache_root_directory(),
        &config_home_directory.join(".astronomical/cache")
    );
}
