//! End-to-end reload coverage for transactional worker replacement failures.

use super::*;

#[tokio::test]
async fn should_reject_mismatched_candidate_without_changing_effective_serving_state() {
    let initial_worker_executable_path =
        fixture_path("astronomical-supervisor-idle-worker", "idle worker");
    let replacement_worker_executable_path = fixture_path(
        "astronomical-supervisor-replacement-ready-worker",
        "mismatched replacement worker",
    );
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{"prompt_cache":{"maximum_size_gb":49}}"#,
    );
    let performance_log_directory = config_home_directory.join("performance");
    std::fs::create_dir_all(&performance_log_directory)
        .expect("performance log directory should be created");
    let mut initial_config = sample_resolved_config();
    initial_config.logging_config = astronomical_config::LoggingConfig::new(
        config_home_directory.join(".astronomical-dev/logs"),
        astronomical_config::LogLevel::Warn,
        7,
    );
    let initial_generation = initial_config.configuration_generation.clone();
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        initial_worker_executable_path,
        Duration::from_secs(2),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("performance log should open"),
        Arc::new(std::collections::HashMap::new()),
        initial_config.worker_startup_configuration(),
    )
    .await
    .expect("idle worker should launch");
    wait_for_effective_generation(&worker_handle, &initial_generation).await;
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.clone(),
        replacement_worker_executable_path,
    );
    let reloadable_config = Arc::new(RwLock::new(initial_config));
    let application = build_application_with_full_control(
        worker_handle.clone(),
        Arc::clone(&reloadable_config),
        runtime_config_resolver,
        ShutdownController::new(),
    );

    let reload_response = post_config_reload(&application).await;
    let reload_status = reload_response.status();
    let reload_body = to_bytes(reload_response.into_body(), 8 * 1024)
        .await
        .expect("reload response should be readable");
    let reload_document: serde_json::Value =
        serde_json::from_slice(&reload_body).expect("reload response should contain JSON");

    assert_eq!(reload_status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_ne!(
        reload_document["candidate_generation"],
        reload_document["effective_generation"]
    );
    assert_eq!(reload_document["effective_generation"], initial_generation);
    let retained_health = worker_handle.worker_health_snapshot();
    assert_eq!(retained_health.status, WorkerHealthStatus::Ready);
    assert_eq!(
        retained_health
            .worker_runtime_feature_configuration
            .as_ref()
            .map(|configuration| configuration.configuration_generation.as_str()),
        Some(initial_generation.as_str())
    );
    assert_eq!(
        reloadable_config
            .read()
            .expect("live config should remain readable")
            .configuration_generation,
        initial_generation
    );
    worker_handle
        .shutdown()
        .await
        .expect("worker should shut down");
}

async fn wait_for_effective_generation(worker_handle: &WorkerHandle, expected_generation: &str) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let health_snapshot = worker_handle.worker_health_snapshot();
        if health_snapshot.status == WorkerHealthStatus::Ready
            && health_snapshot
                .worker_runtime_feature_configuration
                .as_ref()
                .is_some_and(|configuration| {
                    configuration.configuration_generation == expected_generation
                })
        {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "worker acknowledgement timed out"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

fn fixture_path(fixture_name: &str, fixture_description: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(format!("CARGO_BIN_EXE_{fixture_name}"))
            .unwrap_or_else(|_| panic!("Cargo should provide the {fixture_description} path")),
    )
}
