use super::*;

#[tokio::test]
async fn should_not_let_a_rejected_update_restore_over_a_newer_memory_setting() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(&config_home_directory, r#"{}"#);
    let performance_log_directory = config_home_directory.join("performance");
    std::fs::create_dir_all(&performance_log_directory)
        .expect("the performance-log directory should be created");
    let worker_handle = WorkerHandle::launch(
        &worker_executable_path,
        Duration::from_secs(2),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the performance log should open"),
        Arc::new(std::collections::HashMap::new()),
        20_480,
    )
    .await
    .expect("the idle worker should launch");
    wait_for_idle_worker(&worker_handle).await;
    let mut initial_resolved_config = sample_resolved_config();
    initial_resolved_config.worker_executable_path = worker_executable_path.clone();
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.clone(),
        worker_executable_path,
    );
    let application = build_application_with_full_control(
        worker_handle.clone(),
        reloadable_config,
        runtime_config_resolver,
        ShutdownController::new(),
    );

    let rejected_application = application.clone();
    let rejected_update_task =
        tokio::spawn(async move { put_maximum_mlx_memory(&rejected_application, 31).await });
    wait_for_persisted_maximum(&config_home_directory, 31).await;
    let accepted_response = put_maximum_mlx_memory(&application, 32).await;
    let rejected_response = rejected_update_task
        .await
        .expect("the rejected update task should finish");

    assert_eq!(rejected_response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(accepted_response.status(), StatusCode::OK);
    wait_for_persisted_maximum(&config_home_directory, 32).await;
    worker_handle
        .shutdown()
        .await
        .expect("the worker should shut down");
}

async fn put_maximum_mlx_memory(
    application: &axum::Router,
    maximum_mlx_memory_gb: u64,
) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/config/maximum-mlx-memory")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(format!(
                    "{{\"maximum_mlx_memory_gb\":{maximum_mlx_memory_gb}}}"
                )))
                .expect("the memory-limit request should be valid"),
        )
        .await
        .expect("the application should return a memory-limit response")
}

async fn wait_for_persisted_maximum(home_directory: &std::path::Path, expected_gigabytes: u64) {
    let config_file_path = home_directory.join(".astronomical-dev").join("config.json");
    let persistence_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let persisted_gigabytes = std::fs::read(&config_file_path)
            .ok()
            .and_then(|config_bytes| {
                serde_json::from_slice::<serde_json::Value>(&config_bytes).ok()
            })
            .and_then(|config_document| config_document["maximum_mlx_memory_gb"].as_u64());
        if persisted_gigabytes == Some(expected_gigabytes) {
            return;
        }
        assert!(
            Instant::now() < persistence_deadline,
            "maximum_mlx_memory_gb did not become {expected_gigabytes}; observed {persisted_gigabytes:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_idle_worker(worker_handle: &WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready
            && worker_health_snapshot.machine_mlx_memory_ceiling_bytes == 40_000_000_000
        {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "idle worker did not report its memory limits: {worker_health_snapshot:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
}
