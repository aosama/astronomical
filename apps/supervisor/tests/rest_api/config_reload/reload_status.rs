use super::*;

#[tokio::test]
async fn should_apply_prompt_cache_policy_after_config_file_reload() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{
            "prompt_cache": { "enabled": true }
        }"#,
    );
    let performance_log_directory = config_home_directory.join("performance");
    std::fs::create_dir_all(&performance_log_directory)
        .expect("the performance log directory should be created");
    let worker_handle = WorkerHandle::launch(
        &worker_executable_path,
        Duration::from_secs(2),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the performance log should open"),
        Arc::new(std::collections::HashMap::new()),
    )
    .await
    .expect("the idle worker should launch");
    wait_for_ready_idle_worker(&worker_handle).await;
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.clone(),
        worker_executable_path,
    );
    let mut initial_resolved_config = runtime_config_resolver
        .load()
        .expect("the initial configuration should resolve");
    initial_resolved_config.persistent_prompt_cache_enabled = false;
    initial_resolved_config.configured_persistent_prompt_cache_enabled = Some(false);
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
    let application = build_application_with_full_control(
        worker_handle.clone(),
        Arc::clone(&reloadable_config),
        runtime_config_resolver,
        ShutdownController::new(),
    );

    let reload_response = post_config_reload(&application).await;
    let reload_status = reload_response.status();
    let reload_body = to_bytes(reload_response.into_body(), 4 * 1024)
        .await
        .expect("the reload response should be readable");
    let reload_document: serde_json::Value =
        serde_json::from_slice(&reload_body).expect("the reload response should contain JSON");
    assert_eq!(reload_status, StatusCode::OK, "{reload_document}");
    assert_eq!(reload_document["status"], "reloaded");
    assert_eq!(reload_document["worker_restart_completed"], true);
    assert_eq!(
        reload_document["candidate_generation"],
        reload_document["effective_generation"]
    );
    assert_eq!(
        reload_document["worker_runtime_feature_configuration"]["configuration_generation"],
        reload_document["candidate_generation"]
    );
    assert_eq!(
        reload_document["worker_runtime_feature_configuration"]["persistent_prompt_cache_enabled"],
        true
    );

    let status_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return status");
    let status_body = to_bytes(status_response.into_body(), 4 * 1024)
        .await
        .expect("the status response should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_body).expect("the status response should contain JSON");
    assert_eq!(
        status_document["configured_speculative_prefill_enabled"],
        false
    );
    assert_eq!(status_document["speculative_prefill_enabled"], false);
    assert_eq!(
        status_document["configured_generation"],
        status_document["effective_generation"]
    );
    assert_eq!(
        status_document["speculative_prefill_draft_model_id"],
        serde_json::Value::Null
    );
    assert_eq!(
        status_document["worker_runtime_feature_configuration_applied"],
        true
    );
    assert_eq!(
        status_document["worker_runtime_feature_configuration"]["persistent_prompt_cache_enabled"],
        true
    );
    assert!(
        reloadable_config
            .read()
            .expect("the applied config should remain readable")
            .persistent_prompt_cache_enabled
    );

    worker_handle
        .shutdown()
        .await
        .expect("the worker should shut down");
}

async fn wait_for_ready_idle_worker(worker_handle: &WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if worker_handle.worker_health_snapshot().status == WorkerHealthStatus::Ready {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "the fixture worker should become ready before the reload journey"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn should_keep_reporting_restart_required_until_server_is_restarted() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{
            "diagnostics": { "log_level": "info" }
        }"#,
    );
    let reloadable_config = Arc::new(RwLock::new(ResolvedRuntimeConfig {
        configuration_generation:
            "1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
        worker_executable_path: PathBuf::from("/tmp/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        model_discovery_diagnostics: Vec::new(),
        configured_model_directories: Vec::new(),
        model_policy_catalog: Arc::new(std::collections::HashMap::new()),
        unmatched_model_config_ids: Vec::new(),
        maximum_mlx_memory_bytes: None,
        persistent_prompt_cache_enabled: true,
        configured_persistent_prompt_cache_enabled: None,
        configured_prompt_cache_maximum_size_bytes: None,
        performance_attribution_enabled: false,
        experimental_qwen_thinking_channel_seed_enabled: false,
        prompt_cache_config: astronomical_config::PromptCacheConfig::new(
            config_home_directory
                .join(".astronomical-dev")
                .join("cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6733".to_owned(),
        logging_config: astronomical_config::LoggingConfig::new(
            config_home_directory.join(".astronomical-dev").join("logs"),
            astronomical_config::LogLevel::Warn,
            7,
        ),
    }));
    let application = build_development_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        reloadable_config,
        config_home_directory,
    );

    for reload_attempt_number in 1..=2 {
        let reload_response = post_config_reload(&application).await;
        let reload_body = to_bytes(reload_response.into_body(), 4 * 1024)
            .await
            .expect("the reload response should be readable");
        let reload_document: serde_json::Value =
            serde_json::from_slice(&reload_body).expect("the reload response should contain JSON");
        assert_eq!(
            reload_document["status"], "restart_required",
            "reload attempt {reload_attempt_number} must remain restart-required until the listener is actually restarted"
        );
    }
}

#[tokio::test]
async fn should_not_report_a_mixed_reload_effective_when_only_memory_applied() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{
            "runtime": {
                "model_directories": [],
                "maximum_mlx_memory_gb": 32
            },
            "diagnostics": { "log_level": "info" }
        }"#,
    );
    let performance_log_directory = config_home_directory.join("performance");
    std::fs::create_dir_all(&performance_log_directory)
        .expect("the performance log directory should be created");
    let mut initial_resolved_config = sample_resolved_config();
    initial_resolved_config.worker_executable_path = worker_executable_path.clone();
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        &worker_executable_path,
        Duration::from_secs(2),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the performance log should open"),
        Arc::clone(&initial_resolved_config.model_policy_catalog),
        initial_resolved_config.worker_startup_configuration(),
    )
    .await
    .expect("the idle worker should launch");
    wait_for_worker_configuration(&worker_handle).await;
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.clone(),
        worker_executable_path,
    );
    let application = build_application_with_full_control(
        worker_handle.clone(),
        Arc::new(RwLock::new(initial_resolved_config)),
        runtime_config_resolver,
        ShutdownController::new(),
    );

    let reload_response = post_config_reload(&application).await;
    assert_eq!(reload_response.status(), StatusCode::OK);
    let status_response = application
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("status request should be valid"),
        )
        .await
        .expect("status should be returned");
    let status_body = to_bytes(status_response.into_body(), 8 * 1024)
        .await
        .expect("status should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_body).expect("status should contain JSON");

    assert_eq!(status_document["configuration"]["is_effective"], false);
    assert_eq!(status_document["configuration"]["restart_required"], true);
    assert_ne!(
        status_document["configuration"]["resolved_generation"],
        status_document["configuration"]["configured_generation"]
    );
    assert_eq!(
        status_document["configuration"]["effective_generation"],
        status_document["configuration"]["resolved_generation"]
    );
    assert_ne!(
        status_document["configuration"]["effective_generation"],
        status_document["configuration"]["configured_generation"]
    );

    worker_handle
        .shutdown()
        .await
        .expect("the worker should shut down");
}

async fn wait_for_worker_configuration(worker_handle: &WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let health_snapshot = worker_handle.worker_health_snapshot();
        if health_snapshot.status == WorkerHealthStatus::Ready
            && health_snapshot
                .worker_runtime_feature_configuration
                .is_some()
        {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "the fixture worker should acknowledge startup configuration"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn should_keep_worker_and_logging_fields_unchanged_when_application_restart_is_required() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{
            "chunking": { "fixed_prompt_processing_chunk_size_tokens": 4096 },
            "diagnostics": { "log_level": "info" }
        }"#,
    );
    let mut initial_resolved_config = sample_resolved_config();
    initial_resolved_config.logging_config = astronomical_config::LoggingConfig::new(
        config_home_directory.join(".astronomical-dev").join("logs"),
        astronomical_config::LogLevel::Warn,
        7,
    );
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
    let application = build_development_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        Arc::clone(&reloadable_config),
        config_home_directory,
    );

    let reload_response = post_config_reload(&application).await;
    assert_eq!(reload_response.status(), StatusCode::OK);
    let reload_body = to_bytes(reload_response.into_body(), 4 * 1024)
        .await
        .expect("the reload response should be readable");
    let reload_document: serde_json::Value =
        serde_json::from_slice(&reload_body).expect("the reload response should contain JSON");
    assert_eq!(reload_document["status"], "restart_required");
    assert_eq!(reload_document["reloaded_fields"], serde_json::json!([]));
    assert_eq!(
        reload_document["restart_required_fields"],
        serde_json::json!(["logging"])
    );

    let live_config = reloadable_config
        .read()
        .expect("the reloadable config should remain readable");
    assert_eq!(live_config.bind_address, "127.0.0.1:6733");
    assert_eq!(
        live_config.logging_config.level(),
        astronomical_config::LogLevel::Warn
    );
}

#[tokio::test]
async fn should_keep_all_discovered_models_listed_and_routable() {
    const CONFIGURED_TARGET_MODEL_ID: &str = "astronomical/application-test-model";
    const UNCONFIGURED_MODEL_ID: &str = "astronomical/another-test-model";

    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(&config_home_directory, r#"{}"#);
    let mut resolved_config = sample_resolved_config();
    resolved_config.discovered_models = vec![
        discovered_model_for(CONFIGURED_TARGET_MODEL_ID),
        discovered_model_for(UNCONFIGURED_MODEL_ID),
    ];
    let reloadable_config = Arc::new(RwLock::new(resolved_config));
    let scripted_executor = ScriptedExecutor::ready(Vec::new());
    let received_generation_commands = scripted_executor.received_generation_commands();
    let application = build_development_application_with_reload(
        scripted_executor,
        reloadable_config,
        config_home_directory,
    );

    let model_list_response = application
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("the model-list request should be well formed"),
        )
        .await
        .expect("the model-list request should receive a response");
    assert_eq!(model_list_response.status(), StatusCode::OK);
    let model_list_body = to_bytes(model_list_response.into_body(), 16 * 1024)
        .await
        .expect("the model-list response should be readable");
    let model_list_text = String::from_utf8(model_list_body.to_vec())
        .expect("the model-list response should be UTF-8");
    assert!(model_list_text.contains(CONFIGURED_TARGET_MODEL_ID));
    assert!(model_list_text.contains(UNCONFIGURED_MODEL_ID));

    let configured_target_response = application
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(format!(
                    r#"{{"model":"{CONFIGURED_TARGET_MODEL_ID}","messages":[{{"role":"user","content":"hello"}}],"stream":true}}"#
                )))
                .expect("the configured-target request should be well formed"),
        )
        .await
        .expect("the configured-target request should receive a response");
    assert_eq!(configured_target_response.status(), StatusCode::OK);

    let ordinary_model_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(format!(
                    r#"{{"model":"{UNCONFIGURED_MODEL_ID}","messages":[{{"role":"user","content":"hello"}}],"stream":true}}"#
                )))
                .expect("the ordinary-model request should be well formed"),
        )
        .await
        .expect("the ordinary-model request should receive a response");
    assert_eq!(ordinary_model_response.status(), StatusCode::OK);

    let received_generation_commands = received_generation_commands
        .lock()
        .expect("the scripted command log should not be poisoned");
    assert_eq!(received_generation_commands.len(), 2);
    assert_eq!(
        received_generation_commands[0].model,
        CONFIGURED_TARGET_MODEL_ID
    );
    assert_eq!(received_generation_commands[1].model, UNCONFIGURED_MODEL_ID);
}

#[tokio::test]
async fn should_use_the_same_reloaded_discovery_snapshot_for_listing_and_routing() {
    const RELOADED_MODEL_ID: &str = "astronomical/reloaded-laguna";
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(&config_home_directory, r#"{}"#);
    let reloadable_config = Arc::new(RwLock::new(sample_resolved_config()));
    let scripted_executor = ScriptedExecutor::ready(Vec::new());
    let received_generation_commands = scripted_executor.received_generation_commands();
    let application = build_development_application_with_reload(
        scripted_executor,
        Arc::clone(&reloadable_config),
        config_home_directory,
    );
    let mut reloaded_laguna = discovered_model_for(RELOADED_MODEL_ID);
    reloaded_laguna.model_family = astronomical_config::ModelFamily::Laguna;
    reloadable_config
        .write()
        .expect("the reloadable config should remain writable")
        .discovered_models = vec![reloaded_laguna];

    let model_list_response = application
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("the model-list request should be well formed"),
        )
        .await
        .expect("the model-list request should receive a response");
    let model_list_body = to_bytes(model_list_response.into_body(), 16 * 1024)
        .await
        .expect("the model-list response should be readable");
    assert!(String::from_utf8_lossy(&model_list_body).contains(RELOADED_MODEL_ID));

    let generation_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(format!(
                    r#"{{"model":"{RELOADED_MODEL_ID}","messages":[{{"role":"user","content":"hello"}}],"stream":true}}"#
                )))
                .expect("the Laguna request should be well formed"),
        )
        .await
        .expect("the Laguna request should receive a response");

    assert_eq!(generation_response.status(), StatusCode::OK);
    let received_generation_commands = received_generation_commands
        .lock()
        .expect("the scripted command log should remain readable");
    assert_eq!(received_generation_commands.len(), 1);
    assert_eq!(received_generation_commands[0].model, RELOADED_MODEL_ID);
}

fn discovered_model_for(model_id: &str) -> astronomical_config::DiscoveredModel {
    astronomical_config::DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "test-revision".to_owned(),
        model_directory: PathBuf::from(format!("/fictional/models/{model_id}")),
        capabilities: astronomical_config::ModelCapabilities::Chat(
            astronomical_config::ChatModelCapabilities {
                context_window: 2_048,
                max_input_tokens: 1_024,
                max_output_tokens: 128,
                supports_vision: false,
                supports_reasoning: true,
                supports_tool_calls: true,
            },
        ),
        license: None,
        model_size_bytes: 0,
    }
}

#[tokio::test]
async fn should_reject_prompt_cache_reload_when_worker_replacement_is_unavailable() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{
            "prompt_cache": { "enabled": false }
        }"#,
    );
    let mut initial_resolved_config = sample_resolved_config();
    initial_resolved_config.logging_config = astronomical_config::LoggingConfig::new(
        config_home_directory.join(".astronomical-dev").join("logs"),
        astronomical_config::LogLevel::Warn,
        7,
    );
    let application = build_development_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        Arc::new(RwLock::new(initial_resolved_config)),
        config_home_directory,
    );

    let reload_response = post_config_reload(&application).await;
    assert_eq!(reload_response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let reload_body = to_bytes(reload_response.into_body(), 4 * 1024)
        .await
        .expect("the reload response should be readable");
    let reload_document: serde_json::Value =
        serde_json::from_slice(&reload_body).expect("the reload response should contain JSON");
    assert_eq!(reload_document["worker_restart_completed"], false);
    assert_eq!(reload_document["reloaded_fields"], serde_json::json!([]));
    assert_eq!(reload_document["status"], "failed");

    let status_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return status");
    let status_body = to_bytes(status_response.into_body(), 16 * 1024)
        .await
        .expect("the status response should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_body).expect("the status response should contain JSON");

    assert_eq!(status_document["mtp_enabled"], false);
}
