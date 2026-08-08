use super::*;

#[tokio::test]
async fn should_keep_reporting_restart_required_until_server_is_restarted() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{
            "supervisor": { "bind_address": "127.0.0.1:6733" },
            "prompt_cache_max_size_gb": 50
        }"#,
    );
    let reloadable_config = Arc::new(RwLock::new(ResolvedRuntimeConfig {
        worker_executable_path: PathBuf::from("/tmp/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        configured_model_directories: Vec::new(),
        model_directories: Arc::new(std::collections::HashMap::new()),
        max_output_tokens: 20_480,
        maximum_mlx_memory_bytes: None,
        config_warning: None,
        prefill_chunck_sizing_policy: astronomical_config::PrefillChunckSizingPolicy::Fixed {
            fixed_prefill_chunck_tokens: 2_048,
        },
        optimizer_state_directory: config_home_directory
            .join(".astronomical")
            .join("optimizer"),
        persistent_prompt_cache_enabled: true,
        performance_attribution_enabled: false,
        mtp_enabled: false,
        speculative_prefill: astronomical_config::SpeculativePrefillConfig::disabled(),
        speculative_prefill_draft_model_directory: None,
        prompt_cache_config: astronomical_config::PromptCacheConfig::new(
            config_home_directory.join(".astronomical").join("cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6732".to_owned(),
        logging_config: astronomical_config::LoggingConfig::new(
            config_home_directory.join(".astronomical").join("logs"),
            astronomical_config::LogLevel::Warn,
            7,
        ),
    }));
    let application = build_application_with_reload(
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
async fn should_apply_only_in_place_reload_fields_when_a_rest_api_restart_is_required() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{
            "prefill_chunck_size_optimizer_enabled": true,
            "supervisor": { "bind_address": "127.0.0.1:6733" }
        }"#,
    );
    let mut initial_resolved_config = sample_resolved_config();
    initial_resolved_config.config_warning = Some("old startup warning".to_owned());
    initial_resolved_config.logging_config = astronomical_config::LoggingConfig::new(
        config_home_directory.join(".astronomical").join("logs"),
        astronomical_config::LogLevel::Warn,
        7,
    );
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
    let application = build_application_with_reload(
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
    assert_eq!(
        reload_document["reloaded_fields"],
        serde_json::json!(["config_warning"])
    );
    assert_eq!(
        reload_document["restart_required_fields"],
        serde_json::json!(["supervisor.bind_address"])
    );

    let live_config = reloadable_config
        .read()
        .expect("the reloadable config should remain readable");
    assert_eq!(live_config.config_warning, None);
    assert_eq!(live_config.bind_address, "127.0.0.1:6732");
}

#[tokio::test]
async fn should_update_status_config_warning_after_successful_reload() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{
            "prefill_chunck_size_optimizer_enabled": true
        }"#,
    );
    let reloadable_config = Arc::new(RwLock::new(ResolvedRuntimeConfig {
        worker_executable_path: PathBuf::from("/tmp/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        configured_model_directories: Vec::new(),
        model_directories: Arc::new(std::collections::HashMap::new()),
        max_output_tokens: 20_480,
        maximum_mlx_memory_bytes: None,
        config_warning: Some("ignored fixed prefill setting".to_owned()),
        prefill_chunck_sizing_policy: astronomical_config::PrefillChunckSizingPolicy::Optimized {
            optimizer_prefill_chunck_token_candidates: vec![1_024, 2_048, 4_096, 8_192],
        },
        optimizer_state_directory: config_home_directory
            .join(".astronomical")
            .join("optimizer"),
        persistent_prompt_cache_enabled: true,
        performance_attribution_enabled: false,
        mtp_enabled: true,
        speculative_prefill: astronomical_config::SpeculativePrefillConfig::disabled(),
        speculative_prefill_draft_model_directory: None,
        prompt_cache_config: astronomical_config::PromptCacheConfig::new(
            config_home_directory.join(".astronomical").join("cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6732".to_owned(),
        logging_config: astronomical_config::LoggingConfig::new(
            config_home_directory.join(".astronomical").join("logs"),
            astronomical_config::LogLevel::Warn,
            7,
        ),
    }));
    let application = build_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        reloadable_config,
        config_home_directory,
    );

    let reload_response = post_config_reload(&application).await;
    assert_eq!(reload_response.status(), StatusCode::OK);
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
    assert_eq!(status_document["config_warning"], serde_json::Value::Null);
}

#[tokio::test]
async fn should_keep_all_discovered_models_listed_and_routable_with_speculative_prefill() {
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
    resolved_config.speculative_prefill = astronomical_config::SpeculativePrefillConfig::new(
        true,
        Some(CONFIGURED_TARGET_MODEL_ID.to_owned()),
        Some("astronomical/draft-test-model".to_owned()),
        8_192,
        20,
        32,
        512,
        8,
        13,
    );
    let reloadable_config = Arc::new(RwLock::new(resolved_config));
    let scripted_executor = ScriptedExecutor::ready(Vec::new());
    let received_generation_commands = scripted_executor.received_generation_commands();
    let application =
        build_application_with_reload(scripted_executor, reloadable_config, config_home_directory);

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

fn discovered_model_for(model_id: &str) -> astronomical_config::DiscoveredModel {
    astronomical_config::DiscoveredModel {
        model_id: model_id.to_owned(),
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "test-revision".to_owned(),
        model_directory: PathBuf::from(format!("/fictional/models/{model_id}")),
        context_window: 2_048,
        max_input_tokens: 1_024,
        max_output_tokens: 128,
        has_vision: false,
        model_size_bytes: 0,
    }
}

#[tokio::test]
async fn should_reject_mtp_reload_when_worker_replacement_is_unavailable() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(
        &config_home_directory,
        r#"{
            "mtp_enabled": true
        }"#,
    );
    let mut initial_resolved_config = sample_resolved_config();
    initial_resolved_config.logging_config = astronomical_config::LoggingConfig::new(
        config_home_directory.join(".astronomical").join("logs"),
        astronomical_config::LogLevel::Warn,
        7,
    );
    let application = build_application_with_reload(
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
    assert_eq!(reload_document["worker_restart_started"], false);
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
    let status_body = to_bytes(status_response.into_body(), 4 * 1024)
        .await
        .expect("the status response should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_body).expect("the status response should contain JSON");

    assert_eq!(status_document["mtp_enabled"], false);
}
