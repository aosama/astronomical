use super::*;

#[tokio::test]
async fn should_reject_invalid_config_reload_without_mutating_live_state() {
    let temp_config_directory = tempfile::tempdir().expect("a temp config directory is needed");
    let config_home_directory = temp_config_directory.path().to_path_buf();
    write_raw_config_file(&config_home_directory, "{ this is not valid json }");

    let initial_resolved_config = sample_resolved_config();
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config.clone()));
    let application = build_development_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        reloadable_config.clone(),
        config_home_directory,
    );

    let response = post_config_reload(&application).await;
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "an invalid config must produce HTTP 400"
    );
    let response_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the reload error body should be readable");
    let response_json: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the reload error body should be JSON");
    assert_eq!(response_json["status"], "invalid_config");

    let status_response = application
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("status should remain available");
    let status_body = to_bytes(status_response.into_body(), 16 * 1024)
        .await
        .expect("the status response should be readable");
    let status_json: serde_json::Value =
        serde_json::from_slice(&status_body).expect("the status response should contain JSON");
    assert_eq!(status_json["configuration"]["is_effective"], false);
    assert_eq!(status_json["configuration"]["restart_required"], false);
    assert_eq!(
        status_json["configuration"]["validation_error"],
        "Configuration is invalid; correct the local configuration file and retry"
    );

    // Live state must be unchanged after an invalid reload.
    let live_config = reloadable_config
        .read()
        .expect("the reloadable config lock should be readable");
    assert_eq!(*live_config, initial_resolved_config);
}

#[tokio::test]
async fn should_preserve_live_serving_and_expose_path_safe_duplicate_model_feedback() {
    let temporary_home = tempfile::tempdir().expect("a temporary home should be created");
    let first_root = temporary_home.path().join("first-root");
    let second_root = temporary_home.path().join("second-root");
    write_minimal_qwen_model(&first_root.join("ambiguous-model"));
    write_minimal_qwen_model(&second_root.join("ambiguous-model"));
    write_config_file(
        temporary_home.path(),
        &serde_json::json!({
            "runtime": { "model_directories": [first_root, second_root] }
        })
        .to_string(),
    );
    let initial_resolved_config = sample_resolved_config();
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config.clone()));
    let application = build_development_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        Arc::clone(&reloadable_config),
        temporary_home.path().to_path_buf(),
    );

    let response = post_config_reload(&application).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the reload response should be readable");
    let response_text = String::from_utf8_lossy(&response_body);
    assert!(response_text.contains("ambiguous-model"));
    assert!(response_text.contains("entries 1, 2"));
    assert!(!response_text.contains(temporary_home.path().to_string_lossy().as_ref()));
    assert_eq!(
        *reloadable_config
            .read()
            .expect("the live configuration should remain readable"),
        initial_resolved_config
    );

    let status_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("status should remain available");
    let status_body = to_bytes(status_response.into_body(), 16 * 1024)
        .await
        .expect("status should remain readable");
    let status_json: serde_json::Value =
        serde_json::from_slice(&status_body).expect("status should contain JSON");
    assert_eq!(
        status_json["configuration"]["model_discovery_diagnostics"][0]["model_id"],
        "ambiguous-model"
    );
}

#[tokio::test]
async fn should_reject_retired_top_level_speculative_prefill_configuration() {
    let temporary_config_directory =
        tempfile::tempdir().expect("a temporary config directory is needed");
    let config_home_directory = temporary_config_directory.path().to_path_buf();
    write_config_file(
        &config_home_directory,
        r#"{
          "speculative_prefill": {
            "enabled": true,
            "target_model_id": "Qwen3.5-35B-Target",
            "draft_model_id": "Qwen3.5-2B-Draft"
          }
        }"#,
    );

    let initial_resolved_config = sample_resolved_config();
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config.clone()));
    let application = build_development_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        Arc::clone(&reloadable_config),
        config_home_directory,
    );

    let response = post_config_reload(&application).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the reload error body should be readable");
    let response_json: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the reload error body should be JSON");
    assert_eq!(response_json["status"], "invalid_config");
    assert_eq!(
        response_json["message"],
        "Configuration is invalid; correct the local configuration file and retry"
    );
    assert_eq!(
        *reloadable_config
            .read()
            .expect("the reloadable config should remain readable"),
        initial_resolved_config,
    );
}

#[tokio::test]
async fn should_return_invalid_config_feedback_when_fixed_prompt_processing_tokens_are_zero() {
    let temp_config_directory = tempfile::tempdir().expect("a temp config directory is needed");
    let config_home_directory = temp_config_directory.path().to_path_buf();
    write_config_file(
        &config_home_directory,
        r#"{ "chunking": { "fixed_prompt_processing_chunk_size_tokens": 0 } }"#,
    );
    let initial_resolved_config = sample_resolved_config();
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config.clone()));
    let application = build_development_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        reloadable_config.clone(),
        config_home_directory,
    );

    let response = post_config_reload(&application).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the reload error body should be readable");
    let response_json: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the reload error body should be JSON");
    assert_eq!(response_json["status"], "invalid_config");
    assert_eq!(
        response_json["message"],
        "Configuration is invalid; correct the local configuration file and retry"
    );
    let live_config = reloadable_config
        .read()
        .expect("the reloadable config lock should be readable");
    assert_eq!(*live_config, initial_resolved_config);
}

#[tokio::test]
async fn should_return_busy_when_generation_is_active_during_config_reload() {
    let temp_config_directory = tempfile::tempdir().expect("a temp config directory is needed");
    let config_home_directory = temp_config_directory.path().to_path_buf();
    write_config_file(&config_home_directory, "{}");

    let initial_resolved_config = sample_resolved_config();
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config.clone()));
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.is_busy_override = true;
    let application = build_development_application_with_reload(
        executor,
        reloadable_config.clone(),
        config_home_directory,
    );

    let response = post_config_reload(&application).await;
    assert_eq!(
        response.status(),
        StatusCode::CONFLICT,
        "an active generation must produce HTTP 409"
    );
    let response_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the reload busy body should be readable");
    let response_json: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the reload busy body should be JSON");
    assert_eq!(response_json["status"], "busy");

    // Live state must be unchanged when busy.
    let live_config = reloadable_config
        .read()
        .expect("the reloadable config lock should be readable");
    assert_eq!(*live_config, initial_resolved_config);
}
