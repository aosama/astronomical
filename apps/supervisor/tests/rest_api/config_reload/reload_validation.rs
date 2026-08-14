use super::*;

#[tokio::test]
async fn should_reject_invalid_config_reload_without_mutating_live_state() {
    let temp_config_directory = tempfile::tempdir().expect("a temp config directory is needed");
    let config_home_directory = temp_config_directory.path().to_path_buf();
    write_config_file(&config_home_directory, "{ this is not valid json }");

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

    // Live state must be unchanged after an invalid reload.
    let live_config = reloadable_config
        .read()
        .expect("the reloadable config lock should be readable");
    assert_eq!(*live_config, initial_resolved_config);
}

#[tokio::test]
async fn should_reject_enabled_speculative_prefill_without_an_explicit_keep_percentage() {
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
        "invalid Astronomical configuration: speculative_prefill.keep_percentage is required when speculative prefill is enabled",
    );
    assert_eq!(
        *reloadable_config
            .read()
            .expect("the reloadable config should remain readable"),
        initial_resolved_config,
    );
}

#[tokio::test]
async fn should_return_existing_invalid_config_feedback_when_fixed_prefill_chunck_tokens_are_missing()
 {
    let temp_config_directory = tempfile::tempdir().expect("a temp config directory is needed");
    let config_home_directory = temp_config_directory.path().to_path_buf();
    write_config_file(
        &config_home_directory,
        r#"{ "chunking": { "prefill_size_optimizer_enabled": false } }"#,
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
        "invalid Astronomical configuration: chunking.fixed_prefill_tokens is required when chunking.prefill_size_optimizer_enabled is false"
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
