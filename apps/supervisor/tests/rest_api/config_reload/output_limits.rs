use super::*;

#[tokio::test]
async fn should_cap_chat_generation_to_the_reloaded_output_token_ceiling() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    let mut initial_resolved_config = sample_resolved_config();
    initial_resolved_config.max_output_tokens = 20_480;
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_generation_commands = executor.received_generation_commands();
    let application = build_development_application_with_reload(
        executor,
        Arc::clone(&reloadable_config),
        config_home_directory,
    );
    reloadable_config
        .write()
        .expect("the reloadable config should be writable")
        .max_output_tokens = 5_000;

    let response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"astronomical/application-test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":20000,"stream":true}"#,
                ))
                .expect("the capped chat request should be valid"),
        )
        .await
        .expect("the application should return a response");

    assert_eq!(response.status(), StatusCode::OK);
    let received_generation_commands = received_generation_commands
        .lock()
        .expect("the scripted executor command log should be readable");
    assert_eq!(received_generation_commands.len(), 1);
    assert_eq!(
        received_generation_commands[0].settings.max_output_tokens, 5_000,
        "the reloaded configuration must cap an OpenCode 20,000-token request"
    );
}

#[tokio::test]
async fn should_cap_responses_generation_to_the_reloaded_output_token_ceiling() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    let mut initial_resolved_config = sample_resolved_config();
    initial_resolved_config.max_output_tokens = 20_480;
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_generation_commands = executor.received_generation_commands();
    let application = build_development_application_with_reload(
        executor,
        Arc::clone(&reloadable_config),
        config_home_directory,
    );
    reloadable_config
        .write()
        .expect("the reloadable config should be writable")
        .max_output_tokens = 5_000;

    let response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"astronomical/application-test-model","input":"hello","max_output_tokens":20000,"stream":true}"#,
                ))
                .expect("the capped Responses request should be valid"),
        )
        .await
        .expect("the application should return a response");

    assert_eq!(response.status(), StatusCode::OK);
    let received_generation_commands = received_generation_commands
        .lock()
        .expect("the scripted executor command log should be readable");
    assert_eq!(received_generation_commands.len(), 1);
    assert_eq!(
        received_generation_commands[0].settings.max_output_tokens, 5_000,
        "the reloaded configuration must cap an OpenCode-compatible Responses request"
    );
}
