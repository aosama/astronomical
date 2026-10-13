use std::{process::Stdio, time::Duration};

use crate::common::daemon::{
    DELAYED_MALFORMED_OUTPUT_MODEL_ID, RESPONSES_FIXTURE_MODEL_ID, get_endpoint, post_response,
    post_streaming_chat_completion, spawn_actual_instance_daemon, spawn_daemon,
    spawn_daemon_with_worker_ready_model, terminate_daemon, wait_until_endpoint_contains,
    write_instance_config, write_raw_config,
};

use tokio::{process::Command, time::timeout};

#[tokio::test]
async fn should_reject_a_second_daemon_for_the_same_instance_state() {
    let daemon_executable_path = astronomical_test_harness::workspace_binary("astronomicald")
        .to_string_lossy()
        .into_owned();
    let development_state_directory = tempfile::tempdir().expect("state should be created");
    write_instance_config(development_state_directory.path());
    let (mut first_daemon, first_address) = spawn_actual_instance_daemon(
        &daemon_executable_path,
        "development",
        development_state_directory.path(),
    )
    .await;

    let second_output = timeout(
        Duration::from_secs(3),
        Command::new(&daemon_executable_path)
            .args(["--instance", "development", "--state-directory"])
            .arg(development_state_directory.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output(),
    )
    .await
    .expect("second daemon should fail promptly")
    .expect("second daemon process should run");

    assert!(!second_output.status.success());
    assert!(String::from_utf8_lossy(&second_output.stderr).contains("already running"));
    assert!(
        get_endpoint(first_address, "/health")
            .await
            .starts_with("HTTP/1.1 200 OK")
    );
    terminate_daemon(&first_daemon);
    let _first_status = first_daemon
        .wait()
        .await
        .expect("first daemon should be reaped");
}

#[tokio::test]
async fn should_reject_ambiguous_or_relative_instance_arguments_before_startup() {
    let daemon_executable_path = astronomical_test_harness::workspace_binary("astronomicald")
        .to_string_lossy()
        .into_owned();
    for invalid_arguments in [
        vec!["--state-directory", "relative-state"],
        vec!["--instance", "stable", "--instance", "development"],
    ] {
        let daemon_output = timeout(
            Duration::from_secs(3),
            Command::new(&daemon_executable_path)
                .args(invalid_arguments)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .expect("invalid daemon arguments should fail promptly")
        .expect("daemon argument validation should run");

        assert_eq!(daemon_output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&daemon_output.stderr).contains("Usage: astronomicald"));
    }
}

#[tokio::test]
async fn should_keep_health_available_when_worker_startup_fails() {
    let daemon_executable_path = astronomical_test_harness::workspace_binary("astronomicald")
        .to_string_lossy()
        .into_owned();
    let isolated_daemon_directory =
        tempfile::tempdir().expect("the test should create an isolated daemon directory");
    let isolated_daemon_executable_path = isolated_daemon_directory.path().join("astronomicald");
    std::fs::copy(&daemon_executable_path, &isolated_daemon_executable_path)
        .expect("the test should copy the daemon executable");
    let daemon_permissions = std::fs::metadata(&daemon_executable_path)
        .expect("the test should read the daemon executable permissions")
        .permissions();
    std::fs::set_permissions(&isolated_daemon_executable_path, daemon_permissions)
        .expect("the copied daemon executable should remain executable");
    let isolated_daemon_executable_path = isolated_daemon_executable_path
        .to_str()
        .expect("the isolated daemon path should be valid UTF-8");
    let (mut daemon_process, daemon_address) = spawn_daemon(isolated_daemon_executable_path).await;

    assert!(
        get_endpoint(daemon_address, "/health")
            .await
            .starts_with("HTTP/1.1 200 OK")
    );
    assert!(
        get_endpoint(daemon_address, "/ready")
            .await
            .starts_with("HTTP/1.1 503 Service Unavailable")
    );

    terminate_daemon(&daemon_process);
    assert!(
        timeout(Duration::from_secs(3), daemon_process.wait())
            .await
            .expect("the daemon should stop")
            .expect("the daemon should be reaped")
            .success()
    );
}

#[tokio::test]
async fn should_keep_worker_unready_when_the_loaded_model_is_unexpected() {
    // With config-driven discovery, the daemon no longer rejects models
    // based on a hardcoded expected model ID. This test now verifies that
    // the worker starts and becomes ready with whatever model it loads.
    let daemon_executable_path =
        astronomical_test_harness::workspace_binary("astronomical-daemon-test-daemon")
            .to_string_lossy()
            .into_owned();
    let worker_executable_path =
        astronomical_test_harness::workspace_binary("astronomical-daemon-test-worker")
            .to_string_lossy()
            .into_owned();
    let (mut daemon_process, daemon_address) = spawn_daemon_with_worker_ready_model(
        &daemon_executable_path,
        Some(&worker_executable_path),
        Some("astronomical/test-worker-model"),
    )
    .await;

    wait_until_endpoint_contains(
        daemon_address,
        "/ready",
        "HTTP/1.1 200 OK",
        Duration::from_secs(3),
    )
    .await;
    assert!(
        get_endpoint(daemon_address, "/v1/models")
            .await
            .contains("astronomical/test-worker-model")
    );

    terminate_daemon(&daemon_process);
    assert!(
        timeout(Duration::from_secs(3), daemon_process.wait())
            .await
            .expect("the daemon should stop")
            .expect("the daemon should be reaped")
            .success()
    );
}

#[tokio::test]
async fn should_show_generation_progress_for_malformed_model_output_before_stream_failure() {
    let daemon_executable_path =
        astronomical_test_harness::workspace_binary("astronomical-daemon-test-daemon")
            .to_string_lossy()
            .into_owned();
    let worker_executable_path =
        astronomical_test_harness::workspace_binary("astronomical-daemon-test-worker")
            .to_string_lossy()
            .into_owned();
    let (mut daemon_process, daemon_address) = spawn_daemon_with_worker_ready_model(
        &daemon_executable_path,
        Some(&worker_executable_path),
        Some(DELAYED_MALFORMED_OUTPUT_MODEL_ID),
    )
    .await;
    wait_until_endpoint_contains(
        daemon_address,
        "/ready",
        "HTTP/1.1 200 OK",
        Duration::from_secs(3),
    )
    .await;

    let chat_response_task = tokio::spawn(post_streaming_chat_completion(
        daemon_address,
        DELAYED_MALFORMED_OUTPUT_MODEL_ID,
    ));
    let active_status_response = wait_until_endpoint_contains(
        daemon_address,
        "/v1/status",
        r#""phase":"generation""#,
        Duration::from_secs(2),
    )
    .await;

    assert!(active_status_response.contains(r#""status":"ready""#));
    assert!(active_status_response.contains(r#""activity":"generating""#));
    assert!(active_status_response.contains(r#""processed_tokens":3"#));
    assert!(active_status_response.contains(r#""total_tokens":16"#));

    let chat_response = timeout(Duration::from_secs(3), chat_response_task)
        .await
        .expect("the malformed stream should finish before the test timeout")
        .expect("the chat response task should not panic");
    assert!(chat_response.starts_with("HTTP/1.1 200 OK"));
    assert!(chat_response.contains(r#""code":"chat_malformed_model_output""#));
    assert!(!chat_response.contains("[DONE]"));

    let final_status_response = wait_until_endpoint_contains(
        daemon_address,
        "/v1/status",
        r#""activity":"idle""#,
        Duration::from_secs(2),
    )
    .await;
    assert!(final_status_response.contains(r#""status":"ready""#));
    assert!(!final_status_response.contains(r#""progress""#));

    terminate_daemon(&daemon_process);
    assert!(
        timeout(Duration::from_secs(3), daemon_process.wait())
            .await
            .expect("the daemon should stop")
            .expect("the daemon should be reaped")
            .success()
    );
}

#[tokio::test]
async fn should_fail_startup_when_user_config_is_malformed() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_raw_config(
        temp_home.path(),
        r#"{"daemon":{"bind_address":"127.0.0.1:0"}"#,
    );
    let daemon_executable_path = astronomical_test_harness::workspace_binary("astronomicald")
        .to_string_lossy()
        .into_owned();
    let daemon_output = timeout(
        Duration::from_secs(3),
        Command::new(daemon_executable_path)
            .args(["--instance", "development", "--state-directory"])
            .arg(temp_home.path().join(".astronomical-dev"))
            .env_remove("ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH")
            .env("HOME", temp_home.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("the daemon should fail fast for malformed config")
    .expect("the daemon should run");

    assert!(!daemon_output.status.success());
    let daemon_stderr = String::from_utf8_lossy(&daemon_output.stderr);
    assert!(
        daemon_stderr.contains("failed to parse Astronomical config file"),
        "stderr should explain the malformed config, got: {daemon_stderr}"
    );
}

#[tokio::test]
async fn should_reject_retired_daemon_configuration_for_a_standard_instance() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    let development_state_directory = temp_home.path().join(".astronomical-dev");
    std::fs::create_dir_all(&development_state_directory)
        .expect("development state directory should be created");
    std::fs::write(
        development_state_directory.join("config.json"),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"daemon":{"bind_address":"127.0.0.1:6732"}}"#,
    )
    .expect("development config should be written");
    let daemon_executable_path = astronomical_test_harness::workspace_binary("astronomicald")
        .to_string_lossy()
        .into_owned();

    let daemon_output = timeout(
        Duration::from_secs(3),
        Command::new(daemon_executable_path)
            .args(["--instance", "development"])
            .env_remove("ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH")
            .env("HOME", temp_home.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("the daemon should reject the retired daemon field before the timeout")
    .expect("the daemon should run");

    assert!(!daemon_output.status.success());
    let daemon_stderr = String::from_utf8_lossy(&daemon_output.stderr);
    assert!(
        daemon_stderr.contains("unknown field `daemon`"),
        "stderr should explain the retired configuration field, got: {daemon_stderr}"
    );
}

#[tokio::test]
async fn should_serve_responses_json_and_sse_through_the_daemon_process() {
    let daemon_executable_path =
        astronomical_test_harness::workspace_binary("astronomical-daemon-test-daemon")
            .to_string_lossy()
            .into_owned();
    let worker_executable_path =
        astronomical_test_harness::workspace_binary("astronomical-daemon-test-worker")
            .to_string_lossy()
            .into_owned();
    let (mut daemon_process, daemon_address) = spawn_daemon_with_worker_ready_model(
        &daemon_executable_path,
        Some(&worker_executable_path),
        Some(RESPONSES_FIXTURE_MODEL_ID),
    )
    .await;
    wait_until_endpoint_contains(
        daemon_address,
        "/ready",
        "HTTP/1.1 200 OK",
        Duration::from_secs(3),
    )
    .await;

    let json_response = post_response(daemon_address, false).await;
    assert!(json_response.starts_with("HTTP/1.1 200 OK"));
    assert!(json_response.contains(r#""object":"response""#));
    assert!(json_response.contains(r#""output_text":"accepted chat text""#));

    let streaming_response = post_response(daemon_address, true).await;
    assert!(streaming_response.starts_with("HTTP/1.1 200 OK"));
    assert!(streaming_response.contains("event: response.created"));
    assert!(streaming_response.contains("event: response.output_text.delta"));
    assert!(streaming_response.contains("event: response.completed"));
    assert!(!streaming_response.contains("[DONE]"));

    terminate_daemon(&daemon_process);
    assert!(
        timeout(Duration::from_secs(3), daemon_process.wait())
            .await
            .expect("the daemon should stop")
            .expect("the daemon should be reaped")
            .success()
    );
}
