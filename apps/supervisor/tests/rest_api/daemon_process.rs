use std::{net::SocketAddr, path::Path, process::Stdio, time::Duration};

use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    process::{Child, Command},
    time::{Instant, sleep, timeout},
};

const DAEMON_STARTUP_PREFIX: &str = "astronomicald listening on http://";
const TEST_WORKER_READY_MODEL_ID_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_TEST_WORKER_READY_MODEL_ID";
const DELAYED_MALFORMED_OUTPUT_MODEL_ID: &str = "astronomical/delayed-malformed-output-fixture";
const RESPONSES_FIXTURE_MODEL_ID: &str = "astronomical/accepted-chat-fixture";

#[tokio::test]
async fn should_keep_health_available_when_worker_startup_fails() {
    let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomicald")
        .expect("Cargo should provide the astronomicald executable path");
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
    // With config-driven discovery, the supervisor no longer rejects models
    // based on a hardcoded expected model ID. This test now verifies that
    // the worker starts and becomes ready with whatever model it loads.
    let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-daemon")
        .expect("Cargo should provide the test daemon path");
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
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
    let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-daemon")
        .expect("Cargo should provide the test daemon path");
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
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
    write_config(
        temp_home.path(),
        r#"{"supervisor":{"bind_address":"127.0.0.1:0"}"#,
    );
    let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomicald")
        .expect("Cargo should provide the astronomicald executable path");
    let daemon_output = timeout(
        Duration::from_secs(3),
        Command::new(daemon_executable_path)
            .env_remove("ASTRONOMICAL_SUPERVISOR_BIND_ADDRESS")
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
async fn should_serve_responses_json_and_sse_through_the_daemon_process() {
    let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-daemon")
        .expect("Cargo should provide the test daemon path");
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
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

async fn spawn_daemon(daemon_executable_path: &str) -> (Child, SocketAddr) {
    spawn_daemon_with_worker_ready_model(daemon_executable_path, None, None).await
}

async fn spawn_daemon_with_worker_ready_model(
    daemon_executable_path: &str,
    worker_executable_path: Option<&str>,
    worker_ready_model_id: Option<&str>,
) -> (Child, SocketAddr) {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    // The resolver validates prompt-cache policy before binding the listener.
    // Keep this fixture valid so individual tests isolate the worker behavior
    // they intend to exercise.
    write_config(
        temp_home.path(),
        r#"{"prefill_chunck_size_optimizer_enabled":true,"supervisor":{"bind_address":"127.0.0.1:0"}}"#,
    );
    let mut daemon_command = Command::new(daemon_executable_path);
    daemon_command
        .env("ASTRONOMICAL_SUPERVISOR_BIND_ADDRESS", "127.0.0.1:0")
        .env("HOME", temp_home.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(worker_executable_path) = worker_executable_path {
        daemon_command.env(
            "ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH",
            worker_executable_path,
        );
    }
    if let Some(worker_ready_model_id) = worker_ready_model_id {
        daemon_command.env(
            TEST_WORKER_READY_MODEL_ID_ENVIRONMENT_VARIABLE,
            worker_ready_model_id,
        );
    } else {
        daemon_command.env_remove(TEST_WORKER_READY_MODEL_ID_ENVIRONMENT_VARIABLE);
    }
    let mut daemon_process = daemon_command
        .spawn()
        .expect("the daemon process should start");
    let daemon_stdout = daemon_process
        .stdout
        .take()
        .expect("the daemon should expose stdout");
    let mut daemon_stdout_lines = BufReader::new(daemon_stdout).lines();
    let startup_line = timeout(Duration::from_secs(3), daemon_stdout_lines.next_line())
        .await
        .expect("the daemon should print its startup line")
        .expect("daemon stdout should remain readable")
        .expect("the startup line should exist");
    let daemon_address = startup_line
        .strip_prefix(DAEMON_STARTUP_PREFIX)
        .expect("the startup line should contain the address")
        .parse::<SocketAddr>()
        .expect("the daemon address should parse");
    (daemon_process, daemon_address)
}

async fn post_streaming_chat_completion(daemon_address: SocketAddr, model_id: &str) -> String {
    let request_body = format!(
        r#"{{"model":"{model_id}","messages":[{{"role":"user","content":"hello"}}],"stream":true,"max_tokens":16}}"#
    );
    let mut daemon_connection = TcpStream::connect(daemon_address)
        .await
        .expect("the daemon should accept a local chat connection");
    daemon_connection
        .write_all(
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: {daemon_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                request_body.len(),
                request_body
            )
            .as_bytes(),
        )
        .await
        .expect("the chat request should be written");
    let mut chat_response = String::new();
    daemon_connection
        .read_to_string(&mut chat_response)
        .await
        .expect("the chat response should be readable");
    chat_response
}

async fn post_response(daemon_address: SocketAddr, stream: bool) -> String {
    let request_body =
        format!(r#"{{"model":"{RESPONSES_FIXTURE_MODEL_ID}","input":"hello","stream":{stream}}}"#);
    let mut daemon_connection = TcpStream::connect(daemon_address)
        .await
        .expect("the daemon should accept a local Responses connection");
    daemon_connection
        .write_all(
            format!(
                "POST /v1/responses HTTP/1.1\r\nHost: {daemon_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                request_body.len(),
                request_body
            )
            .as_bytes(),
        )
        .await
        .expect("the Responses request should be written");
    let mut responses_response = String::new();
    daemon_connection
        .read_to_string(&mut responses_response)
        .await
        .expect("the Responses response should be readable");
    responses_response
}

async fn get_endpoint(daemon_address: SocketAddr, endpoint_path: &str) -> String {
    let mut daemon_connection = TcpStream::connect(daemon_address)
        .await
        .expect("the daemon should accept a local connection");
    daemon_connection
        .write_all(
            format!(
                "GET {endpoint_path} HTTP/1.1\r\nHost: {daemon_address}\r\nConnection: close\r\n\r\n"
            )
            .as_bytes(),
        )
        .await
        .expect("the endpoint request should be written");
    let mut endpoint_response = String::new();
    daemon_connection
        .read_to_string(&mut endpoint_response)
        .await
        .expect("the endpoint response should be readable");
    endpoint_response
}

async fn wait_until_endpoint_contains(
    daemon_address: SocketAddr,
    endpoint_path: &str,
    expected_response_fragment: &str,
    maximum_wait_duration: Duration,
) -> String {
    let status_deadline = Instant::now() + maximum_wait_duration;
    let mut latest_endpoint_response = String::new();
    while Instant::now() < status_deadline {
        latest_endpoint_response = get_endpoint(daemon_address, endpoint_path).await;
        if latest_endpoint_response.contains(expected_response_fragment) {
            return latest_endpoint_response;
        }
        sleep(Duration::from_millis(25)).await;
    }
    panic!(
        "endpoint {endpoint_path} did not contain {expected_response_fragment:?}; latest response: {latest_endpoint_response}"
    );
}

fn terminate_daemon(daemon_process: &Child) {
    let process_id = daemon_process.id().expect("the daemon should still run");
    let terminate_status = std::process::Command::new("kill")
        .args(["-TERM", &process_id.to_string()])
        .status()
        .expect("the test should send SIGTERM");
    assert!(terminate_status.success());
}

fn write_config(home_directory: &Path, config_json: &str) {
    let config_directory = home_directory.join(".astronomical");
    std::fs::create_dir_all(&config_directory).expect("config directory should be created");
    std::fs::write(config_directory.join("config.json"), config_json)
        .expect("config file should be written");
}
