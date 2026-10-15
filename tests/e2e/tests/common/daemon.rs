//! Shared process-boundary plumbing for the root E2E journeys: spawning
//! the real daemon and fixture workers, HTTP probes, and instance config
//! fixtures.

use std::{net::SocketAddr, path::Path, process::Stdio, time::Duration};

use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    process::{Child, Command},
    time::{Instant, sleep, timeout},
};

pub(crate) const DAEMON_STARTUP_PREFIX: &str = "astronomicald listening on http://";

/// Every daemon this lane spawns carries the documented verification skip.
/// The daemon launches its worker eagerly, the worker's wired-capacity
/// verification walk is real machine graphics state, and daemon shutdown
/// waits for that worker — on loaded continuous-integration runners the
/// walk's timing gates daemon exits past the lifecycle budgets. This lane is
/// an offline lane: sealed from machine state by definition. Real serving
/// sessions still verify.
pub(crate) fn seal_daemon_from_machine_graphics_state(daemon_command: &mut Command) {
    daemon_command.env("ASTRONOMICAL_SKIP_GPU_WIRED_VERIFICATION", "1");
}
pub(crate) const TEST_WORKER_READY_MODEL_ID_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_TEST_WORKER_READY_MODEL_ID";
pub(crate) const DELAYED_MALFORMED_OUTPUT_MODEL_ID: &str =
    "astronomical/delayed-malformed-output-fixture";
pub(crate) const RESPONSES_FIXTURE_MODEL_ID: &str = "astronomical/accepted-chat-fixture";

pub(crate) async fn spawn_daemon(daemon_executable_path: &str) -> (Child, SocketAddr) {
    spawn_daemon_with_worker_ready_model(daemon_executable_path, None, None).await
}

pub(crate) async fn spawn_daemon_with_worker_ready_model(
    daemon_executable_path: &str,
    worker_executable_path: Option<&str>,
    worker_ready_model_id: Option<&str>,
) -> (Child, SocketAddr) {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    // The resolver validates prompt-cache policy before binding the listener.
    // Keep this fixture valid so individual tests isolate the worker behavior
    // they intend to exercise.
    write_v1_config(
        temp_home.path(),
        r#"{"chunking":{"fixed_prompt_processing_chunk_size_tokens":2048}}"#,
    );
    let mut daemon_command = Command::new(daemon_executable_path);
    seal_daemon_from_machine_graphics_state(&mut daemon_command);
    daemon_command
        .args(["--instance", "development", "--state-directory"])
        .arg(temp_home.path().join(".astronomical-dev"))
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

pub(crate) async fn post_streaming_chat_completion(
    daemon_address: SocketAddr,
    model_id: &str,
) -> String {
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

pub(crate) async fn post_response(daemon_address: SocketAddr, stream: bool) -> String {
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

pub(crate) async fn get_endpoint(daemon_address: SocketAddr, endpoint_path: &str) -> String {
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

pub(crate) async fn post_empty_endpoint(daemon_address: SocketAddr, endpoint_path: &str) -> String {
    let mut daemon_connection = TcpStream::connect(daemon_address)
        .await
        .expect("the daemon should accept a local connection");
    daemon_connection
        .write_all(
            format!(
                "POST {endpoint_path} HTTP/1.1\r\nHost: {daemon_address}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
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

pub(crate) async fn spawn_actual_instance_daemon(
    daemon_executable_path: &str,
    runtime_instance: &str,
    state_directory: &Path,
) -> (Child, SocketAddr) {
    let mut daemon_process = Command::new(daemon_executable_path)
        .args(["--instance", runtime_instance, "--state-directory"])
        .arg(state_directory)
        .env_remove("ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH")
        .env("ASTRONOMICAL_SKIP_GPU_WIRED_VERIFICATION", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("instance daemon should start");
    let stdout = daemon_process
        .stdout
        .take()
        .expect("daemon stdout should be captured");
    let mut stdout_lines = BufReader::new(stdout).lines();
    let startup_line = timeout(Duration::from_secs(3), stdout_lines.next_line())
        .await
        .expect("daemon should report its endpoint")
        .expect("daemon startup output should be readable")
        .expect("daemon startup output should not end");
    let daemon_address = startup_line
        .strip_prefix(DAEMON_STARTUP_PREFIX)
        .expect("daemon should report the expected startup prefix")
        .parse()
        .expect("daemon address should parse");
    (daemon_process, daemon_address)
}

pub(crate) async fn wait_until_endpoint_contains(
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

pub(crate) fn terminate_daemon(daemon_process: &Child) {
    let process_id = daemon_process.id().expect("the daemon should still run");
    let terminate_status = std::process::Command::new("kill")
        .args(["-TERM", &process_id.to_string()])
        .status()
        .expect("the test should send SIGTERM");
    assert!(terminate_status.success());
}

pub(crate) fn write_v1_config(home_directory: &Path, configured_fields_json: &str) {
    let configured_fields: serde_json::Value = serde_json::from_str(configured_fields_json)
        .expect("the partial daemon config should be valid JSON");
    let mut config_document = serde_json::json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": { "model_directories": [] }
    });
    for (field_name, field_value) in configured_fields
        .as_object()
        .expect("the partial daemon config should be an object")
    {
        config_document
            .as_object_mut()
            .expect("the v1 daemon config should be an object")
            .insert(field_name.clone(), field_value.clone());
    }
    write_raw_config(
        home_directory,
        &serde_json::to_string_pretty(&config_document)
            .expect("the daemon config should serialize"),
    );
}

pub(crate) fn write_raw_config(home_directory: &Path, config_json: &str) {
    let config_directory = home_directory.join(".astronomical-dev");
    std::fs::create_dir_all(&config_directory).expect("config directory should be created");
    std::fs::write(config_directory.join("config.json"), config_json)
        .expect("config file should be written");
}

pub(crate) fn write_instance_config(state_directory: &Path) {
    // The eager startup worker composes a machine-scale ceiling when no
    // ceiling is configured, and its wired-capacity verification walk gates
    // daemon shutdown; a small pinned ceiling keeps parallel lifecycle
    // assertions inside their exit budgets on every machine.
    std::fs::write(
        state_directory.join("config.json"),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[],"maximum_mlx_memory_gb":2},"chunking":{"fixed_prompt_processing_chunk_size_tokens":2048}}"#,
    )
    .expect("instance config should be written");
}
