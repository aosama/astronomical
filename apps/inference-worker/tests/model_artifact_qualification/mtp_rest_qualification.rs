use std::fs;

use serde_json::{Value, json};
use tokio::time::timeout;

use super::{
    model_artifact_rest_qualification::{
        E2E_TIMEOUT, assert_successful_streaming_chat_response, get_endpoint,
        launch_model_artifact_rest_server_for_model, post_chat_completion,
        stop_model_artifact_rest_server,
    },
    model_artifact_rest_transport::streamed_model_text_from_chat_response,
};

const MTP_MODEL_ID: &str = "Qwen3.6-35B-A3B-oQ4e-mtp";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads configured Qwen3.6-35B-A3B-oQ4e-mtp through the production worker and REST surface"]
async fn should_keep_mtp_active_across_two_greedy_chat_sse_requests_on_one_worker() {
    timeout(E2E_TIMEOUT, run_mtp_rest_qualification())
        .await
        .expect("the MTP REST qualification must finish within 115 seconds");
}

async fn run_mtp_rest_qualification() {
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MTP_MODEL_ID);
    assert!(
        model_directory.is_dir(),
        "the MTP qualification requires {}",
        model_directory.display()
    );
    let isolated_worker_home =
        tempfile::tempdir().expect("the isolated MTP worker home directory should be created");
    let configuration_directory = isolated_worker_home.path().join(".astronomical");
    fs::create_dir(&configuration_directory)
        .expect("the isolated MTP configuration directory should be created");
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&json!({
            "model_directories": [model_directory],
            "mtp_enabled": true,
        }))
        .expect("the isolated MTP configuration should serialize"),
    )
    .expect("the isolated MTP configuration should be written");

    let model_artifact_rest_server = launch_model_artifact_rest_server_for_model(
        MTP_MODEL_ID,
        model_directory,
        Some(isolated_worker_home.path()),
    )
    .await;
    let server_address = model_artifact_rest_server.server_address;

    let first_chat_response = post_chat_completion(
        server_address,
        deterministic_greedy_request_body("Reply with exactly MTP_FIRST_OK and nothing else."),
    )
    .await;
    assert_successful_streaming_chat_response(&first_chat_response);
    let first_model_output = streamed_model_text_from_chat_response(&first_chat_response);
    assert!(
        first_model_output.contains("MTP_FIRST_OK"),
        "the first MTP response must contain the requested deterministic output: {first_model_output:?}"
    );
    let first_status_document = status_document(server_address).await;
    assert_active_mtp_status(&first_status_document);
    assert_eq!(
        first_status_document["serving_session"]["completed_request_count"],
        1
    );
    eprintln!("[mtp-rest 1/2] first greedy SSE request completed with active MTP");

    let second_chat_response = post_chat_completion(
        server_address,
        deterministic_greedy_request_body("Reply with exactly MTP_REUSE_OK and nothing else."),
    )
    .await;
    assert_successful_streaming_chat_response(&second_chat_response);
    let second_model_output = streamed_model_text_from_chat_response(&second_chat_response);
    assert!(
        second_model_output.contains("MTP_REUSE_OK"),
        "the reused worker response must contain the requested deterministic output: {second_model_output:?}"
    );
    let second_status_document = status_document(server_address).await;
    assert_active_mtp_status(&second_status_document);
    assert_eq!(
        second_status_document["serving_session"]["completed_request_count"], 2,
        "the worker-owned session counter must prove both requests used one worker"
    );
    eprintln!("[mtp-rest 2/2] second greedy SSE request reused the active MTP worker");

    stop_model_artifact_rest_server(model_artifact_rest_server).await;
}

fn deterministic_greedy_request_body(user_prompt: &str) -> String {
    json!({
        "model": MTP_MODEL_ID,
        "messages": [{ "role": "user", "content": user_prompt }],
        "stream": true,
        "temperature": 0,
        "max_tokens": 32,
    })
    .to_string()
}

async fn status_document(server_address: std::net::SocketAddr) -> Value {
    let status_response = get_endpoint(server_address, "/v1/status").await;
    assert!(
        status_response.starts_with("HTTP/1.1 200 OK"),
        "the public status endpoint must succeed: {status_response}"
    );
    let status_body = status_response
        .split("\r\n\r\n")
        .nth(1)
        .expect("the status response should contain a body");
    serde_json::from_str(status_body).expect("the status response should contain JSON")
}

fn assert_active_mtp_status(status_document: &Value) {
    assert_eq!(status_document["status"], "ready");
    assert_eq!(status_document["ready_model_id"], MTP_MODEL_ID);
    assert_eq!(status_document["mtp_enabled"], true);
    assert_eq!(status_document["mtp_runtime_state"], "active");
    assert_eq!(status_document["mtp_unavailable_reason"], Value::Null);
}
