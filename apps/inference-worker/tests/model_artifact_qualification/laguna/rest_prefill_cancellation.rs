//! Public cancellation during Laguna XS prefill leaves the worker reusable.

use std::time::{Duration, Instant};

use serde_json::json;
use tokio::time::{sleep, timeout};

use super::artifact::{
    LAGUNA_XS_PUBLIC_MODEL_ID, compact_romeo_and_juliet_source, full_romeo_and_juliet_source,
    resolve_reference_model_directory,
};
use super::http::opencode_shaped_chat_request_body;
use crate::model_artifact_qualification::model_artifact_rest_qualification::{
    assert_successful_streaming_chat_response, get_endpoint,
    launch_model_artifact_rest_server_for_model, post_chat_completion,
    stop_model_artifact_rest_server,
};

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(119);
const STATUS_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "cancels reference Laguna XS during public prefill and reuses the worker"]
async fn should_acknowledge_laguna_xs_rest_prefill_cancellation_and_reuse_worker() {
    timeout(JOURNEY_TIMEOUT, run_cancellation_journey())
        .await
        .expect("the Laguna XS cancellation journey must finish within 119 seconds");
}

async fn run_cancellation_journey() {
    let model_directory = resolve_reference_model_directory();
    let public_model_id = LAGUNA_XS_PUBLIC_MODEL_ID;
    let isolated_home = crate::common::isolated_development_home_from_user_config();
    let rest_server = launch_model_artifact_rest_server_for_model(
        public_model_id,
        model_directory,
        Some(isolated_home.path()),
        None,
    )
    .await;
    let server_address = rest_server.server_address;
    let reusable_request_body = json!({
        "model": public_model_id,
        "messages": [{"role": "user", "content": format!(
            "Use the supplied Romeo and Juliet source. Name the households.\n\n{}",
            compact_romeo_and_juliet_source()
        )}],
        "stream": true,
        "temperature": 1,
        "max_tokens": 2,
    })
    .to_string();
    let warm_response = post_chat_completion(server_address, reusable_request_body.clone()).await;
    assert_successful_streaming_chat_response(&warm_response);

    let long_request_body =
        opencode_shaped_chat_request_body(public_model_id, full_romeo_and_juliet_source(), 8);
    let long_request_task = tokio::spawn(post_chat_completion(server_address, long_request_body));
    wait_for_incomplete_prefill(server_address).await;
    eprintln!("[laguna-rest-cancel] phase=disconnect");
    long_request_task.abort();
    let _aborted_request = long_request_task.await;
    let cancellation_started_at = Instant::now();
    wait_until_ready(server_address).await;
    eprintln!(
        "[laguna-rest-cancel] phase=acknowledged elapsed_millis={}",
        cancellation_started_at.elapsed().as_millis()
    );

    let reused_response = post_chat_completion(server_address, reusable_request_body).await;
    assert_successful_streaming_chat_response(&reused_response);
    stop_model_artifact_rest_server(rest_server).await;
}

async fn wait_for_incomplete_prefill(server_address: std::net::SocketAddr) {
    let mut poll_attempt = 0_u32;
    loop {
        poll_attempt = poll_attempt.saturating_add(1);
        let status_document = status_document(server_address).await;
        let processed_tokens = status_document["progress"]["processed_tokens"]
            .as_u64()
            .unwrap_or(0);
        let total_tokens = status_document["progress"]["total_tokens"]
            .as_u64()
            .unwrap_or(0);
        if processed_tokens > 0 && processed_tokens < total_tokens {
            return;
        }
        if poll_attempt.is_multiple_of(20) {
            eprintln!("[laguna-rest-cancel] phase=wait-prefill poll={poll_attempt}");
        }
        sleep(STATUS_POLL_INTERVAL).await;
    }
}

async fn wait_until_ready(server_address: std::net::SocketAddr) {
    let mut poll_attempt = 0_u32;
    loop {
        poll_attempt = poll_attempt.saturating_add(1);
        let status_document = status_document(server_address).await;
        if status_document["status"] == "ready" && status_document["activity"] == "idle" {
            return;
        }
        if poll_attempt.is_multiple_of(20) {
            eprintln!("[laguna-rest-cancel] phase=wait-ready poll={poll_attempt}");
        }
        sleep(STATUS_POLL_INTERVAL).await;
    }
}

async fn status_document(server_address: std::net::SocketAddr) -> serde_json::Value {
    let status_response = get_endpoint(server_address, "/v1/status").await;
    let response_body = status_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or(status_response.as_str());
    serde_json::from_str(response_body)
        .unwrap_or_else(|_| panic!("GET /v1/status should return JSON: {status_response}"))
}
