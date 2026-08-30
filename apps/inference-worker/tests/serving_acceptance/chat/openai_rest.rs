use std::{net::SocketAddr, time::Duration};

use serde_json::json;

use crate::small_dense_model::configured_deployment_litmus_model;
use crate::support::http::send_http_request;

pub(crate) use crate::support::serving_rest::{
    ServingRestServer, launch_serving_rest_server_for_model, stop_serving_rest_server,
};

pub(crate) const E2E_TIMEOUT: Duration = Duration::from_secs(115);
fn model_id() -> &'static str {
    crate::support::large_sparse_moe_model_id()
}
// The litmus checks stream completion and worker reuse, not long output volume.
const DEPLOYMENT_LITMUS_MAX_OUTPUT_TOKENS: u32 = 512;
pub(crate) const DEPLOYMENT_LITMUS_PROMPT: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

pub(crate) async fn run_serving_chat_request(request_kind: &str, request_body: String) {
    let chat_response =
        run_default_model_chat_request_and_return_response(request_kind, request_body).await;
    assert_successful_streaming_chat_response(&chat_response);
    eprintln!("[serving] {request_kind} output streamed and the worker was reaped");
}

async fn run_default_model_chat_request_and_return_response(
    request_kind: &str,
    request_body: String,
) -> String {
    let serving_rest_server = launch_serving_rest_server().await;
    run_model_artifact_request_and_return_response_with_server(
        serving_rest_server,
        request_kind,
        request_body,
    )
    .await
}

pub(crate) async fn run_serving_chat_request_for_model(
    model_id: &str,
    model_directory: std::path::PathBuf,
    request_kind: &str,
    request_body: String,
) -> String {
    let model_artifact_rest_server =
        launch_serving_rest_server_for_model(model_id, model_directory, None, None).await;
    run_model_artifact_request_and_return_response_with_server(
        model_artifact_rest_server,
        request_kind,
        request_body,
    )
    .await
}

async fn run_model_artifact_request_and_return_response_with_server(
    model_artifact_rest_server: ServingRestServer,
    request_kind: &str,
    request_body: String,
) -> String {
    eprintln!("[e2e] sending one model-artifact OpenAI-compatible {request_kind} chat request");
    let chat_response =
        post_chat_completion(model_artifact_rest_server.server_address, request_body).await;

    stop_serving_rest_server(model_artifact_rest_server).await;

    chat_response
}

pub(crate) async fn run_deployed_rest_surface_litmus() {
    let selected_deployment_litmus_model = configured_deployment_litmus_model();
    let deployment_litmus_model_id = selected_deployment_litmus_model.model_id;
    let model_artifact_rest_server = launch_serving_rest_server_for_model(
        &deployment_litmus_model_id,
        selected_deployment_litmus_model.model_directory,
        None,
        None,
    )
    .await;
    let server_address = model_artifact_rest_server.server_address;
    let repeated_long_prompt = format!(
        "{}\n\nReply with exactly OK.",
        DEPLOYMENT_LITMUS_PROMPT.repeat(3)
    );
    let first_chat_response = post_chat_completion(
        server_address,
        deployment_litmus_chat_request_body(&deployment_litmus_model_id, &repeated_long_prompt),
    )
    .await;
    assert_successful_streaming_chat_response(&first_chat_response);
    eprintln!("[deployment-litmus 1/4] status=success phase=initial_long_chat_request");

    let short_chat_response = post_chat_completion(
        server_address,
        text_chat_request_body_for_model(&deployment_litmus_model_id),
    )
    .await;
    assert_successful_streaming_chat_response(&short_chat_response);
    eprintln!("[deployment-litmus 2/4] status=success phase=intervening_short_chat_request");

    let reused_prompt = format!(
        "{repeated_long_prompt}\n\nConfirm that the second request reached the same deployed worker."
    );
    let second_long_chat_response = post_chat_completion(
        server_address,
        deployment_litmus_chat_request_body(&deployment_litmus_model_id, &reused_prompt),
    )
    .await;
    assert_successful_streaming_chat_response(&second_long_chat_response);
    eprintln!("[deployment-litmus 3/4] status=success phase=reused_long_chat_request");

    let reused_prompt_responses_response = post_responses_completion(
        server_address,
        deployment_litmus_responses_request_body(&deployment_litmus_model_id, &reused_prompt),
    )
    .await;
    assert_successful_streaming_responses_response(&reused_prompt_responses_response);
    let ready_response_after_prompt_reuse = get_endpoint(server_address, "/ready").await;
    assert!(
        ready_response_after_prompt_reuse.starts_with("HTTP/1.1 200 OK"),
        "the deployed worker must remain ready after prompt reuse: {ready_response_after_prompt_reuse}"
    );
    eprintln!(
        "[deployment-litmus 4/4] status=success phase=reused_long_responses_request worker_ready=true"
    );

    stop_serving_rest_server(model_artifact_rest_server).await;
}

async fn launch_serving_rest_server() -> ServingRestServer {
    let configured_model_directory =
        crate::support::configured_installed_model_directory_by_id(model_id());
    launch_serving_rest_server_for_model(model_id(), configured_model_directory, None, None).await
}
pub(crate) async fn get_endpoint(server_address: SocketAddr, endpoint_path: &str) -> String {
    send_http_request(
        server_address,
        format!(
            "GET {endpoint_path} HTTP/1.1\r\nHost: {server_address}\r\nConnection: close\r\n\r\n"
        ),
    )
    .await
}

pub(crate) async fn post_chat_completion(
    server_address: SocketAddr,
    request_body: String,
) -> String {
    let request_text = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {server_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request_body}",
        request_body.len()
    );
    send_http_request(server_address, request_text).await
}

pub(crate) async fn post_responses_completion(
    server_address: SocketAddr,
    request_body: String,
) -> String {
    let request_text = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: {server_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request_body}",
        request_body.len()
    );
    send_http_request(server_address, request_text).await
}

pub(crate) fn text_chat_request_body() -> String {
    text_chat_request_body_for_model(model_id())
}

pub(crate) fn text_chat_request_body_for_model(model_id: &str) -> String {
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": "Reply with exactly ASTRONOMICAL_E2E_OK and nothing else.",
        }],
        "stream": true,
        "temperature": 1,
        "max_tokens": 32,
    })
    .to_string()
}

fn deployment_litmus_chat_request_body(model_id: &str, user_prompt: &str) -> String {
    let production_shaped_tools = (0..67)
        .map(|tool_number| {
            json!({
                "type": "function",
                "function": {
                    "name": format!("deployment_litmus_tool_{tool_number}"),
                    "description": "A deployment litmus tool that must not be called.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false,
                    },
                },
            })
        })
        .collect::<Vec<_>>();
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": user_prompt,
        }],
        "tools": production_shaped_tools,
        "stream": true,
        "temperature": 1,
        "max_tokens": DEPLOYMENT_LITMUS_MAX_OUTPUT_TOKENS,
    })
    .to_string()
}

fn deployment_litmus_responses_request_body(model_id: &str, user_prompt: &str) -> String {
    json!({
        "model": model_id,
        "instructions": "Reply with exactly OK and nothing else. Do not provide reasoning or explanation.",
        "input": user_prompt,
        "stream": true,
        "max_output_tokens": DEPLOYMENT_LITMUS_MAX_OUTPUT_TOKENS,
    })
    .to_string()
}

pub(crate) fn image_chat_request_body_for_model(model_id: &str) -> String {
    let red_pixel_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Name the dominant color in this image in one word."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:image/png;base64,{red_pixel_png_base64}"),
                    },
                },
            ],
        }],
        "stream": true,
        "temperature": 1,
        "max_tokens": 128,
    })
    .to_string()
}

pub(crate) fn assert_successful_streaming_chat_response(chat_response: &str) {
    assert!(
        chat_response.starts_with("HTTP/1.1 200 OK"),
        "unexpected HTTP response: {chat_response}"
    );
    assert!(
        chat_response.contains(r#""delta":{"content":"#)
            || chat_response.contains(r#""delta":{"reasoning_content":"#),
        "real stream did not contain model-generated content: {chat_response}"
    );
    assert!(
        chat_response.contains(r#""finish_reason":"length""#)
            || chat_response.contains(r#""finish_reason":"stop""#),
        "real stream did not contain a terminal completion reason: {chat_response}"
    );
    assert!(
        chat_response.contains("data: [DONE]"),
        "real stream did not finish cleanly: {chat_response}"
    );
}

pub(crate) fn assert_successful_streaming_responses_response(responses_response: &str) {
    assert!(
        responses_response.starts_with("HTTP/1.1 200 OK"),
        "unexpected Responses response: {responses_response}"
    );
    assert!(
        responses_response.contains("event: response.output_text.delta")
            || responses_response.contains("event: response.reasoning_summary_text.delta"),
        "real Responses stream did not contain model-generated content: {responses_response}"
    );
    assert!(
        responses_response.contains("event: response.completed")
            || responses_response.contains("event: response.incomplete"),
        "real Responses stream did not finish cleanly: {responses_response}"
    );
}
