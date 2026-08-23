use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::Duration,
};

use astronomical_supervisor::{
    GenerationPerformanceLog, ResolvedRuntimeConfigResolver, WorkerHandle,
    build_application_with_discovered_models, build_development_application_with_reload,
};
use serde_json::json;
use tokio::{
    net::TcpListener,
    sync::oneshot,
    task::JoinHandle,
    time::{sleep, timeout},
};

use crate::common::discovered_model_artifact;

use super::deployment_litmus_model::configured_deployment_litmus_model;
use super::model_artifact_rest_transport::{
    assert_streamed_model_text_mentions_red, send_http_request,
    streamed_model_text_from_chat_response,
};

pub(crate) const E2E_TIMEOUT: Duration = Duration::from_secs(115);
const MODEL_ID: &str = crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID;
const READY_ATTEMPT_LIMIT: u8 = 70;
// The litmus checks stream completion and worker reuse, not long output volume.
const DEPLOYMENT_LITMUS_MAX_OUTPUT_TOKENS: u32 = 512;
pub(super) const DEPLOYMENT_LITMUS_PROMPT: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

pub(crate) struct ModelArtifactRestServer {
    worker_handle: WorkerHandle,
    pub(crate) server_address: SocketAddr,
    shutdown_sender: oneshot::Sender<()>,
    server_task: JoinHandle<Result<(), std::io::Error>>,
    isolated_development_home: Option<tempfile::TempDir>,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST surface and real worker for a repeated-request deployment litmus"]
async fn should_keep_the_deployed_rest_surface_healthy_across_model_artifact_prompt_reuse() {
    timeout(E2E_TIMEOUT, run_deployed_rest_surface_litmus())
        .await
        .expect("the deployed REST litmus must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the complete local Ornith model through MLX"]
async fn should_stream_model_artifact_output_through_the_openai_endpoint() {
    timeout(
        E2E_TIMEOUT,
        run_model_artifact_request_e2e("text", text_chat_request_body()),
    )
    .await
    .expect("the model-artifact E2E test must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the complete local Ornith model and runs one image prompt through MLX"]
async fn should_stream_model_artifact_image_output_through_the_openai_endpoint() {
    timeout(
        E2E_TIMEOUT,
        run_model_artifact_request_e2e("image", image_chat_request_body()),
    )
    .await
    .expect("the model-artifact image E2E test must finish within 115 seconds");
}

async fn run_model_artifact_request_e2e(request_kind: &str, request_body: String) {
    let chat_response =
        run_model_artifact_request_and_return_response_e2e(request_kind, request_body).await;
    assert_successful_streaming_chat_response(&chat_response);
    eprintln!("[e2e] model artifact {request_kind} output streamed and the worker was reaped");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the complete local Ornith model and checks semantic image recognition"]
async fn should_semantically_identify_the_synthetic_red_fixture_through_the_openai_endpoint() {
    let chat_response = timeout(
        E2E_TIMEOUT,
        run_model_artifact_request_and_return_response_e2e(
            "synthetic-red-image",
            image_chat_request_body(),
        ),
    )
    .await
    .expect("the synthetic-red image E2E test must finish within 115 seconds");

    assert_successful_streaming_chat_response(&chat_response);
    let streamed_model_text = streamed_model_text_from_chat_response(&chat_response);
    let matched_red_term = assert_streamed_model_text_mentions_red(&streamed_model_text);
    eprintln!("[e2e] synthetic-red semantic match term={matched_red_term}");
}

async fn run_model_artifact_request_and_return_response_e2e(
    request_kind: &str,
    request_body: String,
) -> String {
    let model_artifact_rest_server = launch_model_artifact_rest_server().await;
    run_model_artifact_request_and_return_response_with_server(
        model_artifact_rest_server,
        request_kind,
        request_body,
    )
    .await
}

pub(super) async fn run_model_artifact_request_e2e_for_model(
    model_id: &str,
    model_directory: std::path::PathBuf,
    request_kind: &str,
    request_body: String,
) {
    let model_artifact_rest_server =
        launch_model_artifact_rest_server_for_model(model_id, model_directory, None, None).await;
    let chat_response = run_model_artifact_request_and_return_response_with_server(
        model_artifact_rest_server,
        request_kind,
        request_body,
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);
    eprintln!("[e2e] {model_id} {request_kind} output streamed and the worker was reaped");
}

async fn run_model_artifact_request_and_return_response_with_server(
    model_artifact_rest_server: ModelArtifactRestServer,
    request_kind: &str,
    request_body: String,
) -> String {
    eprintln!("[e2e] sending one model-artifact OpenAI-compatible {request_kind} chat request");
    let chat_response =
        post_chat_completion(model_artifact_rest_server.server_address, request_body).await;

    stop_model_artifact_rest_server(model_artifact_rest_server).await;

    chat_response
}

async fn run_deployed_rest_surface_litmus() {
    let selected_deployment_litmus_model = configured_deployment_litmus_model();
    let deployment_litmus_model_id = selected_deployment_litmus_model.model_id;
    let model_artifact_rest_server = launch_model_artifact_rest_server_for_model(
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

    stop_model_artifact_rest_server(model_artifact_rest_server).await;
}

async fn launch_model_artifact_rest_server() -> ModelArtifactRestServer {
    let configured_model_directory =
        crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    launch_model_artifact_rest_server_for_model(MODEL_ID, configured_model_directory, None, None)
        .await
}

pub(crate) async fn launch_model_artifact_rest_server_for_model(
    model_id: &str,
    model_directory: PathBuf,
    isolated_worker_home_directory: Option<&Path>,
    performance_log_directory: Option<&Path>,
) -> ModelArtifactRestServer {
    launch_model_artifact_rest_server_for_model_with_memory_limit(
        model_id,
        model_directory,
        isolated_worker_home_directory,
        performance_log_directory,
        None,
    )
    .await
}

pub(crate) async fn launch_model_artifact_rest_server_for_model_with_memory_limit(
    model_id: &str,
    model_directory: PathBuf,
    isolated_worker_home_directory: Option<&Path>,
    performance_log_directory: Option<&Path>,
    maximum_mlx_memory_bytes: Option<u64>,
) -> ModelArtifactRestServer {
    let production_worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production inference-worker executable path"),
    );
    let owned_isolated_development_home = isolated_worker_home_directory
        .is_none()
        .then(crate::common::isolated_development_home_from_user_config);
    let worker_configuration_home_directory = isolated_worker_home_directory
        .map(Path::to_path_buf)
        .or_else(|| {
            owned_isolated_development_home
                .as_ref()
                .map(|isolated_home| isolated_home.path().to_path_buf())
        })
        .expect("qualification should own or receive an isolated Development home");
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        worker_configuration_home_directory,
        production_worker_executable_path.clone(),
    );
    let worker_runtime_config = runtime_config_resolver
        .load()
        .expect("the deployment litmus worker configuration should resolve");
    let deployment_litmus_log_directory =
        tempfile::tempdir().expect("the deployment litmus log directory should be created");
    let performance_log_directory =
        performance_log_directory.unwrap_or_else(|| deployment_litmus_log_directory.path());
    let mut worker_startup_configuration = worker_runtime_config.worker_startup_configuration();
    if let Some(maximum_mlx_memory_bytes) = maximum_mlx_memory_bytes {
        // The public configuration intentionally accepts whole decimal GB. Qualification also
        // needs half-GB cells, so inject the exact byte ceiling through the production typed
        // worker-startup protocol after normal configuration resolution. The server, subprocess,
        // model load, and HTTP boundary remain the production path; only this test input is exact.
        worker_startup_configuration.configured_maximum_mlx_memory_bytes =
            Some(maximum_mlx_memory_bytes);
    }
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        &production_worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(performance_log_directory)
            .expect("the deployment litmus performance log should open"),
        Arc::clone(&worker_runtime_config.model_policy_catalog),
        worker_startup_configuration,
    )
    .await
    .expect("the production worker should launch for the deployment litmus");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the deployment litmus REST listener should bind");
    let server_address = listener
        .local_addr()
        .expect("the deployment litmus REST listener should expose its address");
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let application = match isolated_worker_home_directory {
        Some(isolated_worker_home_directory) => {
            let runtime_config_resolver =
                ResolvedRuntimeConfigResolver::for_development_home_directory(
                    isolated_worker_home_directory.to_path_buf(),
                    production_worker_executable_path,
                );
            let resolved_runtime_config = runtime_config_resolver
                .load()
                .expect("the isolated model-artifact configuration should resolve");
            build_development_application_with_reload(
                worker_handle.clone(),
                Arc::new(RwLock::new(resolved_runtime_config)),
                isolated_worker_home_directory.to_path_buf(),
            )
        }
        None => build_application_with_discovered_models(
            worker_handle.clone(),
            vec![discovered_model_artifact(
                model_id,
                &model_directory,
                20_480,
            )],
        ),
    };
    let server = axum::serve(listener, application).with_graceful_shutdown(async {
        let _ = shutdown_receiver.await;
    });
    let server_task = tokio::spawn(async move { server.await });

    wait_until_ready(server_address).await;
    ModelArtifactRestServer {
        worker_handle,
        server_address,
        shutdown_sender,
        server_task,
        isolated_development_home: owned_isolated_development_home,
    }
}

pub(crate) async fn stop_model_artifact_rest_server(
    model_artifact_rest_server: ModelArtifactRestServer,
) {
    let ModelArtifactRestServer {
        worker_handle,
        shutdown_sender,
        server_task,
        isolated_development_home,
        ..
    } = model_artifact_rest_server;
    let _ = shutdown_sender.send(());
    server_task
        .await
        .expect("the model-artifact REST server task should not panic")
        .expect("the model-artifact REST server should stop cleanly");
    worker_handle
        .shutdown()
        .await
        .expect("the model-artifact worker should terminate and be reaped");
    drop(isolated_development_home);
}

async fn wait_until_ready(server_address: SocketAddr) {
    for readiness_attempt in 1..=READY_ATTEMPT_LIMIT {
        let readiness_response = get_endpoint(server_address, "/ready").await;
        if readiness_response.starts_with("HTTP/1.1 200 OK") {
            eprintln!("[e2e] model worker ready after {readiness_attempt} attempts");
            return;
        }
        let remaining_seconds = u16::from(READY_ATTEMPT_LIMIT - readiness_attempt);
        eprintln!(
            "[e2e] loading attempt {readiness_attempt}/{READY_ATTEMPT_LIMIT}, ETA <= {remaining_seconds}s"
        );
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the model-artifact worker did not become ready before the E2E deadline");
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

pub(super) async fn post_responses_completion(
    server_address: SocketAddr,
    request_body: String,
) -> String {
    let request_text = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: {server_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request_body}",
        request_body.len()
    );
    send_http_request(server_address, request_text).await
}

fn text_chat_request_body() -> String {
    text_chat_request_body_for_model(MODEL_ID)
}

pub(super) fn text_chat_request_body_for_model(model_id: &str) -> String {
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": "Reply with exactly ASTRONOMICAL_E2E_OK and nothing else.",
        }],
        "stream": true,
        "temperature": 0,
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
        "temperature": 0,
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

fn image_chat_request_body() -> String {
    image_chat_request_body_for_model(MODEL_ID)
}

pub(super) fn image_chat_request_body_for_model(model_id: &str) -> String {
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
        "temperature": 0,
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

pub(super) fn assert_successful_streaming_responses_response(responses_response: &str) {
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
