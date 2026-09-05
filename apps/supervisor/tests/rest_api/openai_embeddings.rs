//! Acceptance coverage for the native OpenAI embeddings HTTP journey.

use std::path::PathBuf;

use astronomical_config::{
    DiscoveredModel, EmbeddingModelCapabilities, ModelCapabilities, ModelFamily, ModelLicense,
};
use astronomical_supervisor::{EmbeddingsExecutionError, build_application_with_discovered_models};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

const EMBEDDING_MODEL_ID: &str = "nomicai-modernbert-embed-base-8bit";
const ROMEO_LINE: &str = "O Romeo, Romeo, wherefore art thou Romeo?";

#[tokio::test]
async fn should_embed_romeo_and_juliet_through_the_public_http_journey() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_embeddings_commands();
    let application = build_application_with_discovered_models(executor, vec![embedding_model()]);

    let response = application
        .oneshot(embeddings_request(valid_request_document()))
        .await
        .expect("the application should return an embeddings response");

    assert_eq!(response.status(), StatusCode::OK);
    let response_document = response_json(response).await;
    assert_eq!(response_document["object"], "list");
    assert_eq!(response_document["model"], EMBEDDING_MODEL_ID);
    assert_eq!(response_document["data"][0]["object"], "embedding");
    assert_eq!(
        response_document["data"][0]["embedding"]
            .as_array()
            .map(Vec::len),
        Some(768)
    );
    assert_eq!(response_document["usage"]["prompt_tokens"], 8);
    assert_eq!(response_document["usage"]["completion_tokens"], 0);

    let commands = received_commands
        .lock()
        .expect("the embeddings command log should remain available");
    assert_eq!(commands.len(), 1);
    assert_eq!(commands[0].model, EMBEDDING_MODEL_ID);
    assert_eq!(commands[0].inputs, vec![ROMEO_LINE.to_owned()]);
}

#[tokio::test]
async fn should_reject_a_chat_model_before_embeddings_dispatch() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_embeddings_commands();
    let application = build_application_with_discovered_models(executor, vec![chat_model()]);

    let response = application
        .oneshot(embeddings_request(serde_json::json!({
            "model": "chat-only-model",
            "input": ROMEO_LINE
        })))
        .await
        .expect("the application should reject a chat model");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document = response_json(response).await;
    assert_eq!(
        response_document["error"]["code"],
        "model_capability_mismatch"
    );
    assert_eq!(response_document["error"]["param"], "model");
    assert!(
        received_commands
            .lock()
            .expect("command log should remain available")
            .is_empty()
    );
}

#[tokio::test]
async fn should_reject_empty_input_before_embeddings_dispatch() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_embeddings_commands();
    let application = build_application_with_discovered_models(executor, vec![embedding_model()]);

    let response = application
        .oneshot(embeddings_request(serde_json::json!({
            "model": EMBEDDING_MODEL_ID,
            "input": []
        })))
        .await
        .expect("the application should reject empty input");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document = response_json(response).await;
    assert_eq!(response_document["error"]["param"], "input");
    assert!(
        received_commands
            .lock()
            .expect("command log should remain available")
            .is_empty()
    );
}

#[tokio::test]
async fn should_not_expose_worker_embeddings_failure_reasons_or_local_paths() {
    let fictional_private_path = "/Users/fictional-person/private-model/encoder.safetensors";
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.embeddings_outcome = Err(EmbeddingsExecutionError::WorkerFailure(
        astronomical_ipc_protocol::EmbeddingsFailureReason::FatalExecution {
            reason: format!("native execution failed while mapping {fictional_private_path}"),
        },
    ));
    let application = build_application_with_discovered_models(executor, vec![embedding_model()]);

    let response = application
        .oneshot(embeddings_request(valid_request_document()))
        .await
        .expect("the application should sanitize the worker failure");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let response_document = response_json(response).await;
    assert_eq!(response_document["error"]["code"], "embeddings_failed");
    assert_eq!(
        response_document["error"]["message"],
        "embedding generation failed in the local worker"
    );
    let response_text = response_document.to_string();
    assert!(
        !response_text.contains(fictional_private_path),
        "worker failure must not leak a local path: {response_text}"
    );
}

fn valid_request_document() -> serde_json::Value {
    serde_json::json!({
        "model": EMBEDDING_MODEL_ID,
        "input": ROMEO_LINE
    })
}

fn embeddings_request(request_document: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(request_document.to_string()))
        .expect("the embeddings request should be valid")
}

async fn response_json(response: axum::response::Response) -> serde_json::Value {
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the bounded response body should be readable");
    serde_json::from_slice(&response_body).expect("the response should contain JSON")
}

fn embedding_model() -> DiscoveredModel {
    DiscoveredModel {
        model_id: EMBEDDING_MODEL_ID.to_owned(),
        provider_model_id: Some("mlx-community/nomicai-modernbert-embed-base-8bit".to_owned()),
        model_family: ModelFamily::ModernBert,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/modernbert"),
        capabilities: ModelCapabilities::Embeddings(EmbeddingModelCapabilities {
            vector_width: 768,
            max_input_tokens: 8_192,
        }),
        license: Some(ModelLicense::Apache20),
        model_size_bytes: 1,
    }
}

fn chat_model() -> DiscoveredModel {
    DiscoveredModel {
        model_id: "chat-only-model".to_owned(),
        provider_model_id: None,
        model_family: ModelFamily::Qwen3_5,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/chat-only"),
        capabilities: ModelCapabilities::Chat(astronomical_config::ChatModelCapabilities {
            context_window: 2_048,
            max_input_tokens: 1_024,
            max_output_tokens: 128,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
        model_size_bytes: 1,
    }
}
