use std::{future::Future, pin::Pin};

use astronomical_daemon::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerHealthSnapshot,
    build_application,
};
use astronomical_ipc_protocol::ChatGenerationCommand;
use astronomical_rest_contract::MAX_OPENAI_REQUEST_BODY_BYTES;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
    response::Response,
};
use serde_json::Value;
use tokio::sync::mpsc;
use tower::ServiceExt;

fn oversized_body_byte_count() -> usize {
    MAX_OPENAI_REQUEST_BODY_BYTES + 1
}

async fn post_chat_body(
    application: Router,
    body_bytes: Vec<u8>,
    content_length: Option<&str>,
) -> Response {
    let mut chat_request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(content_length) = content_length {
        chat_request = chat_request.header(header::CONTENT_LENGTH, content_length);
    }
    application
        .oneshot(
            chat_request
                .body(Body::from(body_bytes))
                .expect("the chat request should be well formed"),
        )
        .await
        .expect("the in-process application should return a chat response")
}

async fn response_error_json(response: Response) -> Value {
    let response_body = to_bytes(response.into_body(), 8 * 1024)
        .await
        .expect("the bounded JSON error body should be readable");
    serde_json::from_slice(&response_body).expect("the error body should be OpenAI JSON")
}

#[tokio::test]
async fn should_return_openai_shaped_413_when_a_chat_body_exceeds_the_request_budget() {
    let application = build_application(PanicOnGenerationStartExecutor);
    let http_response =
        post_chat_body(application, vec![b'x'; oversized_body_byte_count()], None).await;

    assert_eq!(http_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let error_json = response_error_json(http_response).await;
    assert_eq!(error_json["error"]["code"], "request_too_large");
    let rejection_message = error_json["error"]["message"].as_str().unwrap();
    assert!(rejection_message.contains("32000000-byte request budget"));
    assert!(rejection_message.contains("by 1 byte"));
    assert!(rejection_message.contains(
        "mirrors the 32 MB standard-endpoint request size limit of the Anthropic Messages API"
    ));
}

#[tokio::test]
async fn should_reject_a_chat_body_declared_over_the_budget_before_reading_it() {
    let application = build_application(PanicOnGenerationStartExecutor);
    // The tiny body lies about its size the way a hostile client would; the
    // declared Content-Length must trigger the typed rejection without any
    // body read.
    let http_response = post_chat_body(
        application,
        b"{}".to_vec(),
        Some(&oversized_body_byte_count().to_string()),
    )
    .await;

    assert_eq!(http_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let error_json = response_error_json(http_response).await;
    assert_eq!(error_json["error"]["code"], "request_too_large");
}

#[tokio::test]
async fn should_accept_a_chat_body_at_exactly_the_request_budget() {
    let application = build_application(PanicOnGenerationStartExecutor);
    // A body at exactly the budget passes the transport layer; the endpoint
    // then rejects the payload on its JSON merits, not its size.
    let http_response =
        post_chat_body(application, vec![b'x'; MAX_OPENAI_REQUEST_BODY_BYTES], None).await;

    assert_eq!(http_response.status(), StatusCode::BAD_REQUEST);
    let error_json = response_error_json(http_response).await;
    assert_eq!(error_json["error"]["code"], "invalid_json");
}

#[tokio::test]
async fn should_return_openai_shaped_413_when_an_embeddings_body_exceeds_the_request_budget() {
    let application = build_application(PanicOnGenerationStartExecutor);
    let embeddings_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .header(
                    header::CONTENT_LENGTH,
                    oversized_body_byte_count().to_string(),
                )
                .body(Body::from(b"{}".to_vec()))
                .expect("the embeddings request should be well formed"),
        )
        .await
        .expect("the in-process application should return an embeddings response");

    assert_eq!(embeddings_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let error_json = response_error_json(embeddings_response).await;
    assert_eq!(error_json["error"]["code"], "request_too_large");
}

#[tokio::test]
async fn should_return_413_without_exact_overshoot_when_a_chat_body_streams_far_past_the_budget() {
    let application = build_application(PanicOnGenerationStartExecutor);
    // No Content-Length header: the bounded buffer stops one byte past the
    // budget, so a body well beyond it takes the no-exact-overshoot rejection.
    let http_response = post_chat_body(
        application,
        vec![b'x'; MAX_OPENAI_REQUEST_BODY_BYTES + 1000],
        None,
    )
    .await;

    assert_eq!(http_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let error_json = response_error_json(http_response).await;
    assert_eq!(error_json["error"]["code"], "request_too_large");
    let rejection_message = error_json["error"]["message"].as_str().unwrap();
    assert!(rejection_message.contains("32000000-byte request budget"));
    assert!(rejection_message.contains("exceeds"));
    assert!(!rejection_message.contains("by 1 byte"));
}

#[tokio::test]
async fn should_treat_an_unparseable_content_length_as_absent() {
    let application = build_application(PanicOnGenerationStartExecutor);
    let http_response = post_chat_body(application, b"{}".to_vec(), Some("not-a-number")).await;

    assert_eq!(http_response.status(), StatusCode::BAD_REQUEST);
    let error_json = response_error_json(http_response).await;
    assert_eq!(error_json["error"]["code"], "invalid_json");
}

/// Any generation admission proves the body budget failed to hold, so the
/// executor panics instead of silently admitting an oversized request.
struct PanicOnGenerationStartExecutor;

impl ChatGenerationExecutor for PanicOnGenerationStartExecutor {
    fn start_chat_generation(
        &self,
        _chat_generation_command: ChatGenerationCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<ChatGenerationStreamEvent>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async { panic!("generation must not be admitted for an over-budget body") })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        WorkerHealthSnapshot::ready_without_model(0)
    }
}

impl astronomical_daemon::ImageGenerationExecutor for PanicOnGenerationStartExecutor {}
