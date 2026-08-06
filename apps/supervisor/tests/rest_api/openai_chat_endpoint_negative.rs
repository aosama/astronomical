use std::{future::Future, path::PathBuf, pin::Pin};

use astronomical_config::DiscoveredModel;
use astronomical_ipc_protocol::{ChatGenerationCommand, ChatModelCapabilities, MtpRuntimeState};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerHealthSnapshot,
    build_application, build_application_with_config_warning_and_discovered_models,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use tokio::sync::mpsc;
use tower::ServiceExt;

const READY_MODEL_ID: &str = "astronomical/negative-chat-test-worker";

#[tokio::test]
async fn should_reject_openai_chat_model_mismatch_before_worker_admission() {
    let response_text = post_chat_and_read_body(
        PanicOnChatStartExecutor,
        r#"{"model":"astronomical/not-loaded","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
        StatusCode::BAD_REQUEST,
    )
    .await;

    assert!(response_text.contains(r#""param":"model""#));
    assert!(response_text.contains(r#""code":"model_not_found""#));
}

#[tokio::test]
async fn should_return_too_many_requests_when_openai_chat_capacity_is_unavailable() {
    let response_text = post_chat_and_read_body(
        ReadyChatFailureExecutor {
            chat_start_error: GenerationStartError::CapacityUnavailable,
        },
        r#"{"model":"astronomical/negative-chat-test-worker","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
        StatusCode::TOO_MANY_REQUESTS,
    )
    .await;

    assert!(response_text.contains(r#""code":"server_capacity""#));
    assert!(response_text.contains(r#""message":"the generation queue is full""#));
}

#[tokio::test]
async fn should_return_service_unavailable_when_openai_chat_worker_is_unavailable() {
    let response_text = post_chat_and_read_body(
        ReadyChatFailureExecutor {
            chat_start_error: GenerationStartError::WorkerUnavailable,
        },
        r#"{"model":"astronomical/negative-chat-test-worker","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
        StatusCode::SERVICE_UNAVAILABLE,
    )
    .await;

    assert!(response_text.contains(r#""code":"worker_unavailable""#));
    assert!(response_text.contains(r#""message":"the local worker is unavailable""#));
}

#[tokio::test]
async fn should_return_payload_too_large_when_openai_chat_exceeds_the_ipc_frame() {
    let response_text = post_chat_and_read_body(
        ReadyChatFailureExecutor {
            chat_start_error: GenerationStartError::RequestTooLarge {
                actual_ipc_message_bytes: 33_554_433,
                maximum_ipc_message_bytes: 33_554_432,
            },
        },
        r#"{"model":"astronomical/negative-chat-test-worker","messages":[{"role":"user","content":"describe the image"}],"stream":true}"#,
        StatusCode::PAYLOAD_TOO_LARGE,
    )
    .await;

    assert!(response_text.contains(r#""code":"request_too_large""#));
    assert!(response_text.contains("reduce image sizes or conversation history"));
}

#[tokio::test]
async fn should_explain_why_the_requested_chat_model_could_not_be_loaded() {
    let response_text = post_chat_and_read_body(
        ReadyChatFailureExecutor {
            chat_start_error: GenerationStartError::ModelLoadFailed {
                model_load_failure_reason: "OptiQ metadata uses unsupported 2-bit quantization"
                    .to_owned(),
            },
        },
        r#"{"model":"astronomical/negative-chat-test-worker","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
        StatusCode::SERVICE_UNAVAILABLE,
    )
    .await;

    assert!(response_text.contains(r#""code":"model_load_failed""#));
    assert!(response_text.contains(
        r#""message":"the requested model could not be loaded: OptiQ metadata uses unsupported 2-bit quantization""#
    ));
}

#[tokio::test]
async fn should_canonicalize_a_provider_prefixed_model_for_an_idle_worker() {
    let application = build_application_with_config_warning_and_discovered_models(
        IdleWorkerCapacityExecutor,
        None,
        vec![discovered_model()],
    );
    let chat_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"mlx-community/requested-model","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
                ))
                .expect("the chat request should be well formed"),
        )
        .await
        .expect("the in-process application should return a chat response");

    assert_eq!(chat_response.status(), StatusCode::TOO_MANY_REQUESTS);
}

async fn post_chat_and_read_body<GenerationExecutor>(
    generation_executor: GenerationExecutor,
    request_body: &'static str,
    expected_status: StatusCode,
) -> String
where
    GenerationExecutor: ChatGenerationExecutor,
{
    let application = build_application(generation_executor);
    let chat_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request_body))
                .expect("the chat request should be well formed"),
        )
        .await
        .expect("the in-process application should return a chat response");

    assert_eq!(chat_response.status(), expected_status);
    let response_body = to_bytes(chat_response.into_body(), 8 * 1024)
        .await
        .expect("the bounded JSON response body should be readable");
    String::from_utf8(response_body.to_vec()).expect("the JSON response body should contain UTF-8")
}

struct PanicOnChatStartExecutor;

struct ReadyChatFailureExecutor {
    chat_start_error: GenerationStartError,
}

struct IdleWorkerCapacityExecutor;

impl ChatGenerationExecutor for PanicOnChatStartExecutor {
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
        Box::pin(async { panic!("chat generation must not be admitted for this request") })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        ready_worker_health_snapshot()
    }
}

impl ChatGenerationExecutor for ReadyChatFailureExecutor {
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
        Box::pin(async move { Err(self.chat_start_error.clone()) })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        ready_worker_health_snapshot()
    }
}

impl ChatGenerationExecutor for IdleWorkerCapacityExecutor {
    fn start_chat_generation(
        &self,
        chat_generation_command: ChatGenerationCommand,
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
        Box::pin(async move {
            assert_eq!(chat_generation_command.model, "requested-model");
            Err(GenerationStartError::CapacityUnavailable)
        })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        WorkerHealthSnapshot::ready_without_model(0)
    }
}

fn ready_worker_health_snapshot() -> WorkerHealthSnapshot {
    WorkerHealthSnapshot::ready_with_model(
        READY_MODEL_ID.to_owned(),
        ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: true,
            max_input_tokens: 241_664,
            max_output_tokens: 20_480,
            context_window: 262_144,
        },
        MtpRuntimeState::Disabled,
        None,
    )
}

fn discovered_model() -> DiscoveredModel {
    DiscoveredModel {
        model_id: "requested-model".to_owned(),
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "test-revision".to_owned(),
        model_directory: PathBuf::from("/models/requested-model"),
        context_window: 2_048,
        max_input_tokens: 1_024,
        max_output_tokens: 128,
        has_vision: false,
        model_size_bytes: 0,
    }
}
