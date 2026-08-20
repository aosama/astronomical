use std::{future::Future, pin::Pin};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationFailureReason,
    ChatModelCapabilities, MtpRuntimeState,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamErrorCode, ChatGenerationStreamEvent,
    GenerationStartError, WorkerHealthSnapshot, build_application,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use tokio::sync::mpsc;
use tower::ServiceExt;

const MODEL_ID: &str = "astronomical/non-streaming-test-model";

#[tokio::test]
async fn should_return_a_non_streaming_chat_completion_json_response() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::TextFragment("done".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 3,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let (status, content_type, response_body) = post_non_streaming_chat(application).await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type
            .as_deref()
            .unwrap_or("")
            .starts_with("application/json"),
        "the non-streaming response should be JSON, got content-type: {content_type:?}"
    );
    assert!(
        !response_body.contains("data: "),
        "the non-streaming response must not contain SSE frames, got: {response_body}"
    );
    assert!(
        !response_body.contains("[DONE]"),
        "the non-streaming response must not contain the SSE terminator, got: {response_body}"
    );

    let response_document: serde_json::Value =
        serde_json::from_str(&response_body).expect("the non-streaming body should be JSON");
    assert_eq!(response_document["object"], "chat.completion");
    assert_eq!(
        response_document["choices"][0]["message"]["content"],
        "done"
    );
    assert_eq!(response_document["choices"][0]["finish_reason"], "stop");
    assert_eq!(response_document["usage"]["prompt_tokens"], 3);
    assert_eq!(response_document["usage"]["completion_tokens"], 2);
    assert_eq!(response_document["usage"]["total_tokens"], 5);
}

#[tokio::test]
async fn should_accept_a_chat_request_with_the_stream_field_omitted() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::TextFragment("done".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 3,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let (status, _content_type, response_body) = post_chat_with_body(
        application,
        r#"{"model":"astronomical/non-streaming-test-model","messages":[{"role":"user","content":"hello"}]}"#,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_body).expect("the omitted-stream body should be JSON");
    assert_eq!(response_document["object"], "chat.completion");
    assert_eq!(
        response_document["choices"][0]["message"]["content"],
        "done"
    );
    assert_eq!(response_document["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn should_assemble_reasoning_and_text_in_a_non_streaming_response() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::ReasoningFragment("inspect first".to_owned()),
        ChatGenerationStreamEvent::TextFragment("done".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 3,
            generated_token_count: 2,
            reasoning_token_count: 1,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let (status, _content_type, response_body) = post_non_streaming_chat(application).await;

    assert_eq!(status, StatusCode::OK);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_body).expect("the reasoning+text body should be JSON");
    assert_eq!(
        response_document["choices"][0]["message"]["content"],
        "done"
    );
    assert_eq!(
        response_document["choices"][0]["message"]["reasoning_content"],
        "inspect first"
    );
    assert_eq!(response_document["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn should_assemble_tool_calls_in_a_non_streaming_response() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"filePath":"README.md"}"#.to_owned(),
        },
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 1,
            generated_token_count: 1,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::ToolCalls,
        },
    ]));

    let (status, _content_type, response_body) = post_non_streaming_chat(application).await;

    assert_eq!(status, StatusCode::OK);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_body).expect("the tool-call body should be JSON");
    assert_eq!(
        response_document["choices"][0]["finish_reason"],
        "tool_calls"
    );
    let tool_call = &response_document["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(tool_call["function"]["name"], "read");
    assert_eq!(
        tool_call["function"]["arguments"],
        r#"{"filePath":"README.md"}"#
    );
    let tool_call_id = tool_call["id"]
        .as_str()
        .expect("the tool call should have an id");
    assert!(
        tool_call_id.starts_with("call_"),
        "the tool call id should follow the streaming-path format, got: {tool_call_id}"
    );
}

#[tokio::test]
async fn should_include_cached_tokens_in_non_streaming_usage_when_nonzero() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 4_096,
            generated_token_count: 100,
            reasoning_token_count: 0,
            cached_token_count: 2_048,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let (status, _content_type, response_body) = post_non_streaming_chat(application).await;

    assert_eq!(status, StatusCode::OK);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_body).expect("the cached-token body should be JSON");
    assert_eq!(
        response_document["usage"]["prompt_tokens_details"]["cached_tokens"],
        2_048
    );
}

#[tokio::test]
async fn should_return_a_service_unavailable_error_body_for_a_failed_non_streaming_chat() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::InvalidRequest {
                reason: "rendered prompt exceeds the 262144-byte worker limit".to_owned(),
            },
        },
    ]));

    let (status, _content_type, response_body) = post_non_streaming_chat(application).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_body).expect("the failed-chat body should be JSON");
    assert_eq!(response_document["error"]["code"], "chat_invalid_request");
    assert_eq!(
        response_document["error"]["message"],
        "the local worker rejected the chat request: rendered prompt exceeds the 262144-byte worker limit"
    );
}

#[tokio::test]
async fn should_return_a_service_unavailable_error_body_when_the_worker_becomes_unavailable_mid_chat()
 {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable),
    ]));

    let (status, _content_type, response_body) = post_non_streaming_chat(application).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_body).expect("the worker-unavailable body should be JSON");
    assert_eq!(
        response_document["error"]["code"],
        "chat_worker_unavailable"
    );
    assert_eq!(
        response_document["error"]["message"],
        "the local worker became unavailable while processing the chat request"
    );
}

#[tokio::test]
async fn should_report_fatal_worker_execution_as_unavailable_for_a_non_streaming_chat() {
    let bounded_fatal_execution_reason =
        "GPU allocation exceeded the platform buffer limit; reduce the prompt size";
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::FatalExecution {
                reason: bounded_fatal_execution_reason.to_owned(),
            },
        },
    ]));

    let (status, _content_type, response_body) = post_non_streaming_chat(application).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_body).expect("the fatal-execution body should be JSON");
    assert_eq!(
        response_document["error"]["code"],
        "chat_worker_unavailable"
    );
    assert_eq!(
        response_document["error"]["message"],
        format!(
            "the local worker stopped after a fatal model execution error: {bounded_fatal_execution_reason}"
        )
    );
}

#[tokio::test]
async fn should_return_context_length_exceeded_for_a_non_streaming_chat() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: 262_145,
                maximum_context_tokens: 262_144,
            },
        },
    ]));

    let (status, _content_type, response_body) = post_non_streaming_chat(application).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_body).expect("the context error should be JSON");
    assert_eq!(
        response_document["error"]["code"],
        "context_length_exceeded"
    );
    assert_eq!(response_document["error"]["param"], "messages");
}

async fn post_non_streaming_chat(
    application: axum::Router,
) -> (StatusCode, Option<String>, String) {
    post_chat_with_body(
        application,
        r#"{"model":"astronomical/non-streaming-test-model","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
    )
    .await
}

async fn post_chat_with_body(
    application: axum::Router,
    request_body: &'static str,
) -> (StatusCode, Option<String>, String) {
    let chat_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request_body))
                .expect("the chat request should be valid"),
        )
        .await
        .expect("the application should return a chat response");
    let status = chat_response.status();
    let content_type = chat_response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|header_value| header_value.to_str().ok())
        .map(str::to_owned);
    let response_body = to_bytes(chat_response.into_body(), 16 * 1024)
        .await
        .expect("the chat response body should be readable");
    (
        status,
        content_type,
        String::from_utf8(response_body.to_vec())
            .expect("the chat response body should contain UTF-8"),
    )
}

struct ScriptedExecutor {
    health_snapshot: WorkerHealthSnapshot,
    stream_events: Vec<ChatGenerationStreamEvent>,
}

impl ScriptedExecutor {
    fn ready(stream_events: Vec<ChatGenerationStreamEvent>) -> Self {
        Self {
            health_snapshot: WorkerHealthSnapshot::ready_with_model(
                MODEL_ID.to_owned(),
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
            ),
            stream_events,
        }
    }
}

impl ChatGenerationExecutor for ScriptedExecutor {
    fn start_chat_generation(
        &self,
        _generation_command: ChatGenerationCommand,
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
            let (stream_event_sender, stream_event_receiver) =
                mpsc::channel(self.stream_events.len().max(1));
            for stream_event in &self.stream_events {
                stream_event_sender
                    .send(stream_event.clone())
                    .await
                    .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            }
            Ok(stream_event_receiver)
        })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        self.health_snapshot.clone()
    }
}

impl astronomical_supervisor::ImageGenerationExecutor for ScriptedExecutor {}
