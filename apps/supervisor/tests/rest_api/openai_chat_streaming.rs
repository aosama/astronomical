use std::{future::Future, pin::Pin};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatModelCapabilities, MtpRuntimeState,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerHealthSnapshot,
    build_application,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use serde_json::Value;
use tokio::sync::mpsc;
use tower::ServiceExt;

const MODEL_ID: &str = "astronomical/streaming-test-model";

#[tokio::test]
async fn should_stream_openai_compatible_text_lifecycle_for_opencode() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::TextFragment("Visible title".to_owned()),
        completed_generation_event(ChatGenerationCompletionReason::EndOfSequence),
    ]));

    let (status, response_body) = post_chat_with_body(
        application,
        r#"{"model":"astronomical/streaming-test-model","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}"#,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let parsed_stream = ParsedChatSseStream::parse(&response_body);
    assert!(
        parsed_stream.saw_done,
        "OpenAI-compatible chat streams must end with data: [DONE]: {response_body}"
    );
    assert_eq!(parsed_stream.visible_text_for_opencode(), "Visible title");
    assert_eq!(parsed_stream.reasoning_text(), "");
    assert_eq!(parsed_stream.finish_reason(), Some("stop"));
    assert!(
        parsed_stream
            .payloads
            .iter()
            .any(|payload| payload["choices"][0]["delta"]["role"] == "assistant"),
        "chat stream should start with an assistant role chunk: {response_body}"
    );
    let terminal_payload = parsed_stream
        .payloads
        .iter()
        .find(|payload| payload["choices"][0]["finish_reason"] == "stop")
        .expect("stream should contain a terminal finish chunk");
    assert_eq!(terminal_payload["usage"]["prompt_tokens"], 3);
    assert_eq!(terminal_payload["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn should_not_expose_reasoning_only_stream_as_opencode_visible_text() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::ReasoningFragment("thinking only".to_owned()),
        completed_generation_event(ChatGenerationCompletionReason::EndOfSequence),
    ]));

    let (status, response_body) = post_chat_with_body(
        application,
        r#"{"model":"astronomical/streaming-test-model","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let parsed_stream = ParsedChatSseStream::parse(&response_body);
    assert!(
        parsed_stream.saw_done,
        "stream should terminate: {response_body}"
    );
    assert_eq!(parsed_stream.reasoning_text(), "thinking only");
    assert_eq!(
        parsed_stream.visible_text_for_opencode(),
        "",
        "reasoning_content must not be duplicated into delta.content because OpenCode uses content as visible text/title: {response_body}"
    );
    assert_eq!(parsed_stream.finish_reason(), Some("stop"));
}

#[tokio::test]
async fn should_not_fallback_reasoning_only_non_streaming_output_to_message_content() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::ReasoningFragment("thinking only".to_owned()),
        completed_generation_event(ChatGenerationCompletionReason::EndOfSequence),
    ]));

    let (status, response_body) = post_chat_with_body(
        application,
        r#"{"model":"astronomical/streaming-test-model","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let response_document: Value =
        serde_json::from_str(&response_body).expect("the non-streaming body should be JSON");
    assert!(
        response_document["choices"][0]["message"]["content"].is_null(),
        "reasoning-only non-streaming responses must not expose reasoning as visible assistant content: {response_body}"
    );
    assert_eq!(
        response_document["choices"][0]["message"]["reasoning_content"],
        "thinking only"
    );
}

#[tokio::test]
async fn should_expose_only_text_fragment_as_visible_content_when_reasoning_precedes_text() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::ReasoningFragment("internal thought".to_owned()),
        ChatGenerationStreamEvent::TextFragment("Visible title".to_owned()),
        completed_generation_event(ChatGenerationCompletionReason::EndOfSequence),
    ]));

    let (status, response_body) = post_chat_with_body(
        application,
        r#"{"model":"astronomical/streaming-test-model","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let parsed_stream = ParsedChatSseStream::parse(&response_body);
    assert_eq!(parsed_stream.reasoning_text(), "internal thought");
    assert_eq!(
        parsed_stream.visible_text_for_opencode(),
        "Visible title",
        "OpenCode title generation should see only assistant text deltas, not prior reasoning: {response_body}"
    );
    assert_eq!(parsed_stream.finish_reason(), Some("stop"));
}

#[tokio::test]
async fn should_stream_parallel_tool_calls_in_openai_compatible_chunks_for_opencode() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"filePath":"README.md"}"#.to_owned(),
        },
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 1,
            function_name: "glob".to_owned(),
            arguments_json: r#"{"pattern":"tests/**/*.rs"}"#.to_owned(),
        },
        completed_generation_event(ChatGenerationCompletionReason::ToolCalls),
    ]));

    let (status, response_body) = post_chat_with_body(
        application,
        r#"{"model":"astronomical/streaming-test-model","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let parsed_stream = ParsedChatSseStream::parse(&response_body);
    let tool_call_deltas = parsed_stream
        .payloads
        .iter()
        .filter_map(|payload| payload["choices"][0]["delta"]["tool_calls"].as_array())
        .flat_map(|tool_calls| tool_calls.iter())
        .collect::<Vec<_>>();
    assert_eq!(tool_call_deltas.len(), 2, "response body: {response_body}");
    let first_tool_call_delta = tool_call_deltas[0];
    let second_tool_call_delta = tool_call_deltas[1];

    assert_eq!(first_tool_call_delta["index"], 0);
    assert!(
        first_tool_call_delta["id"]
            .as_str()
            .is_some_and(|tool_call_id| tool_call_id.starts_with("call_")),
        "tool call ids should be stable OpenAI-style call ids: {response_body}"
    );
    assert_eq!(first_tool_call_delta["type"], "function");
    assert_eq!(first_tool_call_delta["function"]["name"], "read");
    assert_eq!(
        first_tool_call_delta["function"]["arguments"],
        r#"{"filePath":"README.md"}"#
    );
    assert_eq!(second_tool_call_delta["index"], 1);
    assert_eq!(second_tool_call_delta["function"]["name"], "glob");
    assert_eq!(
        second_tool_call_delta["function"]["arguments"],
        r#"{"pattern":"tests/**/*.rs"}"#
    );
    assert_eq!(parsed_stream.finish_reason(), Some("tool_calls"));
    assert!(
        parsed_stream.saw_done,
        "stream should terminate: {response_body}"
    );
}

fn completed_generation_event(
    completion_reason: ChatGenerationCompletionReason,
) -> ChatGenerationStreamEvent {
    ChatGenerationStreamEvent::Completed {
        prompt_token_count: 3,
        generated_token_count: 2,
        reasoning_token_count: 0,
        cached_token_count: 0,
        reason: completion_reason,
    }
}

async fn post_chat_with_body(
    application: axum::Router,
    request_body: &'static str,
) -> (StatusCode, String) {
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
    let response_body = to_bytes(chat_response.into_body(), 64 * 1024)
        .await
        .expect("the chat response body should be readable");
    (
        status,
        String::from_utf8(response_body.to_vec())
            .expect("the chat response body should contain UTF-8"),
    )
}

struct ParsedChatSseStream {
    payloads: Vec<Value>,
    saw_done: bool,
}

impl ParsedChatSseStream {
    fn parse(response_body: &str) -> Self {
        let mut payloads = Vec::new();
        let mut saw_done = false;
        for response_line in response_body.lines() {
            let Some(data_payload) = response_line.strip_prefix("data: ") else {
                continue;
            };
            if data_payload == "[DONE]" {
                saw_done = true;
                continue;
            }
            payloads.push(
                serde_json::from_str(data_payload)
                    .expect("each chat SSE data payload should be valid JSON"),
            );
        }
        Self { payloads, saw_done }
    }

    fn visible_text_for_opencode(&self) -> String {
        self.payloads
            .iter()
            .filter_map(|payload| payload["choices"][0]["delta"]["content"].as_str())
            .collect::<String>()
    }

    fn reasoning_text(&self) -> String {
        self.payloads
            .iter()
            .filter_map(|payload| payload["choices"][0]["delta"]["reasoning_content"].as_str())
            .collect::<String>()
    }

    fn finish_reason(&self) -> Option<&str> {
        self.payloads
            .iter()
            .filter_map(|payload| payload["choices"][0]["finish_reason"].as_str())
            .next_back()
    }
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
                astronomical_ipc_protocol::ExpertStorageFormat::StandardSafetensors,
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
