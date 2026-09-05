use std::{future::Future, pin::Pin};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatModelCapabilities, MtpRuntimeState,
};
use astronomical_rest_contract::UNENFORCED_RESPONSE_FORMAT_WARNING;
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerHealthSnapshot,
    build_application,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use tokio::sync::mpsc;
use tower::ServiceExt;

const MODEL_ID: &str = "astronomical/non-streaming-test-model";

#[tokio::test]
async fn should_return_extracted_json_and_a_warning_for_json_schema_chat() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::TextFragment(
            "Juliet answers:\n```json\n{\"speaker\":\"Juliet\",\"play\":\"Romeo and Juliet\"}\n```"
                .to_owned(),
        ),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 12,
            generated_token_count: 8,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let chat_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "model":"astronomical/non-streaming-test-model",
                        "messages":[{"role":"user","content":"O Romeo, Romeo, wherefore art thou Romeo?"}],
                        "response_format":{
                            "type":"json_schema",
                            "json_schema":{
                                "name":"romeo_line",
                                "schema":{"type":"object","properties":{"speaker":{"type":"string"},"play":{"type":"string"}},"required":["speaker","play"]}
                            }
                        },
                        "stream":false
                    }"#,
                ))
                .expect("the structured chat request should be valid"),
        )
        .await
        .expect("the application should return a chat response");

    assert_eq!(chat_response.status(), StatusCode::OK);
    let warning_header = chat_response
        .headers()
        .get(header::WARNING)
        .and_then(|header_value| header_value.to_str().ok())
        .expect("unenforced json_schema must disclose a Warning header");
    assert_eq!(warning_header, UNENFORCED_RESPONSE_FORMAT_WARNING);
    let response_body = to_bytes(chat_response.into_body(), 16 * 1024)
        .await
        .expect("the chat body should be readable");
    let response_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the chat body should be JSON");
    let visible_content = response_document["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content should be text");
    let extracted_json: serde_json::Value =
        serde_json::from_str(visible_content).expect("visible content should be JSON");
    assert_eq!(
        extracted_json,
        serde_json::json!({"speaker": "Juliet", "play": "Romeo and Juliet"})
    );
}

#[tokio::test]
async fn should_keep_original_text_when_json_cannot_be_extracted() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::TextFragment(
            "Two households, both alike in dignity.".to_owned(),
        ),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 8,
            generated_token_count: 8,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let chat_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "model":"astronomical/non-streaming-test-model",
                        "messages":[{"role":"user","content":"O Romeo, Romeo, wherefore art thou Romeo?"}],
                        "response_format":{"type":"json_object"},
                        "stream":false
                    }"#,
                ))
                .expect("the json_object chat request should be valid"),
        )
        .await
        .expect("the application should return a chat response");

    assert_eq!(chat_response.status(), StatusCode::OK);
    let response_body = to_bytes(chat_response.into_body(), 16 * 1024)
        .await
        .expect("the chat body should be readable");
    let response_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the chat body should be JSON");
    assert_eq!(
        response_document["choices"][0]["message"]["content"],
        "Two households, both alike in dignity."
    );
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
