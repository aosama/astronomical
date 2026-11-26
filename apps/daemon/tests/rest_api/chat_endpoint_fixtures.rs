//! Scripted chat executor and `POST /v1/chat/completions` helpers shared by
//! the OpenAI chat endpoint tests.

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};

use astronomical_daemon::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerHealthSnapshot,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use ipc_protocol::{ChatGenerationCommand, ChatModelCapabilities, StructuredGenerationConstraint};
use tokio::sync::mpsc;
use tower::ServiceExt;

pub(super) type ReceivedStructuredConstraint = Arc<Mutex<Option<StructuredGenerationConstraint>>>;

/// Replays a fixed event script for every chat generation and records the
/// structured-generation constraint the daemon forwarded to the worker.
pub(super) struct ScriptedChatExecutor {
    health_snapshot: WorkerHealthSnapshot,
    stream_events: Vec<ChatGenerationStreamEvent>,
    received_structured_constraint: ReceivedStructuredConstraint,
}

impl ScriptedChatExecutor {
    pub(super) fn ready(
        ready_model_id: &str,
        stream_events: Vec<ChatGenerationStreamEvent>,
    ) -> Self {
        Self {
            health_snapshot: WorkerHealthSnapshot::ready_with_model(
                ready_model_id.to_owned(),
                ChatModelCapabilities {
                    supports_reasoning: true,
                    supports_tool_calls: true,
                    has_vision: true,
                    max_input_tokens: 241_664,
                    max_output_tokens: 20_480,
                    context_window: 262_144,
                },
            ),
            stream_events,
            received_structured_constraint: Arc::new(Mutex::new(None)),
        }
    }

    pub(super) fn received_structured_constraint_handle(&self) -> ReceivedStructuredConstraint {
        Arc::clone(&self.received_structured_constraint)
    }
}

impl ChatGenerationExecutor for ScriptedChatExecutor {
    fn start_chat_generation(
        &self,
        generation_command: ChatGenerationCommand,
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
            *self
                .received_structured_constraint
                .lock()
                .expect("the structured constraint log should not be poisoned") =
                generation_command.structured_generation.clone();
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

impl astronomical_daemon::ImageGenerationExecutor for ScriptedChatExecutor {}

pub(super) fn chat_completion_request(request_body: impl Into<Body>) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(request_body.into())
        .expect("the chat request should be valid")
}

/// Returns the status, the response content type and the whole body text so
/// one helper serves both the JSON and the server-sent-event assertions.
pub(super) async fn post_chat_completion(
    application: axum::Router,
    request_body: &'static str,
    body_limit_bytes: usize,
) -> (StatusCode, Option<String>, String) {
    let chat_response = application
        .oneshot(chat_completion_request(request_body))
        .await
        .expect("the application should return a chat response");
    let status = chat_response.status();
    let content_type = chat_response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|header_value| header_value.to_str().ok())
        .map(str::to_owned);
    let response_body = to_bytes(chat_response.into_body(), body_limit_bytes)
        .await
        .expect("the chat response body should be readable");
    (
        status,
        content_type,
        String::from_utf8(response_body.to_vec())
            .expect("the chat response body should contain UTF-8"),
    )
}
