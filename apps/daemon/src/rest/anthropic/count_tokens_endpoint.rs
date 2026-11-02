//! `POST /v1/messages/count_tokens`: the prompt token count of a Messages
//! conversation, measured by the worker's own preparation path (chat
//! template, tokenizer, and image-token expansion) rather than estimated, so
//! the count matches what `POST /v1/messages` would prefill. Anthropic
//! clients use it to decide when to compact a conversation.

use astronomical_rest_contract::{
    AnthropicCountTokensRequest, AnthropicMessagesRequestParts, AnthropicTokenCount,
};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    response::{IntoResponse, Response},
};
use tokio::sync::mpsc::Receiver;

use super::messages_endpoint::admit_message_generation;
use super::messages_errors::{
    anthropic_error_response, failure_error, json_rejection_error, validation_rejection_error,
    worker_unavailable_error,
};
use crate::application::ApplicationState;
use crate::serving::chat_generation_executor::ChatWork;
use crate::{ChatGenerationStreamErrorCode, ChatGenerationStreamEvent};

pub(crate) async fn count_message_tokens(
    State(application_state): State<ApplicationState>,
    request_body_bytes: Bytes,
) -> Response {
    let request_parts = match parse_count_tokens_request(request_body_bytes.as_ref()) {
        Ok(request_parts) => request_parts,
        Err(rejection_response) => return rejection_response,
    };
    let admitted_count = match admit_message_generation(
        &application_state,
        request_parts,
        request_body_bytes.len(),
        ChatWork::CountPromptTokens,
    )
    .await
    {
        Ok(admitted_count) => admitted_count,
        Err(rejection_response) => return rejection_response,
    };
    await_token_count(admitted_count.stream_event_receiver).await
}

// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn parse_count_tokens_request(
    request_body_bytes: &[u8],
) -> Result<AnthropicMessagesRequestParts, Response> {
    let count_request =
        match serde_json::from_slice::<AnthropicCountTokensRequest>(request_body_bytes) {
            Ok(count_request) => count_request,
            Err(json_error) => {
                return Err(anthropic_error_response(json_rejection_error(&json_error)));
            }
        };
    match count_request.into_parts() {
        Ok(request_parts) => Ok(request_parts),
        Err(validation_error) => Err(anthropic_error_response(validation_rejection_error(
            &validation_error,
        ))),
    }
}

/// The worker answers a count with one `Completed` event whose prompt count
/// is the answer; no output event precedes it, because nothing generates.
async fn await_token_count(
    mut stream_event_receiver: Receiver<ChatGenerationStreamEvent>,
) -> Response {
    while let Some(stream_event) = stream_event_receiver.recv().await {
        match stream_event {
            ChatGenerationStreamEvent::Completed {
                prompt_token_count, ..
            } => {
                return Json(AnthropicTokenCount {
                    input_tokens: prompt_token_count,
                })
                .into_response();
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                return anthropic_error_response(failure_error(&reason));
            }
            ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable) => {
                return anthropic_error_response(worker_unavailable_error());
            }
            ChatGenerationStreamEvent::ReasoningFragment(_)
            | ChatGenerationStreamEvent::TextFragment(_)
            | ChatGenerationStreamEvent::ToolCall { .. }
            | ChatGenerationStreamEvent::PrefillProgress { .. } => {}
        }
    }
    anthropic_error_response(worker_unavailable_error())
}
