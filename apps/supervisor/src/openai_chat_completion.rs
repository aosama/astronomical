use astronomical_ipc_protocol::{ChatGenerationCompletionReason, ChatGenerationFailureReason};
use astronomical_rest_contract::{
    OpenAiAssistantMessage, OpenAiChatCompletionResponse, OpenAiErrorResponse, OpenAiFinishReason,
    OpenAiResponseToolCall, OpenAiStructuredOutput, OpenAiTokenUsage, compact_extracted_json_text,
};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::{ChatGenerationStreamErrorCode, ChatGenerationStreamEvent};

pub(crate) async fn create_non_streaming_chat_completion(
    mut chat_stream_event_receiver: mpsc::Receiver<ChatGenerationStreamEvent>,
    completion_id: String,
    created_at_unix_seconds: u64,
    model_id: String,
    structured_output: Option<&OpenAiStructuredOutput>,
) -> Response {
    let mut chat_completion_collector =
        OpenAiChatCompletionCollector::new(completion_id, created_at_unix_seconds, model_id);
    while let Some(chat_stream_event) = chat_stream_event_receiver.recv().await {
        match chat_stream_event {
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                reason,
                ..
            } => {
                if structured_output.is_some() {
                    chat_completion_collector.replace_visible_text_with_extracted_json();
                }
                return match chat_completion_collector.into_response(
                    prompt_token_count,
                    generated_token_count,
                    cached_token_count,
                    reason,
                ) {
                    Ok(chat_completion_response) => Json(chat_completion_response).into_response(),
                    Err(chat_completion_assembly_error) => {
                        tracing::error!(
                            error = %chat_completion_assembly_error,
                            "failed to assemble non-streaming OpenAI chat completion response"
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(OpenAiErrorResponse::service_unavailable(
                                "the local server could not assemble the chat completion",
                                Some("chat_completion_assembly_failed"),
                            )),
                        )
                            .into_response()
                    }
                };
            }
            ChatGenerationStreamEvent::Failed {
                reason:
                    ChatGenerationFailureReason::ContextLengthExceeded {
                        actual_total_context_tokens,
                        maximum_context_tokens,
                    },
            } => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(OpenAiErrorResponse::invalid_request(
                        format!(
                            "requested context uses {actual_total_context_tokens} tokens, exceeding the {maximum_context_tokens}-token model context window"
                        ),
                        Some("messages"),
                        Some("context_length_exceeded"),
                    )),
                )
                    .into_response();
            }
            stream_event_before_completion => {
                if let Some(error_response) =
                    chat_completion_collector.ingest_event(stream_event_before_completion)
                {
                    return (StatusCode::SERVICE_UNAVAILABLE, Json(error_response)).into_response();
                }
            }
        }
    }

    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(OpenAiErrorResponse::service_unavailable(
            "the local worker became unavailable while processing the chat request",
            Some("chat_worker_unavailable"),
        )),
    )
        .into_response()
}

/// Collects ordered worker chat events into one OpenAI-compatible non-streaming response.
///
/// Mirrors `OpenAiChatStreamEncoder` so streaming and non-streaming modes stay
/// behaviorally congruent: same finish-reason mapping, same tool-call id format,
/// same error code/message pairs. Only the wire shape differs (one JSON body
/// instead of an SSE frame sequence).
pub(crate) struct OpenAiChatCompletionCollector {
    completion_id: String,
    created: u64,
    model_id: String,
    text_content: String,
    reasoning_content: String,
    tool_calls: Vec<OpenAiResponseToolCall>,
}

impl OpenAiChatCompletionCollector {
    pub(crate) fn new(completion_id: String, created: u64, model_id: String) -> Self {
        Self {
            completion_id,
            created,
            model_id,
            text_content: String::new(),
            reasoning_content: String::new(),
            tool_calls: Vec::new(),
        }
    }

    pub(crate) fn replace_visible_text_with_extracted_json(&mut self) {
        if !self.tool_calls.is_empty() {
            return;
        }
        if let Some(compact_json) = compact_extracted_json_text(&self.text_content) {
            self.text_content = compact_json;
        }
    }

    /// Ingests one ordered worker event.
    ///
    /// Returns `None` for collectable output events. Returns `Some(error_response)`
    /// for `Failed`/`Error` events so the caller can short-circuit with the
    /// OpenAI error body. The caller handles `Completed` before calling this.
    pub(crate) fn ingest_event(
        &mut self,
        stream_event: ChatGenerationStreamEvent,
    ) -> Option<OpenAiErrorResponse> {
        match stream_event {
            ChatGenerationStreamEvent::ReasoningFragment(reasoning) => {
                self.reasoning_content.push_str(&reasoning);
                None
            }
            ChatGenerationStreamEvent::TextFragment(text) => {
                self.text_content.push_str(&text);
                None
            }
            ChatGenerationStreamEvent::ToolCall {
                tool_call_index,
                function_name,
                arguments_json,
            } => {
                let tool_call_id = format!("call_{}_{}", self.completion_id, tool_call_index);
                self.tool_calls.push(OpenAiResponseToolCall::function(
                    tool_call_id,
                    function_name,
                    arguments_json,
                ));
                None
            }
            ChatGenerationStreamEvent::PrefillProgress { .. } => None,
            ChatGenerationStreamEvent::Completed { .. } => None,
            ChatGenerationStreamEvent::Failed { reason } => Some(match reason {
                ChatGenerationFailureReason::InvalidRequest { reason } => {
                    OpenAiErrorResponse::service_unavailable(
                        format!("the local worker rejected the chat request: {reason}"),
                        Some("chat_invalid_request"),
                    )
                }
                ChatGenerationFailureReason::FatalExecution { reason } => {
                    OpenAiErrorResponse::service_unavailable(
                        format!(
                            "the local worker stopped after a fatal model execution error: {reason}"
                        ),
                        Some("chat_worker_unavailable"),
                    )
                }
                ChatGenerationFailureReason::ContextLengthExceeded {
                    actual_total_context_tokens,
                    maximum_context_tokens,
                } => OpenAiErrorResponse::invalid_request(
                    format!(
                        "requested context uses {actual_total_context_tokens} tokens, exceeding the {maximum_context_tokens}-token model context window"
                    ),
                    Some("messages"),
                    Some("context_length_exceeded"),
                ),
                ChatGenerationFailureReason::EngineBusy => {
                    OpenAiErrorResponse::service_unavailable(
                        "the local inference engine is already processing another request",
                        Some("chat_engine_busy"),
                    )
                }
                ChatGenerationFailureReason::MalformedModelOutput => {
                    OpenAiErrorResponse::service_unavailable(
                        "the model produced malformed structured output",
                        Some("chat_malformed_model_output"),
                    )
                }
            }),
            ChatGenerationStreamEvent::Error(error_code) => match error_code {
                ChatGenerationStreamErrorCode::WorkerUnavailable => {
                    Some(OpenAiErrorResponse::service_unavailable(
                        "the local worker became unavailable while processing the chat request",
                        Some("chat_worker_unavailable"),
                    ))
                }
            },
        }
    }

    /// Builds the final non-streaming response after a `Completed` event.
    pub(crate) fn into_response(
        self,
        prompt_token_count: u32,
        generated_token_count: u16,
        cached_token_count: u32,
        completion_reason: ChatGenerationCompletionReason,
    ) -> Result<OpenAiChatCompletionResponse, OpenAiChatCompletionCollectorError> {
        let finish_reason = match completion_reason {
            ChatGenerationCompletionReason::EndOfSequence
            | ChatGenerationCompletionReason::Cancelled => OpenAiFinishReason::Stop,
            ChatGenerationCompletionReason::MaximumOutputTokens => OpenAiFinishReason::Length,
            ChatGenerationCompletionReason::ToolCalls => OpenAiFinishReason::ToolCalls,
        };
        let text_content = if self.text_content.is_empty() {
            None
        } else {
            Some(self.text_content)
        };
        let reasoning_content = if self.reasoning_content.is_empty() {
            None
        } else {
            Some(self.reasoning_content)
        };
        let assistant_message =
            OpenAiAssistantMessage::new(text_content, reasoning_content, self.tool_calls);
        let token_usage =
            OpenAiTokenUsage::new(prompt_token_count, u32::from(generated_token_count))
                .ok_or(OpenAiChatCompletionCollectorError::TokenUsageOverflow {
                    prompt_token_count,
                    generated_token_count,
                })?
                .with_cached_tokens(cached_token_count);
        Ok(OpenAiChatCompletionResponse::new(
            self.completion_id,
            self.created,
            self.model_id,
            assistant_message,
            finish_reason,
            token_usage,
        ))
    }
}

/// Internal assembly failure for one non-streaming chat completion.
#[derive(Clone, Copy, Debug, Error)]
pub(crate) enum OpenAiChatCompletionCollectorError {
    #[error(
        "token usage overflowed: prompt_tokens={prompt_token_count}, completion_tokens={generated_token_count}"
    )]
    TokenUsageOverflow {
        prompt_token_count: u32,
        generated_token_count: u16,
    },
}
