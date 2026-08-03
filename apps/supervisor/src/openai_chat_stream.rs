use std::collections::VecDeque;

use astronomical_ipc_protocol::{ChatGenerationCompletionReason, ChatGenerationFailureReason};
use astronomical_rest_contract::{
    OpenAiChatCompletionChunk, OpenAiErrorResponse, OpenAiFinishReason, OpenAiTokenUsage,
};
use axum::response::sse::Event;
use serde::Serialize;

use crate::{ChatGenerationStreamErrorCode, ChatGenerationStreamEvent};

/// Encodes bounded supervisor chat events into OpenAI-compatible SSE chunks.
pub(crate) struct OpenAiChatStreamEncoder {
    completion_id: String,
    created: u64,
    includes_usage: bool,
    model_id: String,
    request_id: u64,
}

impl OpenAiChatStreamEncoder {
    pub(crate) fn new(
        request_id: u64,
        completion_id: String,
        created: u64,
        model_id: String,
        includes_usage: bool,
    ) -> Self {
        Self {
            completion_id,
            created,
            includes_usage,
            model_id,
            request_id,
        }
    }

    pub(crate) fn initial_event(&self) -> Result<Event, axum::Error> {
        self.serialized_event(
            "assistant_role",
            OpenAiChatCompletionChunk::assistant_role(
                self.completion_id.clone(),
                self.created,
                self.model_id.clone(),
            ),
        )
    }

    pub(crate) fn encode(
        &self,
        stream_event: ChatGenerationStreamEvent,
    ) -> Result<VecDeque<Event>, axum::Error> {
        match stream_event {
            ChatGenerationStreamEvent::ReasoningFragment(reasoning) => {
                self.single_chunk(OpenAiChatCompletionChunk::reasoning_delta(
                    self.completion_id.clone(),
                    self.created,
                    self.model_id.clone(),
                    reasoning,
                ))
            }
            ChatGenerationStreamEvent::TextFragment(text) => {
                self.single_chunk(OpenAiChatCompletionChunk::text_delta(
                    self.completion_id.clone(),
                    self.created,
                    self.model_id.clone(),
                    text,
                ))
            }
            ChatGenerationStreamEvent::ToolCall {
                tool_call_index,
                function_name,
                arguments_json,
            } => self.single_chunk(OpenAiChatCompletionChunk::tool_call_delta(
                self.completion_id.clone(),
                self.created,
                self.model_id.clone(),
                tool_call_index,
                format!("call_{}_{}", self.completion_id, tool_call_index),
                function_name,
                arguments_json,
            )),
            ChatGenerationStreamEvent::PrefillProgress { .. } => Ok(VecDeque::new()),
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                reason,
                ..
            } => self.completed_events(
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                reason,
            ),
            ChatGenerationStreamEvent::Failed { reason } => match reason {
                ChatGenerationFailureReason::ContextLengthExceeded {
                    actual_total_context_tokens,
                    maximum_context_tokens,
                } => self.error_event(
                    format!(
                        "requested context uses {actual_total_context_tokens} tokens, exceeding the {maximum_context_tokens}-token model context window"
                    ),
                    "context_length_exceeded",
                ),
                ChatGenerationFailureReason::InvalidRequest { reason } => self.error_event(
                    format!("the local worker rejected the chat request: {reason}"),
                    "chat_invalid_request",
                ),
                ChatGenerationFailureReason::FatalExecution { reason } => self.error_event(
                    format!("the local worker stopped after a fatal model execution error: {reason}"),
                    "chat_worker_unavailable",
                ),
                ChatGenerationFailureReason::EngineBusy => self.error_event(
                    "the local inference engine is already processing another request",
                    "chat_engine_busy",
                ),
                ChatGenerationFailureReason::MalformedModelOutput => self.error_event(
                    "the model produced malformed structured output",
                    "chat_malformed_model_output",
                ),
            },
            ChatGenerationStreamEvent::Error(error_code) => match error_code {
                ChatGenerationStreamErrorCode::WorkerUnavailable => self.error_event(
                    "the local worker became unavailable while processing the chat request",
                    "chat_worker_unavailable",
                ),
            },
        }
    }

    fn error_event(
        &self,
        message: impl Into<String>,
        code: &'static str,
    ) -> Result<VecDeque<Event>, axum::Error> {
        self.serialized_event(
            "error",
            OpenAiErrorResponse::service_unavailable(message, Some(code)),
        )
        .map(|event| VecDeque::from([event]))
    }

    fn single_chunk(
        &self,
        completion_chunk: OpenAiChatCompletionChunk,
    ) -> Result<VecDeque<Event>, axum::Error> {
        self.serialized_event("chunk", completion_chunk)
            .map(|event| VecDeque::from([event]))
    }

    fn completed_events(
        &self,
        prompt_token_count: u32,
        generated_token_count: u16,
        cached_token_count: u32,
        completion_reason: ChatGenerationCompletionReason,
    ) -> Result<VecDeque<Event>, axum::Error> {
        let finish_reason = match completion_reason {
            ChatGenerationCompletionReason::EndOfSequence => OpenAiFinishReason::Stop,
            ChatGenerationCompletionReason::MaximumOutputTokens => OpenAiFinishReason::Length,
            ChatGenerationCompletionReason::ToolCalls => OpenAiFinishReason::ToolCalls,
            ChatGenerationCompletionReason::Cancelled => OpenAiFinishReason::Stop,
        };
        let mut completion_chunk = OpenAiChatCompletionChunk::finished(
            self.completion_id.clone(),
            self.created,
            self.model_id.clone(),
            finish_reason,
        );
        if self.includes_usage
            && let Some(token_usage) =
                OpenAiTokenUsage::new(prompt_token_count, u32::from(generated_token_count))
        {
            let token_usage = token_usage.with_cached_tokens(cached_token_count);
            completion_chunk = completion_chunk.with_usage(token_usage);
        }
        let completion_event = self.serialized_event("completion", completion_chunk)?;
        tracing::trace!(
            request_id = self.request_id,
            completion_id = %self.completion_id,
            sse_data = "[DONE]",
            "encoded OpenAI SSE response event"
        );
        Ok(VecDeque::from([
            completion_event,
            Event::default().data("[DONE]"),
        ]))
    }

    fn serialized_event(
        &self,
        event_kind: &'static str,
        sse_payload: impl Serialize,
    ) -> Result<Event, axum::Error> {
        let sse_data = serde_json::to_string(&sse_payload).map_err(axum::Error::new)?;
        tracing::trace!(
            request_id = self.request_id,
            completion_id = %self.completion_id,
            event_kind,
            "encoded OpenAI SSE response event"
        );
        Ok(Event::default().data(sse_data))
    }
}
