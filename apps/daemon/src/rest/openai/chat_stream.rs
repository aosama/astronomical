use std::collections::VecDeque;

use axum::response::sse::Event;
use ipc_protocol::ChatGenerationCompletionReason;
use rest_contract::{
    OpenAiChatCompletionChunk, OpenAiErrorResponse, OpenAiFinishReason, OpenAiTokenUsage,
};
use serde::Serialize;

use super::chat_failure_wording::chat_failure_wording;
use crate::{ChatGenerationStreamErrorCode, ChatGenerationStreamEvent};

/// Encodes bounded daemon chat events into OpenAI-compatible SSE chunks.
pub(crate) struct OpenAiChatStreamEncoder {
    completion_id: String,
    created: u64,
    includes_usage: bool,
    model_id: String,
    request_id: u64,
    reasoning_excluded: bool,
    /// What drafting did for the reply, reported just before its completion
    /// and put in the final chunk's usage. A `Cell` because encoding takes
    /// `&self`.
    reply_drafting: std::cell::Cell<Option<ipc_protocol::RequestDraftingStatistics>>,
}

impl OpenAiChatStreamEncoder {
    pub(crate) fn new(
        request_id: u64,
        completion_id: String,
        created: u64,
        model_id: String,
        includes_usage: bool,
        reasoning_excluded: bool,
    ) -> Self {
        Self {
            completion_id,
            created,
            includes_usage,
            model_id,
            request_id,
            reasoning_excluded,
            reply_drafting: std::cell::Cell::new(None),
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
                if self.reasoning_excluded {
                    // The model still thought; only the client-visible deltas are withheld.
                    return Ok(VecDeque::new());
                }
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
            progress_or_drafting_event @ (ChatGenerationStreamEvent::PrefillProgress { .. }
            | ChatGenerationStreamEvent::DraftingReport(_)) => {
                self.keep_drafting_report(progress_or_drafting_event);
                Ok(VecDeque::new())
            }
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                astro_ssd_prompt_cache_restored_token_count,
                reason,
                ..
            } => self.completed_events(
                prompt_token_count,
                generated_token_count,
                astro_ssd_prompt_cache_restored_token_count,
                reason,
            ),
            ChatGenerationStreamEvent::Failed { reason } => {
                let wording = chat_failure_wording(&reason);
                self.error_event(wording.message, wording.code)
            }
            ChatGenerationStreamEvent::Error(error_code) => match error_code {
                ChatGenerationStreamErrorCode::WorkerUnavailable => self.error_event(
                    "the local worker became unavailable while processing the chat request",
                    "chat_worker_unavailable",
                ),
            },
        }
    }

    /// Keeps a drafting report for the final chunk's usage; progress events
    /// share the caller's arm and produce no chunk.
    fn keep_drafting_report(&self, stream_event: ChatGenerationStreamEvent) {
        if let ChatGenerationStreamEvent::DraftingReport(reply_drafting) = stream_event {
            self.reply_drafting.set(Some(reply_drafting));
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
        astro_ssd_prompt_cache_restored_token_count: u32,
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
            let token_usage = token_usage.with_astro_ssd_prompt_cache_restored_tokens(
                astro_ssd_prompt_cache_restored_token_count,
            );
            let token_usage =
                super::chat_completion::with_drafted_tokens(token_usage, self.reply_drafting.get());
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

#[cfg(test)]
#[path = "chat_stream_drafting_tests.rs"]
mod drafting_tests;
