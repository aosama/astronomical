//! Collects ordered worker events into one non-streaming Anthropic message,
//! plus the stop-reason, identifier, and failure mappings the stream encoder
//! shares.

use astronomical_ipc_protocol::{ChatGenerationCompletionReason, ChatGenerationFailureReason};
use astronomical_rest_contract::{
    AnthropicError, AnthropicMessage, AnthropicOutputContentBlock, AnthropicStopReason,
    AnthropicUsage,
};
use serde_json::Value;
use thiserror::Error;

use crate::{ChatGenerationStreamErrorCode, ChatGenerationStreamEvent};

pub(crate) struct AnthropicMessageCollector {
    message_id: String,
    model_id: String,
    thinking_text: String,
    output_text: String,
    tool_uses: Vec<CollectedToolUse>,
}

struct CollectedToolUse {
    tool_call_index: u16,
    function_name: String,
    arguments_json: String,
}

/// The worker's token counts from its terminal completion event.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorkerTokenCounts {
    pub(crate) prompt_token_count: u32,
    pub(crate) cached_token_count: u32,
    pub(crate) generated_token_count: u16,
}

impl WorkerTokenCounts {
    pub(crate) fn anthropic_usage(self) -> AnthropicUsage {
        AnthropicUsage::from_worker_counts(
            self.prompt_token_count,
            self.cached_token_count,
            u32::from(self.generated_token_count),
        )
    }
}

impl AnthropicMessageCollector {
    pub(crate) fn new(message_id: String, model_id: String) -> Self {
        Self {
            message_id,
            model_id,
            thinking_text: String::new(),
            output_text: String::new(),
            tool_uses: Vec::new(),
        }
    }

    /// Accumulates one non-terminal event; terminal events belong to the
    /// caller, which owns the completion and failure responses.
    pub(crate) fn ingest_event(
        &mut self,
        stream_event: ChatGenerationStreamEvent,
    ) -> Result<(), AnthropicMessagesAssemblyError> {
        match stream_event {
            ChatGenerationStreamEvent::ReasoningFragment(thinking_text) => {
                self.thinking_text.push_str(&thinking_text);
            }
            ChatGenerationStreamEvent::TextFragment(output_text) => {
                self.output_text.push_str(&output_text);
            }
            ChatGenerationStreamEvent::ToolCall {
                tool_call_index,
                function_name,
                arguments_json,
            } => self.tool_uses.push(CollectedToolUse {
                tool_call_index,
                function_name,
                arguments_json,
            }),
            ChatGenerationStreamEvent::PrefillProgress { .. } => {}
            ChatGenerationStreamEvent::Completed { .. } => {
                return Err(AnthropicMessagesAssemblyError::UnexpectedCompletionEvent);
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                return Err(AnthropicMessagesAssemblyError::WorkerFailure { reason });
            }
            ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable) => {
                return Err(AnthropicMessagesAssemblyError::WorkerUnavailable);
            }
        }
        Ok(())
    }

    /// Builds the message in Anthropic block order: thinking, text, tool uses.
    pub(crate) fn into_message(
        self,
        worker_token_counts: WorkerTokenCounts,
        completion_reason: ChatGenerationCompletionReason,
    ) -> Result<AnthropicMessage, AnthropicMessagesAssemblyError> {
        let stop_reason = anthropic_stop_reason(completion_reason)?;
        let mut content_blocks = Vec::new();
        if !self.thinking_text.is_empty() {
            content_blocks.push(AnthropicOutputContentBlock::Thinking {
                thinking: self.thinking_text,
                signature: String::new(),
            });
        }
        if !self.output_text.is_empty() {
            content_blocks.push(AnthropicOutputContentBlock::Text {
                text: self.output_text,
            });
        }
        for tool_use in self.tool_uses {
            content_blocks.push(AnthropicOutputContentBlock::ToolUse {
                id: tool_use_id(&self.message_id, tool_use.tool_call_index),
                name: tool_use.function_name,
                input: parse_tool_input(&tool_use.arguments_json)?,
            });
        }
        let mut message = AnthropicMessage::empty(self.message_id, self.model_id);
        message.content = content_blocks;
        message.stop_reason = Some(stop_reason);
        message.usage = worker_token_counts.anthropic_usage();
        Ok(message)
    }
}

pub(crate) fn anthropic_stop_reason(
    completion_reason: ChatGenerationCompletionReason,
) -> Result<AnthropicStopReason, AnthropicMessagesAssemblyError> {
    match completion_reason {
        ChatGenerationCompletionReason::EndOfSequence => Ok(AnthropicStopReason::EndTurn),
        ChatGenerationCompletionReason::MaximumOutputTokens => Ok(AnthropicStopReason::MaxTokens),
        ChatGenerationCompletionReason::ToolCalls => Ok(AnthropicStopReason::ToolUse),
        ChatGenerationCompletionReason::Cancelled => Err(AnthropicMessagesAssemblyError::Cancelled),
    }
}

/// Tool-use identifiers reuse the message suffix so they stay unique across
/// requests and stable between the streamed and non-streamed forms.
pub(crate) fn tool_use_id(message_id: &str, tool_call_index: u16) -> String {
    let identifier_suffix = message_id.strip_prefix("msg_").unwrap_or(message_id);
    format!("toolu_{identifier_suffix}-{tool_call_index}")
}

/// The worker validates tool arguments as a JSON document before emitting
/// them, so a parse failure here means a broken worker contract.
pub(crate) fn parse_tool_input(
    arguments_json: &str,
) -> Result<Value, AnthropicMessagesAssemblyError> {
    serde_json::from_str(arguments_json)
        .map_err(AnthropicMessagesAssemblyError::MalformedToolArguments)
}

/// Maps a worker failure onto an Anthropic error. Context overflow uses
/// Anthropic's own "prompt is too long" wording because Anthropic clients
/// match that text to trigger conversation compaction.
pub(crate) fn failure_error(failure_reason: &ChatGenerationFailureReason) -> AnthropicError {
    match failure_reason {
        ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens,
            maximum_context_tokens,
        } => AnthropicError::invalid_request(format!(
            "prompt is too long: {actual_total_context_tokens} tokens > {maximum_context_tokens} maximum"
        )),
        ChatGenerationFailureReason::InvalidRequest { reason } => {
            AnthropicError::invalid_request(reason.clone())
        }
        ChatGenerationFailureReason::EngineBusy => {
            AnthropicError::overloaded("the local worker is busy with another request")
        }
        other_failure_reason => AnthropicError::api_error(format!(
            "the local worker could not generate the message: {other_failure_reason:?}"
        )),
    }
}

#[derive(Debug, Error)]
pub(crate) enum AnthropicMessagesAssemblyError {
    #[error("received a completion event through the non-terminal ingestion path")]
    UnexpectedCompletionEvent,
    #[error("the local worker rejected the message request: {reason:?}")]
    WorkerFailure { reason: ChatGenerationFailureReason },
    #[error("the local worker became unavailable while generating the message")]
    WorkerUnavailable,
    #[error("the message request was cancelled")]
    Cancelled,
    #[error("the worker emitted tool arguments that are not JSON: {0}")]
    MalformedToolArguments(#[source] serde_json::Error),
}
