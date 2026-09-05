use std::time::{SystemTime, UNIX_EPOCH};

use astronomical_ipc_protocol::{ChatGenerationCompletionReason, ChatGenerationFailureReason};
use astronomical_rest_contract::{
    OpenAiResponse, OpenAiResponseOutputItem, OpenAiResponseRequestConfiguration,
    OpenAiResponseUsage, compact_extracted_json_text,
};
use thiserror::Error;

use crate::{ChatGenerationStreamErrorCode, ChatGenerationStreamEvent};

/// Collects ordered worker events into one terminal Responses object.
pub struct OpenAiResponsesCollector {
    response_id: String,
    created_at_unix_seconds: u64,
    model_id: String,
    instructions: Option<String>,
    request_configuration: OpenAiResponseRequestConfiguration,
    reasoning_text: String,
    output_text: String,
    function_calls: Vec<CollectedFunctionCall>,
}

struct CollectedFunctionCall {
    tool_call_index: u16,
    function_name: String,
    arguments_json: String,
}

impl OpenAiResponsesCollector {
    #[must_use]
    pub fn new(
        response_id: String,
        created_at_unix_seconds: u64,
        model_id: String,
        instructions: Option<String>,
        request_configuration: OpenAiResponseRequestConfiguration,
    ) -> Self {
        Self {
            response_id,
            created_at_unix_seconds,
            model_id,
            instructions,
            request_configuration,
            reasoning_text: String::new(),
            output_text: String::new(),
            function_calls: Vec::new(),
        }
    }

    pub(crate) fn replace_output_text_with_extracted_json(&mut self) {
        if !self.function_calls.is_empty() {
            return;
        }
        if let Some(compact_json) = compact_extracted_json_text(&self.output_text) {
            self.output_text = compact_json;
        }
    }

    pub fn ingest_event(
        &mut self,
        stream_event: ChatGenerationStreamEvent,
    ) -> Result<(), OpenAiResponsesAssemblyError> {
        match stream_event {
            ChatGenerationStreamEvent::ReasoningFragment(reasoning_text) => {
                self.reasoning_text.push_str(&reasoning_text);
                Ok(())
            }
            ChatGenerationStreamEvent::TextFragment(output_text) => {
                self.output_text.push_str(&output_text);
                Ok(())
            }
            ChatGenerationStreamEvent::ToolCall {
                tool_call_index,
                function_name,
                arguments_json,
            } => {
                self.function_calls.push(CollectedFunctionCall {
                    tool_call_index,
                    function_name,
                    arguments_json,
                });
                Ok(())
            }
            ChatGenerationStreamEvent::PrefillProgress { .. } => Ok(()),
            ChatGenerationStreamEvent::Completed { .. } => {
                Err(OpenAiResponsesAssemblyError::UnexpectedCompletionEvent)
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                Err(OpenAiResponsesAssemblyError::WorkerFailure { reason })
            }
            ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable) => {
                Err(OpenAiResponsesAssemblyError::WorkerUnavailable)
            }
        }
    }

    pub(crate) fn reasoning_text(&self) -> &str {
        &self.reasoning_text
    }

    pub(crate) fn output_text(&self) -> &str {
        &self.output_text
    }

    pub fn into_response(
        self,
        input_token_count: u32,
        output_token_count: u16,
        cached_input_token_count: u32,
        reasoning_token_count: u16,
        completion_reason: ChatGenerationCompletionReason,
    ) -> Result<OpenAiResponse, OpenAiResponsesAssemblyError> {
        if completion_reason == ChatGenerationCompletionReason::Cancelled {
            return Err(OpenAiResponsesAssemblyError::Cancelled);
        }
        let completed_at_unix_seconds = current_unix_timestamp_seconds()?;
        let output_items = self.output_items();
        let response_id = self.response_id;
        let created_at_unix_seconds = self.created_at_unix_seconds;
        let model_id = self.model_id;
        let instructions = self.instructions;
        let request_configuration = self.request_configuration;
        let usage = OpenAiResponseUsage::new(
            input_token_count,
            u32::from(output_token_count),
            cached_input_token_count,
            u32::from(reasoning_token_count),
        )
        .ok_or(OpenAiResponsesAssemblyError::TokenUsageOverflow {
            input_token_count,
            output_token_count,
        })?;
        if completion_reason == ChatGenerationCompletionReason::MaximumOutputTokens {
            return Ok(OpenAiResponse::incomplete_at_output_token_limit(
                response_id,
                created_at_unix_seconds,
                model_id,
                instructions,
                output_items,
                usage,
            )
            .with_request_configuration(&request_configuration));
        }
        Ok(OpenAiResponse::completed(
            response_id,
            created_at_unix_seconds,
            completed_at_unix_seconds,
            model_id,
            instructions,
            output_items,
            usage,
        )
        .with_request_configuration(&request_configuration))
    }

    #[must_use]
    pub fn into_failed_response(
        self,
        failure_reason: ChatGenerationFailureReason,
    ) -> OpenAiResponse {
        let (error_code, error_message) = failure_details(&failure_reason);
        let output_items = self.output_items();
        let response_id = self.response_id;
        let created_at_unix_seconds = self.created_at_unix_seconds;
        let model_id = self.model_id;
        let instructions = self.instructions;
        let request_configuration = self.request_configuration;
        OpenAiResponse::failed(
            response_id,
            created_at_unix_seconds,
            model_id,
            instructions,
            output_items,
            error_code,
            error_message,
        )
        .with_request_configuration(&request_configuration)
    }

    fn output_items(&self) -> Vec<OpenAiResponseOutputItem> {
        let identifier_suffix = self
            .response_id
            .strip_prefix("resp_")
            .unwrap_or(&self.response_id);
        let mut output_items = Vec::new();
        if !self.reasoning_text.is_empty() {
            output_items.push(OpenAiResponseOutputItem::reasoning(
                format!("rs_{identifier_suffix}"),
                self.reasoning_text.clone(),
            ));
        }
        if !self.output_text.is_empty() {
            output_items.push(OpenAiResponseOutputItem::message(
                format!("msg_{identifier_suffix}"),
                self.output_text.clone(),
            ));
        }
        for function_call in &self.function_calls {
            output_items.push(OpenAiResponseOutputItem::function_call(
                format!("fc_{identifier_suffix}-{}", function_call.tool_call_index),
                format!("call_{identifier_suffix}-{}", function_call.tool_call_index),
                function_call.function_name.clone(),
                function_call.arguments_json.clone(),
            ));
        }
        output_items
    }
}

fn current_unix_timestamp_seconds() -> Result<u64, OpenAiResponsesAssemblyError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration_since_epoch| duration_since_epoch.as_secs())
        .map_err(
            |system_time_error| OpenAiResponsesAssemblyError::SystemClockBeforeUnixEpoch {
                system_time_error,
            },
        )
}

fn failure_details(failure_reason: &ChatGenerationFailureReason) -> (String, String) {
    match failure_reason {
        ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens,
            maximum_context_tokens,
        } => (
            "context_length_exceeded".to_owned(),
            format!(
                "requested context uses {actual_total_context_tokens} tokens, exceeding the {maximum_context_tokens}-token model context window"
            ),
        ),
        other_failure_reason => (
            "response_generation_failed".to_owned(),
            format!("the local worker could not generate the response: {other_failure_reason:?}"),
        ),
    }
}

#[derive(Debug, Error)]
pub enum OpenAiResponsesAssemblyError {
    #[error("received a completion event through the non-terminal ingestion path")]
    UnexpectedCompletionEvent,
    #[error("the local worker rejected the response request: {reason:?}")]
    WorkerFailure { reason: ChatGenerationFailureReason },
    #[error("the local worker became unavailable while generating a response")]
    WorkerUnavailable,
    #[error("the response request was cancelled")]
    Cancelled,
    #[error("the system clock predates the Unix epoch: {system_time_error}")]
    SystemClockBeforeUnixEpoch {
        #[source]
        system_time_error: std::time::SystemTimeError,
    },
    #[error(
        "token usage overflowed: input_tokens={input_token_count}, output_tokens={output_token_count}"
    )]
    TokenUsageOverflow {
        input_token_count: u32,
        output_token_count: u16,
    },
}
