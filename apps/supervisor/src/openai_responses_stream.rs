use std::collections::VecDeque;

use astronomical_ipc_protocol::{ChatGenerationCompletionReason, ChatGenerationFailureReason};
use astronomical_rest_contract::{
    OpenAiResponse, OpenAiResponseOutputContent, OpenAiResponseOutputItem,
    OpenAiResponseRequestConfiguration, OpenAiResponseStreamEvent,
};
use thiserror::Error;

use crate::{
    ChatGenerationStreamErrorCode, ChatGenerationStreamEvent, OpenAiResponsesAssemblyError,
    OpenAiResponsesCollector,
};

/// Encodes supervisor generation events into semantic Responses events.
pub struct OpenAiResponsesStreamEncoder {
    response_id: String,
    created_at_unix_seconds: u64,
    model_id: String,
    instructions: Option<String>,
    request_configuration: OpenAiResponseRequestConfiguration,
    next_sequence_number: u64,
    next_output_index: usize,
    reasoning_output_index: Option<usize>,
    text_output_index: Option<usize>,
    collector: Option<OpenAiResponsesCollector>,
}

impl OpenAiResponsesStreamEncoder {
    #[must_use]
    pub fn new(
        response_id: String,
        created_at_unix_seconds: u64,
        model_id: String,
        instructions: Option<String>,
        request_configuration: OpenAiResponseRequestConfiguration,
    ) -> Self {
        let collector = OpenAiResponsesCollector::new(
            response_id.clone(),
            created_at_unix_seconds,
            model_id.clone(),
            instructions.clone(),
            request_configuration.clone(),
        );
        Self {
            response_id,
            created_at_unix_seconds,
            model_id,
            instructions,
            request_configuration,
            next_sequence_number: 0,
            next_output_index: 0,
            reasoning_output_index: None,
            text_output_index: None,
            collector: Some(collector),
        }
    }

    #[must_use]
    pub fn initial_events(&mut self) -> VecDeque<OpenAiResponseStreamEvent> {
        let response = self.in_progress_response();
        let created_sequence_number = self.take_sequence_number();
        let in_progress_sequence_number = self.take_sequence_number();
        VecDeque::from([
            OpenAiResponseStreamEvent::Created {
                sequence_number: created_sequence_number,
                response: response.clone(),
            },
            OpenAiResponseStreamEvent::InProgress {
                sequence_number: in_progress_sequence_number,
                response,
            },
        ])
    }

    pub fn encode(
        &mut self,
        stream_event: ChatGenerationStreamEvent,
    ) -> Result<VecDeque<OpenAiResponseStreamEvent>, OpenAiResponsesStreamEncodingError> {
        match stream_event {
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                reasoning_token_count,
                cached_token_count,
                reason,
            } => self.complete(
                prompt_token_count,
                generated_token_count,
                reasoning_token_count,
                cached_token_count,
                reason,
            ),
            ChatGenerationStreamEvent::Failed { reason } => Ok(self.worker_failure(reason)),
            ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable) => {
                Ok(self.error_event(
                    "worker_unavailable",
                    "the local worker became unavailable while generating the response",
                    None,
                ))
            }
            ChatGenerationStreamEvent::PrefillProgress { .. } => Ok(VecDeque::new()),
            ChatGenerationStreamEvent::ReasoningFragment(reasoning_text) => {
                self.collector_mut()?.ingest_event(
                    ChatGenerationStreamEvent::ReasoningFragment(reasoning_text.clone()),
                )?;
                Ok(self.reasoning_delta(reasoning_text))
            }
            ChatGenerationStreamEvent::TextFragment(output_text) => {
                self.collector_mut()?
                    .ingest_event(ChatGenerationStreamEvent::TextFragment(output_text.clone()))?;
                Ok(self.output_text_delta(output_text))
            }
            ChatGenerationStreamEvent::ToolCall {
                tool_call_index,
                function_name,
                arguments_json,
            } => {
                self.collector_mut()?
                    .ingest_event(ChatGenerationStreamEvent::ToolCall {
                        tool_call_index,
                        function_name: function_name.clone(),
                        arguments_json: arguments_json.clone(),
                    })?;
                Ok(self.function_call(tool_call_index, function_name, arguments_json))
            }
        }
    }

    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.collector.is_none()
    }

    fn reasoning_delta(&mut self, reasoning_delta: String) -> VecDeque<OpenAiResponseStreamEvent> {
        let mut events = VecDeque::new();
        let reasoning_output_index = match self.reasoning_output_index {
            Some(reasoning_output_index) => reasoning_output_index,
            None => {
                let reasoning_output_index = self.take_output_index();
                self.reasoning_output_index = Some(reasoning_output_index);
                let sequence_number = self.take_sequence_number();
                events.push_back(OpenAiResponseStreamEvent::OutputItemAdded {
                    sequence_number,
                    output_index: reasoning_output_index,
                    item: OpenAiResponseOutputItem::reasoning_in_progress(self.reasoning_item_id()),
                });
                reasoning_output_index
            }
        };
        let sequence_number = self.take_sequence_number();
        events.push_back(OpenAiResponseStreamEvent::ReasoningSummaryTextDelta {
            sequence_number,
            item_id: self.reasoning_item_id(),
            output_index: reasoning_output_index,
            summary_index: 0,
            delta: reasoning_delta,
        });
        events
    }

    fn output_text_delta(
        &mut self,
        output_text_delta: String,
    ) -> VecDeque<OpenAiResponseStreamEvent> {
        let mut events = self.close_reasoning();
        let text_output_index = match self.text_output_index {
            Some(text_output_index) => text_output_index,
            None => {
                let text_output_index = self.take_output_index();
                self.text_output_index = Some(text_output_index);
                let item_added_sequence_number = self.take_sequence_number();
                events.push_back(OpenAiResponseStreamEvent::OutputItemAdded {
                    sequence_number: item_added_sequence_number,
                    output_index: text_output_index,
                    item: OpenAiResponseOutputItem::message_in_progress(self.message_item_id()),
                });
                let part_added_sequence_number = self.take_sequence_number();
                events.push_back(OpenAiResponseStreamEvent::ContentPartAdded {
                    sequence_number: part_added_sequence_number,
                    item_id: self.message_item_id(),
                    output_index: text_output_index,
                    content_index: 0,
                    part: OpenAiResponseOutputContent::output_text(""),
                });
                text_output_index
            }
        };
        let sequence_number = self.take_sequence_number();
        events.push_back(OpenAiResponseStreamEvent::OutputTextDelta {
            sequence_number,
            item_id: self.message_item_id(),
            output_index: text_output_index,
            content_index: 0,
            delta: output_text_delta,
            logprobs: Vec::new(),
        });
        events
    }

    fn function_call(
        &mut self,
        tool_call_index: u16,
        function_name: String,
        arguments_json: String,
    ) -> VecDeque<OpenAiResponseStreamEvent> {
        let mut events = self.close_reasoning();
        events.extend(self.close_output_text());
        let output_index = self.take_output_index();
        let function_item_id = self.function_item_id(tool_call_index);
        let function_call_id = self.function_call_id(tool_call_index);
        let item_added_sequence_number = self.take_sequence_number();
        events.push_back(OpenAiResponseStreamEvent::OutputItemAdded {
            sequence_number: item_added_sequence_number,
            output_index,
            item: OpenAiResponseOutputItem::function_call_in_progress(
                function_item_id.clone(),
                function_call_id.clone(),
                function_name.clone(),
            ),
        });
        let delta_sequence_number = self.take_sequence_number();
        events.push_back(OpenAiResponseStreamEvent::FunctionCallArgumentsDelta {
            sequence_number: delta_sequence_number,
            item_id: function_item_id.clone(),
            output_index,
            delta: arguments_json.clone(),
        });
        let done_sequence_number = self.take_sequence_number();
        events.push_back(OpenAiResponseStreamEvent::FunctionCallArgumentsDone {
            sequence_number: done_sequence_number,
            item_id: function_item_id.clone(),
            output_index,
            name: function_name.clone(),
            arguments: arguments_json.clone(),
        });
        let item_done_sequence_number = self.take_sequence_number();
        events.push_back(OpenAiResponseStreamEvent::OutputItemDone {
            sequence_number: item_done_sequence_number,
            output_index,
            item: OpenAiResponseOutputItem::function_call(
                function_item_id,
                function_call_id,
                function_name,
                arguments_json,
            ),
        });
        events
    }

    fn complete(
        &mut self,
        prompt_token_count: u32,
        generated_token_count: u16,
        reasoning_token_count: u16,
        cached_token_count: u32,
        reason: ChatGenerationCompletionReason,
    ) -> Result<VecDeque<OpenAiResponseStreamEvent>, OpenAiResponsesStreamEncodingError> {
        let mut events = self.close_reasoning();
        events.extend(self.close_output_text());
        let collector = self
            .collector
            .take()
            .ok_or(OpenAiResponsesStreamEncodingError::AlreadyCompleted)?;
        let response = collector.into_response(
            prompt_token_count,
            generated_token_count,
            cached_token_count,
            reasoning_token_count,
            reason,
        )?;
        let sequence_number = self.take_sequence_number();
        if reason == ChatGenerationCompletionReason::MaximumOutputTokens {
            events.push_back(OpenAiResponseStreamEvent::Incomplete {
                sequence_number,
                response,
            });
        } else {
            events.push_back(OpenAiResponseStreamEvent::Completed {
                sequence_number,
                response,
            });
        }
        Ok(events)
    }

    fn close_reasoning(&mut self) -> VecDeque<OpenAiResponseStreamEvent> {
        let Some(output_index) = self.reasoning_output_index.take() else {
            return VecDeque::new();
        };
        let reasoning_item_id = self.reasoning_item_id();
        let reasoning_text = self
            .collector
            .as_ref()
            .map(OpenAiResponsesCollector::reasoning_text)
            .unwrap_or_default()
            .to_owned();
        let done_sequence_number = self.take_sequence_number();
        let item_done_sequence_number = self.take_sequence_number();
        VecDeque::from([
            OpenAiResponseStreamEvent::ReasoningSummaryTextDone {
                sequence_number: done_sequence_number,
                item_id: reasoning_item_id.clone(),
                output_index,
                summary_index: 0,
                text: reasoning_text.clone(),
            },
            OpenAiResponseStreamEvent::OutputItemDone {
                sequence_number: item_done_sequence_number,
                output_index,
                item: OpenAiResponseOutputItem::reasoning(reasoning_item_id, reasoning_text),
            },
        ])
    }

    fn close_output_text(&mut self) -> VecDeque<OpenAiResponseStreamEvent> {
        let Some(output_index) = self.text_output_index.take() else {
            return VecDeque::new();
        };
        let message_item_id = self.message_item_id();
        let output_text = self
            .collector
            .as_ref()
            .map(OpenAiResponsesCollector::output_text)
            .unwrap_or_default()
            .to_owned();
        let output_content = OpenAiResponseOutputContent::output_text(output_text.clone());
        let text_done_sequence_number = self.take_sequence_number();
        let part_done_sequence_number = self.take_sequence_number();
        let item_done_sequence_number = self.take_sequence_number();
        VecDeque::from([
            OpenAiResponseStreamEvent::OutputTextDone {
                sequence_number: text_done_sequence_number,
                item_id: message_item_id.clone(),
                output_index,
                content_index: 0,
                text: output_text.clone(),
                logprobs: Vec::new(),
            },
            OpenAiResponseStreamEvent::ContentPartDone {
                sequence_number: part_done_sequence_number,
                item_id: message_item_id.clone(),
                output_index,
                content_index: 0,
                part: output_content,
            },
            OpenAiResponseStreamEvent::OutputItemDone {
                sequence_number: item_done_sequence_number,
                output_index,
                item: OpenAiResponseOutputItem::message(message_item_id, output_text),
            },
        ])
    }

    fn worker_failure(
        &mut self,
        reason: ChatGenerationFailureReason,
    ) -> VecDeque<OpenAiResponseStreamEvent> {
        let Some(collector) = self.collector.take() else {
            return self.error_event(
                "response_generation_failed",
                "the Responses stream had already terminated",
                None,
            );
        };
        VecDeque::from([OpenAiResponseStreamEvent::Failed {
            sequence_number: self.take_sequence_number(),
            response: collector.into_failed_response(reason),
        }])
    }

    fn error_event(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        param: Option<String>,
    ) -> VecDeque<OpenAiResponseStreamEvent> {
        self.collector.take();
        let sequence_number = self.take_sequence_number();
        VecDeque::from([OpenAiResponseStreamEvent::Error {
            sequence_number,
            code: Some(code.into()),
            message: message.into(),
            param,
        }])
    }

    fn collector_mut(
        &mut self,
    ) -> Result<&mut OpenAiResponsesCollector, OpenAiResponsesStreamEncodingError> {
        self.collector
            .as_mut()
            .ok_or(OpenAiResponsesStreamEncodingError::AlreadyCompleted)
    }

    fn in_progress_response(&self) -> OpenAiResponse {
        OpenAiResponse::in_progress(
            self.response_id.clone(),
            self.created_at_unix_seconds,
            self.model_id.clone(),
            self.instructions.clone(),
        )
        .with_request_configuration(&self.request_configuration)
    }

    fn take_sequence_number(&mut self) -> u64 {
        let sequence_number = self.next_sequence_number;
        self.next_sequence_number = self.next_sequence_number.saturating_add(1);
        sequence_number
    }

    fn take_output_index(&mut self) -> usize {
        let output_index = self.next_output_index;
        self.next_output_index = self.next_output_index.saturating_add(1);
        output_index
    }

    fn identifier_suffix(&self) -> &str {
        self.response_id
            .strip_prefix("resp_")
            .unwrap_or(&self.response_id)
    }

    fn reasoning_item_id(&self) -> String {
        format!("rs_{}", self.identifier_suffix())
    }

    fn message_item_id(&self) -> String {
        format!("msg_{}", self.identifier_suffix())
    }

    fn function_item_id(&self, tool_call_index: u16) -> String {
        format!("fc_{}-{tool_call_index}", self.identifier_suffix())
    }

    fn function_call_id(&self, tool_call_index: u16) -> String {
        format!("call_{}-{tool_call_index}", self.identifier_suffix())
    }
}

#[derive(Debug, Error)]
pub enum OpenAiResponsesStreamEncodingError {
    #[error("the Responses stream has already completed")]
    AlreadyCompleted,
    #[error("failed to assemble the Responses stream: {0}")]
    Assembly(#[from] OpenAiResponsesAssemblyError),
}
