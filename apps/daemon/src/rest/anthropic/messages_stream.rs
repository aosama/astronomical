//! Encodes ordered worker events into the Anthropic Messages SSE sequence:
//! `message_start`, then per content block `content_block_start`, its
//! deltas, and `content_block_stop`, then `message_delta` and
//! `message_stop`. At most one thinking or text block is open at a time;
//! a tool call always arrives complete, so its block opens and closes at once.

use std::collections::VecDeque;

use astronomical_ipc_protocol::ChatGenerationCompletionReason;
use astronomical_rest_contract::{
    AnthropicContentBlockDelta, AnthropicError, AnthropicMessage, AnthropicMessageDeltaBody,
    AnthropicMessageStreamEvent, AnthropicOutputContentBlock,
};

use super::messages_assembly::{
    WorkerTokenCounts, anthropic_stop_reason, failure_error, parse_tool_input, tool_use_id,
};
use crate::{ChatGenerationStreamErrorCode, ChatGenerationStreamEvent};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenBlockKind {
    Thinking,
    Text,
}

#[derive(Clone, Copy, Debug)]
struct OpenContentBlock {
    index: usize,
    kind: OpenBlockKind,
}

pub(crate) struct AnthropicMessagesStreamEncoder {
    message_id: String,
    model_id: String,
    next_block_index: usize,
    open_block: Option<OpenContentBlock>,
    is_terminal: bool,
}

impl AnthropicMessagesStreamEncoder {
    pub(crate) fn new(message_id: String, model_id: String) -> Self {
        Self {
            message_id,
            model_id,
            next_block_index: 0,
            open_block: None,
            is_terminal: false,
        }
    }

    /// Usage is unknown until the worker completes, so `message_start`
    /// carries zero counts and `message_delta` carries the final ones.
    pub(crate) fn initial_events(&self) -> VecDeque<AnthropicMessageStreamEvent> {
        VecDeque::from([AnthropicMessageStreamEvent::MessageStart {
            message: AnthropicMessage::empty(self.message_id.clone(), self.model_id.clone()),
        }])
    }

    pub(crate) fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    pub(crate) fn encode(
        &mut self,
        stream_event: ChatGenerationStreamEvent,
    ) -> VecDeque<AnthropicMessageStreamEvent> {
        match stream_event {
            ChatGenerationStreamEvent::ReasoningFragment(thinking) => self.block_delta(
                OpenBlockKind::Thinking,
                AnthropicContentBlockDelta::ThinkingDelta { thinking },
            ),
            ChatGenerationStreamEvent::TextFragment(text) => self.block_delta(
                OpenBlockKind::Text,
                AnthropicContentBlockDelta::TextDelta { text },
            ),
            ChatGenerationStreamEvent::ToolCall {
                tool_call_index,
                function_name,
                arguments_json,
            } => self.tool_use_block(tool_call_index, function_name, arguments_json),
            ChatGenerationStreamEvent::PrefillProgress { .. } => VecDeque::new(),
            terminal_event => self.encode_terminal(terminal_event),
        }
    }

    fn encode_terminal(
        &mut self,
        terminal_event: ChatGenerationStreamEvent,
    ) -> VecDeque<AnthropicMessageStreamEvent> {
        match terminal_event {
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                reason,
                ..
            } => {
                let worker_token_counts = WorkerTokenCounts {
                    prompt_token_count,
                    cached_token_count,
                    generated_token_count,
                };
                self.complete(worker_token_counts, reason)
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                self.error_event(failure_error(&reason))
            }
            ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable) => {
                self.error_event(AnthropicError::api_error(
                    "the local worker became unavailable while generating the message",
                ))
            }
            // `encode` routes every non-terminal event before it gets here.
            ChatGenerationStreamEvent::ReasoningFragment(_)
            | ChatGenerationStreamEvent::TextFragment(_)
            | ChatGenerationStreamEvent::ToolCall { .. }
            | ChatGenerationStreamEvent::PrefillProgress { .. } => VecDeque::new(),
        }
    }

    fn block_delta(
        &mut self,
        block_kind: OpenBlockKind,
        delta: AnthropicContentBlockDelta,
    ) -> VecDeque<AnthropicMessageStreamEvent> {
        let mut stream_events = VecDeque::new();
        let block_index = match self.open_block {
            Some(open_block) if open_block.kind == block_kind => open_block.index,
            _ => {
                stream_events.extend(self.close_open_block());
                let block_index = self.take_block_index();
                self.open_block = Some(OpenContentBlock {
                    index: block_index,
                    kind: block_kind,
                });
                stream_events.push_back(AnthropicMessageStreamEvent::ContentBlockStart {
                    index: block_index,
                    content_block: empty_content_block(block_kind),
                });
                block_index
            }
        };
        stream_events.push_back(AnthropicMessageStreamEvent::ContentBlockDelta {
            index: block_index,
            delta,
        });
        stream_events
    }

    fn tool_use_block(
        &mut self,
        tool_call_index: u16,
        function_name: String,
        arguments_json: String,
    ) -> VecDeque<AnthropicMessageStreamEvent> {
        // A malformed argument document is a worker contract failure; the
        // client must not receive a tool call it cannot parse.
        if let Err(assembly_error) = parse_tool_input(&arguments_json) {
            return self.error_event(AnthropicError::api_error(assembly_error.to_string()));
        }
        let mut stream_events = self.close_open_block();
        let block_index = self.take_block_index();
        stream_events.push_back(AnthropicMessageStreamEvent::ContentBlockStart {
            index: block_index,
            content_block: AnthropicOutputContentBlock::ToolUse {
                id: tool_use_id(&self.message_id, tool_call_index),
                name: function_name,
                input: serde_json::json!({}),
            },
        });
        stream_events.push_back(AnthropicMessageStreamEvent::ContentBlockDelta {
            index: block_index,
            delta: AnthropicContentBlockDelta::InputJsonDelta {
                partial_json: arguments_json,
            },
        });
        stream_events
            .push_back(AnthropicMessageStreamEvent::ContentBlockStop { index: block_index });
        stream_events
    }

    fn complete(
        &mut self,
        worker_token_counts: WorkerTokenCounts,
        completion_reason: ChatGenerationCompletionReason,
    ) -> VecDeque<AnthropicMessageStreamEvent> {
        let Ok(stop_reason) = anthropic_stop_reason(completion_reason) else {
            return self.error_event(AnthropicError::api_error(
                "the message request was cancelled",
            ));
        };
        let mut stream_events = self.close_open_block();
        stream_events.push_back(AnthropicMessageStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason,
                stop_sequence: None,
            },
            usage: worker_token_counts.anthropic_usage(),
        });
        stream_events.push_back(AnthropicMessageStreamEvent::MessageStop);
        self.is_terminal = true;
        stream_events
    }

    fn error_event(&mut self, error: AnthropicError) -> VecDeque<AnthropicMessageStreamEvent> {
        self.is_terminal = true;
        self.open_block = None;
        VecDeque::from([AnthropicMessageStreamEvent::Error { error }])
    }

    fn close_open_block(&mut self) -> VecDeque<AnthropicMessageStreamEvent> {
        match self.open_block.take() {
            Some(open_block) => VecDeque::from([AnthropicMessageStreamEvent::ContentBlockStop {
                index: open_block.index,
            }]),
            None => VecDeque::new(),
        }
    }

    fn take_block_index(&mut self) -> usize {
        let block_index = self.next_block_index;
        self.next_block_index = self.next_block_index.saturating_add(1);
        block_index
    }
}

fn empty_content_block(block_kind: OpenBlockKind) -> AnthropicOutputContentBlock {
    match block_kind {
        OpenBlockKind::Thinking => AnthropicOutputContentBlock::Thinking {
            thinking: String::new(),
            signature: String::new(),
        },
        OpenBlockKind::Text => AnthropicOutputContentBlock::Text {
            text: String::new(),
        },
    }
}
