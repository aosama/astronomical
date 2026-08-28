//! Salvage Qwen3.5 tool calls when generation ends or a memory bound is crossed.
//!
//! Resource bounds exist to stop unbounded buffering. Aborting the stream as malformed
//! output drops a usable function name the coding client could reject or retry.

use super::super::output_parser_error::Qwen3_5OutputParserError;
use super::{Qwen3_5OutputEvent, Qwen3_5OutputParser, Qwen3_5OutputParserState};

impl Qwen3_5OutputParser {
    pub(super) fn salvage_unclosed_tool_call(&mut self) -> Vec<Qwen3_5OutputEvent> {
        let tool_call_body = std::mem::take(&mut self.pending_output);
        self.state = Qwen3_5OutputParserState::Text;
        match self.fail_open_closed_tool_call(&tool_call_body) {
            Some(salvaged_event) => {
                vec![self.emit_tool_call_or_visible_text(salvaged_event, tool_call_body)]
            }
            None => Vec::new(),
        }
    }

    pub(super) fn emit_tool_call_or_visible_text(
        &mut self,
        salvaged_event: Qwen3_5OutputEvent,
        tool_call_body: String,
    ) -> Qwen3_5OutputEvent {
        match salvaged_event {
            Qwen3_5OutputEvent::ToolCall(_) => {
                if self.try_record_completed_tool_call() {
                    salvaged_event
                } else {
                    Qwen3_5OutputEvent::TextDelta(tool_call_body)
                }
            }
            other_event => other_event,
        }
    }

    pub(super) fn try_record_completed_tool_call(&mut self) -> bool {
        // OpenAI tool_call.index is u16. Overflowing that width must not abort generation.
        let Some(next_completed_tool_call_count) = self.completed_tool_call_count.checked_add(1)
        else {
            return false;
        };
        self.completed_tool_call_count = next_completed_tool_call_count;
        true
    }
}

pub(super) fn log_fail_open_closed_tool_call(
    parser_error: &Qwen3_5OutputParserError,
    tool_call_body: &str,
    fail_open_event: Option<&Qwen3_5OutputEvent>,
) {
    let (function_name, forwarded_arguments_json) = match fail_open_event {
        Some(Qwen3_5OutputEvent::ToolCall(tool_call)) => (
            tool_call.function_name.as_str(),
            tool_call.arguments_json.as_str(),
        ),
        _ => ("", ""),
    };
    tracing::warn!(
        diagnostic_code = parser_error.diagnostic_code(),
        parser_error = %parser_error,
        function_name,
        forwarded_arguments_json,
        closed_tool_call_body = bounded_fail_open_log_body(tool_call_body),
        "fail-open closed Qwen3.5 tool call"
    );
}

fn bounded_fail_open_log_body(tool_call_body: &str) -> &str {
    let maximum_log_bytes = 256.min(tool_call_body.len());
    let bound = tool_call_body.floor_char_boundary(maximum_log_bytes);
    &tool_call_body[..bound]
}
