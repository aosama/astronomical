//! Salvage Poolside tool calls when generation ends or a memory bound is crossed.
//!
//! Resource bounds exist to stop unbounded buffering. Aborting the stream as malformed
//! output drops a usable function name the coding client could reject or retry.

use super::super::LagunaOutputParserError;
use super::{
    ARGUMENT_VALUE_END_MARKER, ARGUMENT_VALUE_START_MARKER, LagunaOutputEvent, LagunaOutputParser,
    LagunaOutputParserState,
};

impl LagunaOutputParser {
    pub(super) fn salvage_unclosed_tool_call(&mut self) -> Vec<LagunaOutputEvent> {
        let tool_call_body = std::mem::take(&mut self.pending_output);
        self.state = LagunaOutputParserState::Text;
        match self.fail_open_closed_tool_call(&tool_call_body) {
            Some(salvaged_event) => {
                vec![self.emit_tool_call_or_visible_text(salvaged_event, tool_call_body)]
            }
            None => Vec::new(),
        }
    }

    pub(super) fn emit_tool_call_or_visible_text(
        &mut self,
        salvaged_event: LagunaOutputEvent,
        tool_call_body: String,
    ) -> LagunaOutputEvent {
        match salvaged_event {
            LagunaOutputEvent::ToolCall { .. } => {
                if self.try_record_completed_tool_call() {
                    salvaged_event
                } else {
                    LagunaOutputEvent::TextDelta(tool_call_body)
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

    pub(super) fn open_tool_argument_exceeds_bound(&self) -> bool {
        let Some(value_start_offset) = self.pending_output.rfind(ARGUMENT_VALUE_START_MARKER)
        else {
            return false;
        };
        let value_content_offset = value_start_offset + ARGUMENT_VALUE_START_MARKER.len();
        if self.pending_output[value_content_offset..].contains(ARGUMENT_VALUE_END_MARKER) {
            return false;
        }
        let actual_bytes = self
            .pending_output
            .len()
            .saturating_sub(value_content_offset);
        actual_bytes > Self::MAXIMUM_TOOL_ARGUMENT_BYTES
    }
}

pub(super) fn log_fail_open_closed_tool_call(
    parser_error: &LagunaOutputParserError,
    tool_call_body: &str,
    fail_open_event: Option<&LagunaOutputEvent>,
) {
    let (function_name, forwarded_arguments_json) = match fail_open_event {
        Some(LagunaOutputEvent::ToolCall {
            function_name,
            arguments_json,
            ..
        }) => (function_name.as_str(), arguments_json.as_str()),
        _ => ("", ""),
    };
    tracing::warn!(
        diagnostic_code = parser_error.diagnostic_code(),
        parser_error = %parser_error,
        function_name,
        forwarded_arguments_json,
        closed_tool_call_body = bounded_fail_open_log_body(tool_call_body),
        "fail-open closed Laguna tool call"
    );
}

fn bounded_fail_open_log_body(tool_call_body: &str) -> &str {
    let maximum_log_bytes = 256.min(tool_call_body.len());
    let bound = tool_call_body.floor_char_boundary(maximum_log_bytes);
    &tool_call_body[..bound]
}
