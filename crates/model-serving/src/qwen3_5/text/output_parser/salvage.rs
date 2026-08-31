//! Salvage Qwen3.5 tool calls when generation ends or a memory bound is crossed.
//!
//! Resource bounds exist to stop unbounded buffering. Aborting the stream as malformed
//! output drops a usable function name the coding client could reject or retry.

use super::super::output_parser_error::Qwen3_5OutputParserError;
use super::{
    BARE_FUNCTION_START_MARKER, FUNCTION_END_MARKER, INVOKE_END_MARKER, INVOKE_START_MARKER,
    Qwen3_5OutputEvent, Qwen3_5OutputParser, Qwen3_5OutputParserState, TOOL_CALL_END_MARKER,
    ToolCallEntryKind,
};

impl Qwen3_5OutputParser {
    pub(super) fn salvage_unclosed_tool_call(&mut self) -> Vec<Qwen3_5OutputEvent> {
        let entry = match self.state {
            Qwen3_5OutputParserState::ToolCall(entry) => entry.kind,
            _ => ToolCallEntryKind::Envelope,
        };
        let remaining_body = std::mem::take(&mut self.pending_output);
        self.state = Qwen3_5OutputParserState::Text;
        let tool_call_body = reconstruct_tool_call_body(entry, &remaining_body);
        match self.fail_open_closed_tool_call(&tool_call_body) {
            Some(salvaged_event) => {
                log_salvaged_unclosed_tool_call(
                    self.parse_tool_call(&tool_call_body).err().as_ref(),
                    &tool_call_body,
                    Some(&salvaged_event),
                );
                vec![self.emit_tool_call_or_visible_text(salvaged_event, tool_call_body)]
            }
            None => {
                log_salvaged_unclosed_tool_call(
                    self.parse_tool_call(&tool_call_body).err().as_ref(),
                    &tool_call_body,
                    None,
                );
                Vec::new()
            }
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

/// Salvaging drops the pending call from the state machine without the closed-path
/// warning, so this log is the only production signal that a generation ended with
/// an unclosed tool-call attempt.
pub(super) fn log_salvaged_unclosed_tool_call(
    parser_error: Option<&Qwen3_5OutputParserError>,
    tool_call_body: &str,
    salvaged_event: Option<&Qwen3_5OutputEvent>,
) {
    let (function_name, forwarded_arguments_json) = match salvaged_event {
        Some(Qwen3_5OutputEvent::ToolCall(tool_call)) => (
            tool_call.function_name.as_str(),
            tool_call.arguments_json.as_str(),
        ),
        _ => ("", ""),
    };
    tracing::warn!(
        diagnostic_code = parser_error
            .map(Qwen3_5OutputParserError::diagnostic_code)
            .unwrap_or("unclosed_tool_call"),
        function_name,
        forwarded_arguments_json,
        unclosed_tool_call_body = bounded_fail_open_log_body(tool_call_body),
        "salvaged unclosed Qwen3.5 tool call"
    );
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

pub(super) fn reconstruct_tool_call_body(entry: ToolCallEntryKind, remaining_body: &str) -> String {
    match entry {
        ToolCallEntryKind::Envelope => remaining_body.to_owned(),
        ToolCallEntryKind::BareFunction => {
            let mut reconstructed =
                String::with_capacity(BARE_FUNCTION_START_MARKER.len() + remaining_body.len());
            reconstructed.push_str(BARE_FUNCTION_START_MARKER);
            reconstructed.push_str(remaining_body);
            reconstructed
        }
        ToolCallEntryKind::InvokeTag => {
            let mut reconstructed =
                String::with_capacity(INVOKE_START_MARKER.len() + remaining_body.len());
            reconstructed.push_str(INVOKE_START_MARKER);
            reconstructed.push_str(remaining_body);
            reconstructed
        }
    }
}

pub(super) fn tool_call_end_markers(entry: ToolCallEntryKind) -> &'static [&'static str] {
    match entry {
        ToolCallEntryKind::Envelope => &[TOOL_CALL_END_MARKER],
        ToolCallEntryKind::BareFunction | ToolCallEntryKind::InvokeTag => {
            &[TOOL_CALL_END_MARKER, FUNCTION_END_MARKER, INVOKE_END_MARKER]
        }
    }
}
