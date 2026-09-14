//! The output parser's state-machine vocabulary and marker-scanning helpers.
//!
//! The parser advances through text, reasoning, tool-call, and suppressed-
//! reasoning states; this module owns the state types and the shared scanning
//! primitives so the driver in the parent module stays focused on transitions.

use super::super::output_parser_error::Qwen3_5OutputParserError;
use super::{BARE_FUNCTION_START_MARKER, FUNCTION_END_MARKER};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen3_5OutputParserState {
    Text,
    Reasoning,
    ToolCall(ToolCallEntry),
    SuppressedLateReasoning,
}

/// One entered tool-call attempt: its dialect kind, the exact opener marker to
/// restore when the attempt turns out to be quoted prose, and the channel the
/// opener appeared in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ToolCallEntry {
    pub(super) kind: ToolCallEntryKind,
    pub(super) opener: &'static str,
    pub(super) opened_in_reasoning: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ToolCallEntryKind {
    Envelope,
    BareFunction,
    InvokeTag,
}

impl Qwen3_5OutputParserState {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Reasoning => "reasoning",
            Self::ToolCall(_) => "tool_call",
            Self::SuppressedLateReasoning => "suppressed_late_reasoning",
        }
    }

    pub(super) fn requires_marker_scan_pending_output_cap(self) -> bool {
        matches!(
            self,
            Self::Text | Self::Reasoning | Self::SuppressedLateReasoning
        )
    }
}

pub(super) fn split_qwen_function_envelope(
    tool_call_body: &str,
) -> Result<(String, String), Qwen3_5OutputParserError> {
    // Closed envelopes still reach the harness when the model drops `<` or the function close tag.
    let after_function_open = strip_qwen_function_open(tool_call_body)
        .ok_or(Qwen3_5OutputParserError::ToolCallMissingFunction)?;
    let function_name_end = after_function_open
        .find(|character: char| character == '>' || character == '<' || character.is_whitespace())
        .unwrap_or(after_function_open.len());
    let function_name = after_function_open[..function_name_end].trim();
    if function_name.is_empty() {
        return Err(Qwen3_5OutputParserError::ToolCallMissingFunction);
    }
    let after_function_name = after_function_open[function_name_end..]
        .trim_start_matches('>')
        .trim();
    let parameter_content = after_function_name
        .strip_suffix(FUNCTION_END_MARKER)
        .unwrap_or(after_function_name)
        .trim()
        .to_owned();
    Ok((function_name.to_owned(), parameter_content))
}

pub(super) fn strip_qwen_function_open(tool_call_body: &str) -> Option<&str> {
    tool_call_body
        .strip_prefix(BARE_FUNCTION_START_MARKER)
        .or_else(|| tool_call_body.strip_prefix("function="))
}

pub(super) fn earliest_marker<'a>(text: &str, markers: &'a [&'a str]) -> Option<(usize, &'a str)> {
    markers
        .iter()
        .filter_map(|marker| {
            text.find(marker)
                .map(|marker_index| (marker_index, *marker))
        })
        .min_by_key(|(marker_index, _)| *marker_index)
}

pub(super) fn longest_suffix_prefix_for_markers(text: &str, markers: &[&str]) -> usize {
    let maximum_prefix_bytes = markers
        .iter()
        .map(|marker| marker.len().saturating_sub(1))
        .max()
        .unwrap_or(0)
        .min(text.len());
    for suffix_bytes in (1..=maximum_prefix_bytes).rev() {
        let suffix_start = text.len() - suffix_bytes;
        if !text.is_char_boundary(suffix_start) {
            continue;
        }
        let suffix = &text[suffix_start..];
        if markers.iter().any(|marker| marker.starts_with(suffix)) {
            return suffix_bytes;
        }
    }
    0
}
