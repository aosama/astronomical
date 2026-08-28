//! Incremental Qwen3.5 output parser for reasoning, text, and tool calls.
//!
//! Dense and mixture-of-experts share this parser. Tool-call defects fail open so a
//! coding client can reject or retry. Incomplete control-marker prefixes flush as
//! visible text at generation end instead of aborting the stream.

use std::collections::BTreeMap;

use astronomical_ipc_protocol::ChatToolDefinition;
use serde_json::Value;

use super::output_parser_error::Qwen3_5OutputParserError;
use super::tool_schema::{DeclaredTool, parse_tool_parameters};

mod foreign_syntax;
mod salvage;

const MAX_OUTPUT_FRAGMENT_BYTES: usize = 16 * 1024;
const MAX_MARKER_SCAN_PENDING_OUTPUT_BYTES: usize = 128 * 1024;
const THINK_END_MARKER: &str = concat!("<", "/think", ">");
const THINK_START_MARKER: &str = concat!("<", "think", ">");
pub(super) const TOOL_CALL_END_MARKER: &str = concat!("<", "/tool_call", ">");
const TOOL_CALL_START_MARKER: &str = concat!("<", "tool_call", ">");
pub(super) const BARE_FUNCTION_START_MARKER: &str = concat!("<", "function=");
pub(super) const FUNCTION_END_MARKER: &str = concat!("<", "/function", ">");
pub(super) const INVOKE_START_MARKER: &str = concat!("<", "invoke");
pub(super) const INVOKE_END_MARKER: &str = concat!("<", "/invoke", ">");

const TEXT_STATE_START_MARKERS: &[&str] = &[
    THINK_START_MARKER,
    TOOL_CALL_START_MARKER,
    BARE_FUNCTION_START_MARKER,
    INVOKE_START_MARKER,
];

// mlx-lm transitions from reasoning to tool on a tool-call start. Ornith often
// writes the call inside the still-open think channel; scanning only for the
// think end marker would dump that call as reasoning text and end the turn.
const REASONING_SCAN_MARKERS: &[&str] = &[
    THINK_END_MARKER,
    TOOL_CALL_START_MARKER,
    BARE_FUNCTION_START_MARKER,
    INVOKE_START_MARKER,
];

/// A model-output event translated from Qwen3.5 syntax into neutral structured output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen3_5OutputEvent {
    /// Reasoning content with think markers removed.
    ReasoningDelta(String),
    /// Normal assistant response content with control markers removed.
    TextDelta(String),
    /// One complete function call, including well-formed names the request did not declare.
    ToolCall(Qwen3_5ToolCall),
    /// A correction that must be injected back into the model context, not streamed to the client.
    ModelVisibleCorrection { correction_text: String },
}

/// One validated function call parsed from a complete Qwen3.5 XML block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5ToolCall {
    /// The zero-based output order of this function call.
    pub index: u16,
    /// The selected function name.
    pub function_name: String,
    /// Canonical JSON object arguments for the selected function.
    pub arguments_json: String,
}

/// Bounded incremental parser for Qwen3.5 reasoning and XML-style function output.
#[derive(Debug)]
pub struct Qwen3_5OutputParser {
    completed_tool_call_count: u16,
    declared_tools: BTreeMap<String, DeclaredTool>,
    has_streamed_visible_text: bool,
    pending_output: String,
    state: Qwen3_5OutputParserState,
}

impl Qwen3_5OutputParser {
    /// Creates an output parser from the exact tools declared for one request.
    pub fn new(declared_tools: &[ChatToolDefinition]) -> Result<Self, Qwen3_5OutputParserError> {
        Self::new_with_state(declared_tools, Qwen3_5OutputParserState::Text)
    }

    /// Creates a parser for generation that continues after the prompt's think prefix.
    pub fn new_after_thinking_prefix(
        declared_tools: &[ChatToolDefinition],
    ) -> Result<Self, Qwen3_5OutputParserError> {
        Self::new_with_state(declared_tools, Qwen3_5OutputParserState::Reasoning)
    }

    fn new_with_state(
        declared_tools: &[ChatToolDefinition],
        initial_state: Qwen3_5OutputParserState,
    ) -> Result<Self, Qwen3_5OutputParserError> {
        let mut tools_by_name = BTreeMap::new();
        for tool_definition in declared_tools {
            let declared_tool = DeclaredTool::from_definition(tool_definition)?;
            if tools_by_name
                .insert(tool_definition.name.clone(), declared_tool)
                .is_some()
            {
                return Err(Qwen3_5OutputParserError::DuplicateDeclaredTool {
                    function_name: tool_definition.name.clone(),
                });
            }
        }
        Ok(Self {
            completed_tool_call_count: 0,
            declared_tools: tools_by_name,
            has_streamed_visible_text: false,
            pending_output: String::new(),
            state: initial_state,
        })
    }

    /// Processes one decoded text fragment and emits only stable structured content.
    pub fn push_fragment(
        &mut self,
        decoded_fragment: &str,
    ) -> Result<Vec<Qwen3_5OutputEvent>, Qwen3_5OutputParserError> {
        if decoded_fragment.len() > MAX_OUTPUT_FRAGMENT_BYTES {
            if matches!(self.state, Qwen3_5OutputParserState::ToolCall(_)) {
                // The fragment cannot be buffered. Salvage the pending call so a usable
                // name still reaches the harness instead of aborting generation.
                return Ok(self.salvage_unclosed_tool_call());
            }
            let mut output_events = self.flush_pending_as_visible_delta();
            if let Some(fragment_event) =
                self.visible_delta_for_current_state(decoded_fragment.to_owned())
            {
                output_events.push(fragment_event);
            }
            return Ok(output_events);
        }
        let pending_output_bytes = self
            .pending_output
            .len()
            .saturating_add(decoded_fragment.len());
        if matches!(self.state, Qwen3_5OutputParserState::ToolCall(_))
            && self
                .pending_output
                .len()
                .checked_add(decoded_fragment.len())
                .is_none()
        {
            return Ok(self.salvage_unclosed_tool_call());
        }
        if self.state.requires_marker_scan_pending_output_cap()
            && pending_output_bytes > MAX_MARKER_SCAN_PENDING_OUTPUT_BYTES
        {
            // Drain retained text so memory stays bounded. Generation continues.
            let mut output_events = self.flush_pending_as_visible_delta();
            self.pending_output.push_str(decoded_fragment);
            while self.advance(&mut output_events)? {}
            return Ok(output_events);
        }
        self.pending_output.push_str(decoded_fragment);

        let mut output_events = Vec::new();
        while self.advance(&mut output_events)? {}
        Ok(output_events)
    }

    /// Completes the stream. Unclosed tool calls salvage; leftover marker prefixes flush as text.
    pub fn finish(&mut self) -> Result<Vec<Qwen3_5OutputEvent>, Qwen3_5OutputParserError> {
        match self.state {
            Qwen3_5OutputParserState::Text => Ok(self.flush_pending_as_visible_delta()),
            Qwen3_5OutputParserState::Reasoning => Ok(self.flush_pending_as_visible_delta()),
            Qwen3_5OutputParserState::ToolCall(_) => Ok(self.salvage_unclosed_tool_call()),
            Qwen3_5OutputParserState::SuppressedLateReasoning => {
                self.pending_output.clear();
                Ok(Vec::new())
            }
        }
    }

    fn advance(
        &mut self,
        output_events: &mut Vec<Qwen3_5OutputEvent>,
    ) -> Result<bool, Qwen3_5OutputParserError> {
        match self.state {
            Qwen3_5OutputParserState::Text => self.advance_text(output_events),
            Qwen3_5OutputParserState::Reasoning => self.advance_reasoning(output_events),
            Qwen3_5OutputParserState::ToolCall(entry) => {
                self.advance_tool_call(output_events, entry)
            }
            Qwen3_5OutputParserState::SuppressedLateReasoning => {
                self.advance_suppressed_late_reasoning()
            }
        }
    }

    fn advance_text(
        &mut self,
        output_events: &mut Vec<Qwen3_5OutputEvent>,
    ) -> Result<bool, Qwen3_5OutputParserError> {
        let next_marker = earliest_marker(&self.pending_output, TEXT_STATE_START_MARKERS);
        if let Some((marker_index, marker)) = next_marker {
            if marker_index > 0 {
                self.has_streamed_visible_text = true;
                output_events.push(Qwen3_5OutputEvent::TextDelta(
                    self.take_pending_prefix(marker_index),
                ));
                return Ok(true);
            }
            self.take_pending_prefix(marker.len());
            self.enter_state_after_start_marker(marker);
            return Ok(true);
        }

        let stable_text_bytes =
            self.pending_output
                .len()
                .saturating_sub(longest_suffix_prefix_for_markers(
                    &self.pending_output,
                    TEXT_STATE_START_MARKERS,
                ));
        if stable_text_bytes == 0 {
            return Ok(false);
        }
        self.has_streamed_visible_text = true;
        output_events.push(Qwen3_5OutputEvent::TextDelta(
            self.take_pending_prefix(stable_text_bytes),
        ));
        Ok(true)
    }

    fn advance_reasoning(
        &mut self,
        output_events: &mut Vec<Qwen3_5OutputEvent>,
    ) -> Result<bool, Qwen3_5OutputParserError> {
        let next_marker = earliest_marker(&self.pending_output, REASONING_SCAN_MARKERS);
        if let Some((marker_index, marker)) = next_marker {
            if marker_index > 0 {
                output_events.push(Qwen3_5OutputEvent::ReasoningDelta(
                    self.take_pending_prefix(marker_index),
                ));
                return Ok(true);
            }
            self.take_pending_prefix(marker.len());
            if marker == THINK_END_MARKER {
                self.state = Qwen3_5OutputParserState::Text;
            } else {
                self.enter_state_after_start_marker(marker);
            }
            return Ok(true);
        }

        let stable_reasoning_bytes =
            self.pending_output
                .len()
                .saturating_sub(longest_suffix_prefix_for_markers(
                    &self.pending_output,
                    REASONING_SCAN_MARKERS,
                ));
        if stable_reasoning_bytes == 0 {
            return Ok(false);
        }
        output_events.push(Qwen3_5OutputEvent::ReasoningDelta(
            self.take_pending_prefix(stable_reasoning_bytes),
        ));
        Ok(true)
    }

    fn advance_suppressed_late_reasoning(&mut self) -> Result<bool, Qwen3_5OutputParserError> {
        if let Some(marker_index) = self.pending_output.find(THINK_END_MARKER) {
            let hidden_reasoning_and_marker_bytes = marker_index + THINK_END_MARKER.len();
            self.take_pending_prefix(hidden_reasoning_and_marker_bytes);
            self.state = Qwen3_5OutputParserState::Text;
            return Ok(true);
        }

        let stable_hidden_reasoning_bytes =
            self.pending_output
                .len()
                .saturating_sub(longest_suffix_prefix_for_markers(
                    &self.pending_output,
                    &[THINK_END_MARKER],
                ));
        if stable_hidden_reasoning_bytes == 0 {
            return Ok(false);
        }
        self.take_pending_prefix(stable_hidden_reasoning_bytes);
        Ok(true)
    }

    fn advance_tool_call(
        &mut self,
        output_events: &mut Vec<Qwen3_5OutputEvent>,
        entry: ToolCallEntryKind,
    ) -> Result<bool, Qwen3_5OutputParserError> {
        let Some((marker_index, end_marker)) =
            earliest_marker(&self.pending_output, salvage::tool_call_end_markers(entry))
        else {
            return Ok(false);
        };
        let remaining_body = self.take_pending_prefix(marker_index);
        self.take_pending_prefix(end_marker.len());
        if end_marker != TOOL_CALL_END_MARKER {
            self.consume_trailing_envelope_close_if_present();
        }
        let reconstructed_body = salvage::reconstruct_tool_call_body(entry, &remaining_body);
        // Closed envelopes fail open: the harness owns retries.
        match self.parse_tool_call(&reconstructed_body) {
            Ok(tool_call) => {
                output_events.push(self.emit_tool_call_or_visible_text(
                    Qwen3_5OutputEvent::ToolCall(tool_call),
                    reconstructed_body,
                ));
            }
            Err(parser_error) => {
                let fail_open_event = self.fail_open_closed_tool_call(&reconstructed_body);
                salvage::log_fail_open_closed_tool_call(
                    &parser_error,
                    &reconstructed_body,
                    fail_open_event.as_ref(),
                );
                if let Some(fail_open_event) = fail_open_event {
                    output_events.push(
                        self.emit_tool_call_or_visible_text(fail_open_event, reconstructed_body),
                    );
                }
            }
        }
        self.state = Qwen3_5OutputParserState::Text;
        Ok(true)
    }

    fn parse_tool_call(
        &self,
        tool_call_body: &str,
    ) -> Result<Qwen3_5ToolCall, Qwen3_5OutputParserError> {
        let normalized_body = foreign_syntax::normalize_foreign_tool_call_syntax(tool_call_body);
        let (function_name, parameter_content) =
            split_qwen_function_envelope(normalized_body.trim())?;
        // Unknown names and sloppy-but-closed XML are forwarded so the harness
        // can return "no such tool" and the model can retry.
        let parsed_arguments = parse_tool_parameters(
            &parameter_content,
            self.declared_tools.get(function_name.as_str()),
        )?;
        let arguments_json = serde_json::to_string(&Value::Object(parsed_arguments))
            .map_err(Qwen3_5OutputParserError::SerializeToolArguments)?;
        Ok(Qwen3_5ToolCall {
            index: self.completed_tool_call_count,
            function_name: function_name.to_owned(),
            arguments_json,
        })
    }

    fn fail_open_closed_tool_call(&self, tool_call_body: &str) -> Option<Qwen3_5OutputEvent> {
        let normalized_body = foreign_syntax::normalize_foreign_tool_call_syntax(tool_call_body);
        let Ok((function_name, parameter_content)) =
            split_qwen_function_envelope(normalized_body.trim())
        else {
            if tool_call_body.is_empty() {
                return None;
            }
            return Some(Qwen3_5OutputEvent::TextDelta(tool_call_body.to_owned()));
        };
        // Declared-schema rejection would still abort. Passthrough lets the harness
        // return invalid-argument or unknown-tool instead of killing the stream.
        let parsed_arguments = parse_tool_parameters(&parameter_content, None).unwrap_or_default();
        let arguments_json = serde_json::to_string(&Value::Object(parsed_arguments))
            .unwrap_or_else(|_| "{}".to_owned());
        Some(Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: self.completed_tool_call_count,
            function_name,
            arguments_json,
        }))
    }

    fn enter_state_after_start_marker(&mut self, marker: &str) {
        self.state = match marker {
            THINK_START_MARKER => {
                if self.has_streamed_visible_text {
                    Qwen3_5OutputParserState::SuppressedLateReasoning
                } else {
                    Qwen3_5OutputParserState::Reasoning
                }
            }
            TOOL_CALL_START_MARKER => {
                Qwen3_5OutputParserState::ToolCall(ToolCallEntryKind::Envelope)
            }
            BARE_FUNCTION_START_MARKER => {
                Qwen3_5OutputParserState::ToolCall(ToolCallEntryKind::BareFunction)
            }
            INVOKE_START_MARKER => Qwen3_5OutputParserState::ToolCall(ToolCallEntryKind::InvokeTag),
            _ => Qwen3_5OutputParserState::Text,
        };
    }

    fn take_pending_prefix(&mut self, byte_count: usize) -> String {
        self.pending_output.drain(..byte_count).collect()
    }

    fn consume_trailing_envelope_close_if_present(&mut self) {
        let trimmed = self.pending_output.trim_start();
        let Some(after_marker) = trimmed.strip_prefix(TOOL_CALL_END_MARKER) else {
            return;
        };
        let consumed_bytes = self.pending_output.len() - after_marker.len();
        self.take_pending_prefix(consumed_bytes);
    }

    fn flush_pending_as_visible_delta(&mut self) -> Vec<Qwen3_5OutputEvent> {
        if self.pending_output.is_empty() {
            return Vec::new();
        }
        let pending_output = std::mem::take(&mut self.pending_output);
        self.visible_delta_for_current_state(pending_output)
            .into_iter()
            .collect()
    }

    fn visible_delta_for_current_state(&mut self, text: String) -> Option<Qwen3_5OutputEvent> {
        match self.state {
            Qwen3_5OutputParserState::Text | Qwen3_5OutputParserState::ToolCall(_) => {
                self.has_streamed_visible_text = true;
                Some(Qwen3_5OutputEvent::TextDelta(text))
            }
            Qwen3_5OutputParserState::Reasoning => Some(Qwen3_5OutputEvent::ReasoningDelta(text)),
            Qwen3_5OutputParserState::SuppressedLateReasoning => None,
        }
    }

    pub(crate) fn state_for_diagnostics(&self) -> &'static str {
        self.state.as_str()
    }

    pub(crate) fn pending_output_for_diagnostics(&self) -> &str {
        &self.pending_output
    }

    pub(crate) fn reset_after_model_visible_correction(&mut self, enable_thinking: bool) {
        self.pending_output.clear();
        self.has_streamed_visible_text = false;
        self.state = if enable_thinking {
            Qwen3_5OutputParserState::Reasoning
        } else {
            Qwen3_5OutputParserState::Text
        };
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen3_5OutputParserState {
    Text,
    Reasoning,
    ToolCall(ToolCallEntryKind),
    SuppressedLateReasoning,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ToolCallEntryKind {
    Envelope,
    BareFunction,
    InvokeTag,
}

impl Qwen3_5OutputParserState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Reasoning => "reasoning",
            Self::ToolCall(_) => "tool_call",
            Self::SuppressedLateReasoning => "suppressed_late_reasoning",
        }
    }

    fn requires_marker_scan_pending_output_cap(self) -> bool {
        matches!(
            self,
            Self::Text | Self::Reasoning | Self::SuppressedLateReasoning
        )
    }
}

fn split_qwen_function_envelope(
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

fn strip_qwen_function_open(tool_call_body: &str) -> Option<&str> {
    tool_call_body
        .strip_prefix(BARE_FUNCTION_START_MARKER)
        .or_else(|| tool_call_body.strip_prefix("function="))
}

fn earliest_marker<'a>(text: &str, markers: &'a [&'a str]) -> Option<(usize, &'a str)> {
    markers
        .iter()
        .filter_map(|marker| {
            text.find(marker)
                .map(|marker_index| (marker_index, *marker))
        })
        .min_by_key(|(marker_index, _)| *marker_index)
}

fn longest_suffix_prefix_for_markers(text: &str, markers: &[&str]) -> usize {
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
