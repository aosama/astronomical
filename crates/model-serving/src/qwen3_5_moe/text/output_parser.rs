use std::collections::BTreeMap;

use astronomical_ipc_protocol::ChatToolDefinition;
use serde_json::Value;

use super::output_parser_error::Qwen3_5MoEOutputParserError;
use super::tool_schema::{DeclaredTool, parse_tool_parameters};

const MAX_OUTPUT_FRAGMENT_BYTES: usize = 16 * 1024;
const MAX_MARKER_SCAN_PENDING_OUTPUT_BYTES: usize = 128 * 1024;
const MAX_TOOL_CALL_COUNT: u16 = u16::MAX;
const THINK_END_MARKER: &str = "</think>";
const THINK_START_MARKER: &str = "<think>";
const TOOL_CALL_END_MARKER: &str = "</tool_call>";
const TOOL_CALL_START_MARKER: &str = "<tool_call>";

/// A model-output event translated from Qwen3.5-MoE syntax into neutral structured output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen3_5MoEOutputEvent {
    /// Reasoning content with `<think>` markers removed.
    ReasoningDelta(String),
    /// Normal assistant response content with control markers removed.
    TextDelta(String),
    /// One fully validated function call.
    ToolCall(Qwen3_5MoEToolCall),
    /// A correction that must be injected back into the model context, not streamed to the client.
    ModelVisibleCorrection { correction_text: String },
}

/// One validated function call parsed from a complete Qwen3.5-MoE XML block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5MoEToolCall {
    /// The zero-based output order of this function call.
    pub index: u16,
    /// The selected declared function name.
    pub function_name: String,
    /// Canonical JSON object arguments for the selected function.
    pub arguments_json: String,
}

/// Bounded incremental parser for Qwen3.5-MoE reasoning and XML-style function output.
#[derive(Debug)]
pub struct Qwen3_5MoEOutputParser {
    completed_tool_call_count: u16,
    declared_tools: BTreeMap<String, DeclaredTool>,
    has_streamed_visible_text: bool,
    pending_output: String,
    state: Qwen3_5MoEOutputParserState,
}

impl Qwen3_5MoEOutputParser {
    /// Creates an output parser from the exact tools declared for one request.
    pub fn new(declared_tools: &[ChatToolDefinition]) -> Result<Self, Qwen3_5MoEOutputParserError> {
        Self::new_with_state(declared_tools, Qwen3_5MoEOutputParserState::Text)
    }

    /// Creates a parser for generation that continues after the prompt's `<think>` prefix.
    pub fn new_after_thinking_prefix(
        declared_tools: &[ChatToolDefinition],
    ) -> Result<Self, Qwen3_5MoEOutputParserError> {
        Self::new_with_state(declared_tools, Qwen3_5MoEOutputParserState::Reasoning)
    }

    fn new_with_state(
        declared_tools: &[ChatToolDefinition],
        initial_state: Qwen3_5MoEOutputParserState,
    ) -> Result<Self, Qwen3_5MoEOutputParserError> {
        let mut tools_by_name = BTreeMap::new();
        for tool_definition in declared_tools {
            let declared_tool = DeclaredTool::from_definition(tool_definition)?;
            if tools_by_name
                .insert(tool_definition.name.clone(), declared_tool)
                .is_some()
            {
                return Err(Qwen3_5MoEOutputParserError::DuplicateDeclaredTool {
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
    ) -> Result<Vec<Qwen3_5MoEOutputEvent>, Qwen3_5MoEOutputParserError> {
        if decoded_fragment.len() > MAX_OUTPUT_FRAGMENT_BYTES {
            return Err(Qwen3_5MoEOutputParserError::FragmentTooLarge {
                actual_fragment_bytes: decoded_fragment.len(),
                maximum_fragment_bytes: MAX_OUTPUT_FRAGMENT_BYTES,
            });
        }
        let pending_output_bytes = self
            .pending_output
            .len()
            .checked_add(decoded_fragment.len())
            .ok_or(Qwen3_5MoEOutputParserError::PendingOutputTooLarge {
                actual_pending_bytes: usize::MAX,
                maximum_pending_bytes: MAX_MARKER_SCAN_PENDING_OUTPUT_BYTES,
            })?;
        if self.state.requires_marker_scan_pending_output_cap()
            && pending_output_bytes > MAX_MARKER_SCAN_PENDING_OUTPUT_BYTES
        {
            return Err(Qwen3_5MoEOutputParserError::PendingOutputTooLarge {
                actual_pending_bytes: pending_output_bytes,
                maximum_pending_bytes: MAX_MARKER_SCAN_PENDING_OUTPUT_BYTES,
            });
        }
        self.pending_output.push_str(decoded_fragment);

        let mut output_events = Vec::new();
        while self.advance(&mut output_events)? {}
        Ok(output_events)
    }

    /// Completes the stream, rejecting unfinished tool calls or control markers.
    pub fn finish(&mut self) -> Result<Vec<Qwen3_5MoEOutputEvent>, Qwen3_5MoEOutputParserError> {
        match self.state {
            Qwen3_5MoEOutputParserState::Text => {
                if is_marker_prefix(&self.pending_output) {
                    return Err(Qwen3_5MoEOutputParserError::IncompleteControlMarker);
                }
                if self.pending_output.is_empty() {
                    return Ok(Vec::new());
                }
                let final_text = std::mem::take(&mut self.pending_output);
                self.has_streamed_visible_text = true;
                Ok(vec![Qwen3_5MoEOutputEvent::TextDelta(final_text)])
            }
            Qwen3_5MoEOutputParserState::Reasoning => {
                if self.pending_output.is_empty() {
                    return Ok(Vec::new());
                }
                let final_reasoning = std::mem::take(&mut self.pending_output);
                Ok(vec![Qwen3_5MoEOutputEvent::ReasoningDelta(final_reasoning)])
            }
            Qwen3_5MoEOutputParserState::ToolCall => {
                Err(Qwen3_5MoEOutputParserError::UnclosedToolCall)
            }
            Qwen3_5MoEOutputParserState::SuppressedLateReasoning => {
                self.pending_output.clear();
                Ok(Vec::new())
            }
        }
    }

    fn advance(
        &mut self,
        output_events: &mut Vec<Qwen3_5MoEOutputEvent>,
    ) -> Result<bool, Qwen3_5MoEOutputParserError> {
        match self.state {
            Qwen3_5MoEOutputParserState::Text => self.advance_text(output_events),
            Qwen3_5MoEOutputParserState::Reasoning => self.advance_reasoning(output_events),
            Qwen3_5MoEOutputParserState::ToolCall => self.advance_tool_call(output_events),
            Qwen3_5MoEOutputParserState::SuppressedLateReasoning => {
                self.advance_suppressed_late_reasoning()
            }
        }
    }

    fn advance_text(
        &mut self,
        output_events: &mut Vec<Qwen3_5MoEOutputEvent>,
    ) -> Result<bool, Qwen3_5MoEOutputParserError> {
        let next_marker = earliest_marker(
            &self.pending_output,
            &[THINK_START_MARKER, TOOL_CALL_START_MARKER],
        );
        if let Some((marker_index, marker)) = next_marker {
            if marker_index > 0 {
                self.has_streamed_visible_text = true;
                output_events.push(Qwen3_5MoEOutputEvent::TextDelta(
                    self.take_pending_prefix(marker_index),
                ));
                return Ok(true);
            }
            self.take_pending_prefix(marker.len());
            self.state = if marker == THINK_START_MARKER {
                if self.has_streamed_visible_text {
                    Qwen3_5MoEOutputParserState::SuppressedLateReasoning
                } else {
                    Qwen3_5MoEOutputParserState::Reasoning
                }
            } else {
                Qwen3_5MoEOutputParserState::ToolCall
            };
            return Ok(true);
        }

        let stable_text_bytes = self
            .pending_output
            .len()
            .saturating_sub(longest_marker_prefix_suffix(&self.pending_output));
        if stable_text_bytes == 0 {
            return Ok(false);
        }
        self.has_streamed_visible_text = true;
        output_events.push(Qwen3_5MoEOutputEvent::TextDelta(
            self.take_pending_prefix(stable_text_bytes),
        ));
        Ok(true)
    }

    fn advance_reasoning(
        &mut self,
        output_events: &mut Vec<Qwen3_5MoEOutputEvent>,
    ) -> Result<bool, Qwen3_5MoEOutputParserError> {
        if let Some(marker_index) = self.pending_output.find(THINK_END_MARKER) {
            if marker_index > 0 {
                output_events.push(Qwen3_5MoEOutputEvent::ReasoningDelta(
                    self.take_pending_prefix(marker_index),
                ));
                return Ok(true);
            }
            self.take_pending_prefix(THINK_END_MARKER.len());
            self.state = Qwen3_5MoEOutputParserState::Text;
            return Ok(true);
        }

        let stable_reasoning_bytes =
            self.pending_output
                .len()
                .saturating_sub(longest_suffix_prefix_for_markers(
                    &self.pending_output,
                    &[THINK_END_MARKER],
                ));
        if stable_reasoning_bytes == 0 {
            return Ok(false);
        }
        output_events.push(Qwen3_5MoEOutputEvent::ReasoningDelta(
            self.take_pending_prefix(stable_reasoning_bytes),
        ));
        Ok(true)
    }

    fn advance_suppressed_late_reasoning(&mut self) -> Result<bool, Qwen3_5MoEOutputParserError> {
        if let Some(marker_index) = self.pending_output.find(THINK_END_MARKER) {
            let hidden_reasoning_and_marker_bytes = marker_index + THINK_END_MARKER.len();
            self.take_pending_prefix(hidden_reasoning_and_marker_bytes);
            self.state = Qwen3_5MoEOutputParserState::Text;
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
        output_events: &mut Vec<Qwen3_5MoEOutputEvent>,
    ) -> Result<bool, Qwen3_5MoEOutputParserError> {
        let Some(marker_index) = self.pending_output.find(TOOL_CALL_END_MARKER) else {
            return Ok(false);
        };
        let tool_call_body = self.take_pending_prefix(marker_index);
        self.take_pending_prefix(TOOL_CALL_END_MARKER.len());
        match self.parse_tool_call(&tool_call_body) {
            Ok(tool_call) => {
                output_events.push(Qwen3_5MoEOutputEvent::ToolCall(tool_call));
                self.completed_tool_call_count = self
                    .completed_tool_call_count
                    .checked_add(1)
                    .ok_or(Qwen3_5MoEOutputParserError::TooManyToolCalls {
                        maximum_tool_call_count: MAX_TOOL_CALL_COUNT,
                    })?;
            }
            Err(Qwen3_5MoEOutputParserError::UndeclaredFunction { function_name }) => {
                output_events.push(Qwen3_5MoEOutputEvent::ModelVisibleCorrection {
                    correction_text: self.build_undeclared_function_correction_text(&function_name),
                });
            }
            Err(parser_error) => return Err(parser_error),
        }
        self.state = Qwen3_5MoEOutputParserState::Text;
        Ok(true)
    }

    fn build_undeclared_function_correction_text(&self, undeclared_function_name: &str) -> String {
        if self.declared_tools.is_empty() {
            return format!(
                "The previous tool call requested undeclared function '{undeclared_function_name}', but this request declared no tools. Please correct the tool call by answering without a tool call."
            );
        }

        format!(
            "The previous tool call requested undeclared function '{undeclared_function_name}', but no tool with that exact name exists. Please correct the tool call by using one of the exact declared tool names."
        )
    }

    fn parse_tool_call(
        &self,
        tool_call_body: &str,
    ) -> Result<Qwen3_5MoEToolCall, Qwen3_5MoEOutputParserError> {
        if self.completed_tool_call_count == MAX_TOOL_CALL_COUNT {
            return Err(Qwen3_5MoEOutputParserError::TooManyToolCalls {
                maximum_tool_call_count: MAX_TOOL_CALL_COUNT,
            });
        }
        let tool_call_body = tool_call_body.trim();
        let function_content = tool_call_body
            .strip_prefix("<function=")
            .ok_or(Qwen3_5MoEOutputParserError::ToolCallMissingFunction)?;
        let function_name_end = function_content
            .find('>')
            .ok_or(Qwen3_5MoEOutputParserError::ToolCallMissingFunctionNameEnd)?;
        let function_name = &function_content[..function_name_end];
        let declared_tool = self.declared_tools.get(function_name).ok_or_else(|| {
            Qwen3_5MoEOutputParserError::UndeclaredFunction {
                function_name: function_name.to_owned(),
            }
        })?;
        let parameter_content = function_content[function_name_end + 1..]
            .trim()
            .strip_suffix("</function>")
            .ok_or(Qwen3_5MoEOutputParserError::ToolCallMissingFunctionEnd)?
            .trim();
        let parsed_arguments = parse_tool_parameters(parameter_content, declared_tool)?;
        let arguments_json = serde_json::to_string(&Value::Object(parsed_arguments))
            .map_err(Qwen3_5MoEOutputParserError::SerializeToolArguments)?;
        Ok(Qwen3_5MoEToolCall {
            index: self.completed_tool_call_count,
            function_name: function_name.to_owned(),
            arguments_json,
        })
    }

    fn take_pending_prefix(&mut self, byte_count: usize) -> String {
        self.pending_output.drain(..byte_count).collect()
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
            Qwen3_5MoEOutputParserState::Reasoning
        } else {
            Qwen3_5MoEOutputParserState::Text
        };
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen3_5MoEOutputParserState {
    Text,
    Reasoning,
    ToolCall,
    SuppressedLateReasoning,
}

impl Qwen3_5MoEOutputParserState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Reasoning => "reasoning",
            Self::ToolCall => "tool_call",
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

fn earliest_marker<'a>(text: &str, markers: &'a [&'a str]) -> Option<(usize, &'a str)> {
    markers
        .iter()
        .filter_map(|marker| {
            text.find(marker)
                .map(|marker_index| (marker_index, *marker))
        })
        .min_by_key(|(marker_index, _)| *marker_index)
}

fn longest_marker_prefix_suffix(text: &str) -> usize {
    longest_suffix_prefix_for_markers(text, &[THINK_START_MARKER, TOOL_CALL_START_MARKER])
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

fn is_marker_prefix(text: &str) -> bool {
    !text.is_empty()
        && [THINK_START_MARKER, TOOL_CALL_START_MARKER]
            .iter()
            .any(|marker| marker.starts_with(text))
}
