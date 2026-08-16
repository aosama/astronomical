use std::collections::BTreeMap;

use astronomical_ipc_protocol::ChatToolDefinition;
use serde_json::Value;

use super::tool_contract::{LagunaDeclaredTool, bounded_text};
use super::{LagunaOutputParserError, LagunaTextArtifactDescriptor};

const THINK_START_MARKER: &str = "<think>";
const THINK_END_MARKER: &str = "</think>";
const TOOL_CALL_START_MARKER: &str = "<tool_call>";
const TOOL_CALL_END_MARKER: &str = "</tool_call>";
const ARGUMENT_KEY_START_MARKER: &str = "<arg_key>";
const ARGUMENT_KEY_END_MARKER: &str = "</arg_key>";
const ARGUMENT_VALUE_START_MARKER: &str = "<arg_value>";
const ARGUMENT_VALUE_END_MARKER: &str = "</arg_value>";
const MAXIMUM_FRAGMENT_BYTES: usize = 64 * 1024;
const MAXIMUM_PENDING_TEXT_BYTES: usize = 256 * 1024;

/// One architecture-neutral event parsed from Poolside output syntax.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LagunaOutputEvent {
    ReasoningDelta(String),
    TextDelta(String),
    ToolCall {
        index: u16,
        function_name: String,
        arguments_json: String,
    },
}

/// Strict bounded incremental parser for the certified Poolside output contract.
#[derive(Debug)]
pub struct LagunaOutputParser {
    completed_tool_call_count: u16,
    declared_tools: BTreeMap<String, LagunaDeclaredTool>,
    pending_output: String,
    state: LagunaOutputParserState,
}

impl LagunaOutputParser {
    /// Maximum aggregate bytes accepted inside one generated argument value.
    pub const MAXIMUM_TOOL_ARGUMENT_BYTES: usize = 128 * 1024;

    /// Creates request-local parser state from the exact declared tools.
    pub fn new(
        descriptor: &LagunaTextArtifactDescriptor,
        declared_tools: &[ChatToolDefinition],
        generation_starts_in_reasoning: bool,
    ) -> Result<Self, LagunaOutputParserError> {
        if descriptor.reasoning_parser_id() != "poolside_v1"
            || descriptor.tool_call_parser_id() != "poolside_v1"
        {
            return Err(LagunaOutputParserError::MalformedToolCall);
        }
        let mut tools_by_name = BTreeMap::new();
        for tool_definition in declared_tools {
            let declared_tool = LagunaDeclaredTool::from_definition(tool_definition)?;
            if tools_by_name
                .insert(tool_definition.name.clone(), declared_tool)
                .is_some()
            {
                return Err(LagunaOutputParserError::DuplicateDeclaredTool {
                    function_name: bounded_text(&tool_definition.name),
                });
            }
        }
        Ok(Self {
            completed_tool_call_count: 0,
            declared_tools: tools_by_name,
            pending_output: String::new(),
            state: if generation_starts_in_reasoning {
                LagunaOutputParserState::Reasoning
            } else {
                LagunaOutputParserState::Text
            },
        })
    }

    /// Consumes one tokenizer-stable text fragment and emits only complete events.
    pub fn push_fragment(
        &mut self,
        decoded_fragment: &str,
    ) -> Result<Vec<LagunaOutputEvent>, LagunaOutputParserError> {
        if decoded_fragment.len() > MAXIMUM_FRAGMENT_BYTES {
            return Err(LagunaOutputParserError::FragmentTooLarge {
                maximum_bytes: MAXIMUM_FRAGMENT_BYTES,
            });
        }
        self.pending_output.push_str(decoded_fragment);
        self.validate_pending_bound()?;
        let mut output_events = Vec::new();
        while self.advance(&mut output_events)? {}
        Ok(output_events)
    }

    /// Flushes ordinary text and reasoning while rejecting incomplete control syntax.
    pub fn finish(&mut self) -> Result<Vec<LagunaOutputEvent>, LagunaOutputParserError> {
        match self.state {
            LagunaOutputParserState::ToolCall => Err(LagunaOutputParserError::IncompleteToolCall),
            LagunaOutputParserState::Text => {
                if is_prefix_of_any_marker(
                    &self.pending_output,
                    &[THINK_START_MARKER, TOOL_CALL_START_MARKER],
                ) {
                    return Err(LagunaOutputParserError::IncompleteControlMarker);
                }
                let remaining_text = std::mem::take(&mut self.pending_output);
                Ok((!remaining_text.is_empty())
                    .then_some(LagunaOutputEvent::TextDelta(remaining_text))
                    .into_iter()
                    .collect())
            }
            LagunaOutputParserState::Reasoning => {
                let remaining_reasoning = std::mem::take(&mut self.pending_output);
                Ok((!remaining_reasoning.is_empty())
                    .then_some(LagunaOutputEvent::ReasoningDelta(remaining_reasoning))
                    .into_iter()
                    .collect())
            }
        }
    }

    fn advance(
        &mut self,
        output_events: &mut Vec<LagunaOutputEvent>,
    ) -> Result<bool, LagunaOutputParserError> {
        match self.state {
            LagunaOutputParserState::Text => self.advance_text(output_events),
            LagunaOutputParserState::Reasoning => self.advance_reasoning(output_events),
            LagunaOutputParserState::ToolCall => self.advance_tool_call(output_events),
        }
    }

    fn advance_text(
        &mut self,
        output_events: &mut Vec<LagunaOutputEvent>,
    ) -> Result<bool, LagunaOutputParserError> {
        if let Some((marker_offset, marker)) = earliest_marker(
            &self.pending_output,
            &[THINK_START_MARKER, TOOL_CALL_START_MARKER],
        ) {
            if marker_offset > 0 {
                output_events.push(LagunaOutputEvent::TextDelta(
                    self.take_pending_prefix(marker_offset),
                ));
                return Ok(true);
            }
            self.take_pending_prefix(marker.len());
            self.state = if marker == THINK_START_MARKER {
                LagunaOutputParserState::Reasoning
            } else {
                LagunaOutputParserState::ToolCall
            };
            return Ok(true);
        }
        let stable_bytes =
            self.pending_output
                .len()
                .saturating_sub(longest_suffix_matching_marker_prefix(
                    &self.pending_output,
                    &[THINK_START_MARKER, TOOL_CALL_START_MARKER],
                ));
        if stable_bytes == 0 {
            return Ok(false);
        }
        output_events.push(LagunaOutputEvent::TextDelta(
            self.take_pending_prefix(stable_bytes),
        ));
        Ok(true)
    }

    fn advance_reasoning(
        &mut self,
        output_events: &mut Vec<LagunaOutputEvent>,
    ) -> Result<bool, LagunaOutputParserError> {
        if let Some(marker_offset) = self.pending_output.find(THINK_END_MARKER) {
            if marker_offset > 0 {
                output_events.push(LagunaOutputEvent::ReasoningDelta(
                    self.take_pending_prefix(marker_offset),
                ));
                return Ok(true);
            }
            self.take_pending_prefix(THINK_END_MARKER.len());
            self.state = LagunaOutputParserState::Text;
            return Ok(true);
        }
        let stable_bytes =
            self.pending_output
                .len()
                .saturating_sub(longest_suffix_matching_marker_prefix(
                    &self.pending_output,
                    &[THINK_END_MARKER],
                ));
        if stable_bytes == 0 {
            return Ok(false);
        }
        output_events.push(LagunaOutputEvent::ReasoningDelta(
            self.take_pending_prefix(stable_bytes),
        ));
        Ok(true)
    }

    fn advance_tool_call(
        &mut self,
        output_events: &mut Vec<LagunaOutputEvent>,
    ) -> Result<bool, LagunaOutputParserError> {
        self.validate_open_argument_size()?;
        let Some(end_marker_offset) = self.pending_output.find(TOOL_CALL_END_MARKER) else {
            return Ok(false);
        };
        let tool_call_body = self.take_pending_prefix(end_marker_offset);
        self.take_pending_prefix(TOOL_CALL_END_MARKER.len());
        let output_event = self.parse_tool_call(&tool_call_body)?;
        output_events.push(output_event);
        self.completed_tool_call_count = self
            .completed_tool_call_count
            .checked_add(1)
            .ok_or(LagunaOutputParserError::TooManyToolCalls)?;
        self.state = LagunaOutputParserState::Text;
        Ok(true)
    }

    fn parse_tool_call(
        &self,
        tool_call_body: &str,
    ) -> Result<LagunaOutputEvent, LagunaOutputParserError> {
        let first_argument_offset = tool_call_body
            .find(ARGUMENT_KEY_START_MARKER)
            .unwrap_or(tool_call_body.len());
        let function_name = tool_call_body[..first_argument_offset].trim();
        if function_name.is_empty() || function_name.contains('<') {
            return Err(LagunaOutputParserError::MalformedToolCall);
        }
        let declared_tool = self.declared_tools.get(function_name).ok_or_else(|| {
            LagunaOutputParserError::UndeclaredFunction {
                function_name: bounded_text(function_name),
            }
        })?;
        let raw_arguments = parse_argument_pairs(&tool_call_body[first_argument_offset..])?;
        let parsed_arguments = declared_tool.parse_arguments(function_name, raw_arguments)?;
        let arguments_json = serde_json::to_string(&Value::Object(parsed_arguments))
            .map_err(LagunaOutputParserError::SerializeToolArguments)?;
        Ok(LagunaOutputEvent::ToolCall {
            index: self.completed_tool_call_count,
            function_name: function_name.to_owned(),
            arguments_json,
        })
    }

    fn validate_pending_bound(&self) -> Result<(), LagunaOutputParserError> {
        let maximum_pending_bytes = match self.state {
            LagunaOutputParserState::ToolCall => {
                Self::MAXIMUM_TOOL_ARGUMENT_BYTES + MAXIMUM_PENDING_TEXT_BYTES
            }
            LagunaOutputParserState::Text | LagunaOutputParserState::Reasoning => {
                MAXIMUM_PENDING_TEXT_BYTES
            }
        };
        if self.pending_output.len() > maximum_pending_bytes {
            return Err(LagunaOutputParserError::PendingOutputTooLarge {
                maximum_bytes: maximum_pending_bytes,
            });
        }
        Ok(())
    }

    fn validate_open_argument_size(&self) -> Result<(), LagunaOutputParserError> {
        let Some(value_start_offset) = self.pending_output.rfind(ARGUMENT_VALUE_START_MARKER)
        else {
            return Ok(());
        };
        let value_content_offset = value_start_offset + ARGUMENT_VALUE_START_MARKER.len();
        if self.pending_output[value_content_offset..].contains(ARGUMENT_VALUE_END_MARKER) {
            return Ok(());
        }
        let actual_bytes = self
            .pending_output
            .len()
            .saturating_sub(value_content_offset);
        if actual_bytes > Self::MAXIMUM_TOOL_ARGUMENT_BYTES {
            return Err(LagunaOutputParserError::ToolArgumentsTooLarge {
                actual_bytes,
                maximum_bytes: Self::MAXIMUM_TOOL_ARGUMENT_BYTES,
            });
        }
        Ok(())
    }

    fn take_pending_prefix(&mut self, byte_count: usize) -> String {
        self.pending_output.drain(..byte_count).collect()
    }

    pub(crate) const fn state_for_diagnostics(&self) -> &'static str {
        match self.state {
            LagunaOutputParserState::Text => "text",
            LagunaOutputParserState::Reasoning => "reasoning",
            LagunaOutputParserState::ToolCall => "tool_call",
        }
    }

    pub(crate) fn pending_output_for_diagnostics(&self) -> &str {
        &self.pending_output
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LagunaOutputParserState {
    Text,
    Reasoning,
    ToolCall,
}

fn parse_argument_pairs(
    argument_content: &str,
) -> Result<Vec<(String, String)>, LagunaOutputParserError> {
    let mut remaining_content = argument_content;
    let mut raw_arguments = Vec::new();
    while !remaining_content.is_empty() {
        let key_content = remaining_content
            .strip_prefix(ARGUMENT_KEY_START_MARKER)
            .ok_or(LagunaOutputParserError::MalformedToolCall)?;
        let key_end_offset = key_content
            .find(ARGUMENT_KEY_END_MARKER)
            .ok_or(LagunaOutputParserError::MalformedToolCall)?;
        let argument_name = &key_content[..key_end_offset];
        if contains_argument_marker(argument_name) {
            return Err(LagunaOutputParserError::NestedToolArgumentMarker);
        }
        let value_with_marker = &key_content[key_end_offset + ARGUMENT_KEY_END_MARKER.len()..];
        let value_content = value_with_marker
            .strip_prefix(ARGUMENT_VALUE_START_MARKER)
            .ok_or(LagunaOutputParserError::MalformedToolCall)?;
        let value_end_offset = value_content
            .find(ARGUMENT_VALUE_END_MARKER)
            .ok_or(LagunaOutputParserError::MalformedToolCall)?;
        let argument_value = &value_content[..value_end_offset];
        if contains_argument_marker(argument_value) {
            return Err(LagunaOutputParserError::NestedToolArgumentMarker);
        }
        if argument_value.len() > LagunaOutputParser::MAXIMUM_TOOL_ARGUMENT_BYTES {
            return Err(LagunaOutputParserError::ToolArgumentsTooLarge {
                actual_bytes: argument_value.len(),
                maximum_bytes: LagunaOutputParser::MAXIMUM_TOOL_ARGUMENT_BYTES,
            });
        }
        raw_arguments.push((argument_name.to_owned(), argument_value.to_owned()));
        remaining_content = &value_content[value_end_offset + ARGUMENT_VALUE_END_MARKER.len()..];
    }
    Ok(raw_arguments)
}

fn contains_argument_marker(text: &str) -> bool {
    [
        ARGUMENT_KEY_START_MARKER,
        ARGUMENT_KEY_END_MARKER,
        ARGUMENT_VALUE_START_MARKER,
        ARGUMENT_VALUE_END_MARKER,
        TOOL_CALL_START_MARKER,
        TOOL_CALL_END_MARKER,
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn earliest_marker<'a>(text: &str, markers: &'a [&'a str]) -> Option<(usize, &'a str)> {
    markers
        .iter()
        .filter_map(|marker| text.find(marker).map(|offset| (offset, *marker)))
        .min_by_key(|(offset, _)| *offset)
}

fn longest_suffix_matching_marker_prefix(text: &str, markers: &[&str]) -> usize {
    let maximum_prefix_bytes = markers
        .iter()
        .map(|marker| marker.len().saturating_sub(1))
        .max()
        .unwrap_or(0)
        .min(text.len());
    for suffix_bytes in (1..=maximum_prefix_bytes).rev() {
        let suffix_offset = text.len() - suffix_bytes;
        if text.is_char_boundary(suffix_offset)
            && markers
                .iter()
                .any(|marker| marker.starts_with(&text[suffix_offset..]))
        {
            return suffix_bytes;
        }
    }
    0
}

fn is_prefix_of_any_marker(text: &str, markers: &[&str]) -> bool {
    !text.is_empty() && markers.iter().any(|marker| marker.starts_with(text))
}
