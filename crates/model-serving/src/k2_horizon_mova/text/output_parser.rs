//! Incremental IFM think-channel and tool-envelope parser.

use astronomical_ipc_protocol::ChatGenerationOutput;
use serde_json::{Map, Value};

const THINK_CLOSE: &str = "</ifm|think>";
const THINK_FAST_CLOSE: &str = "</ifm|think_fast>";
const THINK_FASTER_CLOSE: &str = "</ifm|think_faster>";
const TOOL_CALLS_OPEN: &str = "<ifm|tool_calls>";
const TOOL_CALLS_CLOSE: &str = "</ifm|tool_calls>";
const TOOL_CALL_OPEN: &str = "<ifm|tool_call>";
const TOOL_CALL_CLOSE: &str = "</ifm|tool_call>";
const ARG_KEY_OPEN: &str = "<ifm|arg_key>";
const ARG_KEY_CLOSE: &str = "</ifm|arg_key>";
const ARG_VALUE_OPEN: &str = "<ifm|arg_value>";
const ARG_VALUE_CLOSE: &str = "</ifm|arg_value>";
const TURN_END_MARKERS: [&str; 2] = ["<|ifm|im_end|>", "<|ifm|endoftext|>"];

/// Parses generated text into reasoning, visible text, and IFM tool envelopes.
#[derive(Debug)]
pub struct K2HorizonMoVAOutputParser {
    buffer: String,
    in_reasoning: bool,
    in_tool_calls: bool,
    completed_tool_call_count: u16,
    declared_tool_names: Vec<String>,
    accumulated_reasoning: String,
    has_emitted_visible_text: bool,
}

impl K2HorizonMoVAOutputParser {
    /// Prompt rendering already opened the think channel.
    #[must_use]
    pub fn new_after_thinking_prefix() -> Self {
        Self::with_declared_tool_names(Vec::new())
    }

    #[must_use]
    pub fn with_declared_tool_names(declared_tool_names: Vec<String>) -> Self {
        Self {
            buffer: String::new(),
            in_reasoning: true,
            in_tool_calls: false,
            completed_tool_call_count: 0,
            declared_tool_names,
            accumulated_reasoning: String::new(),
            has_emitted_visible_text: false,
        }
    }

    pub fn push_text(&mut self, fragment: &str) -> Vec<ChatGenerationOutput> {
        self.buffer.push_str(fragment);
        cut_completed_turn_end(&mut self.buffer);
        let mut outputs = Vec::new();
        loop {
            if self.in_reasoning {
                if !self.drain_reasoning(&mut outputs) {
                    break;
                }
                continue;
            }
            if self.in_tool_calls {
                if !self.drain_tool_calls(&mut outputs) {
                    break;
                }
                continue;
            }
            if let Some(tool_offset) = self.buffer.find(TOOL_CALLS_OPEN) {
                let visible = self.buffer[..tool_offset].to_owned();
                self.buffer = self.buffer[tool_offset + TOOL_CALLS_OPEN.len()..].to_owned();
                self.in_tool_calls = true;
                self.emit_visible_text(&mut outputs, visible);
                continue;
            }
            if let Some(held) = hold_incomplete_tag(&self.buffer) {
                let visible = self.buffer[..self.buffer.len() - held.len()].to_owned();
                self.buffer = held.to_owned();
                self.emit_visible_text(&mut outputs, visible);
                break;
            }
            if !self.buffer.is_empty() {
                let visible = std::mem::take(&mut self.buffer);
                self.emit_visible_text(&mut outputs, visible);
            }
            break;
        }
        outputs
    }

    pub fn finish(&mut self) -> Vec<ChatGenerationOutput> {
        let mut outputs = Vec::new();
        if self.in_tool_calls {
            self.drain_tool_calls(&mut outputs);
        }
        if let Some((reasoning, call, remainder)) = take_declared_tool_call(
            &self.buffer,
            &self.declared_tool_names,
            self.completed_tool_call_count,
        ) {
            self.buffer = remainder;
            self.completed_tool_call_count = self.completed_tool_call_count.saturating_add(1);
            self.emit_reasoning(&mut outputs, reasoning);
            outputs.push(call);
        }
        if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            if self.in_reasoning {
                self.emit_reasoning(&mut outputs, remaining);
            } else if !self.in_tool_calls {
                self.emit_visible_text(&mut outputs, remaining);
            }
        }
        if !self.has_emitted_visible_text
            && self.completed_tool_call_count == 0
            && !self.accumulated_reasoning.is_empty()
        {
            outputs.push(ChatGenerationOutput::Text {
                text: self.accumulated_reasoning.clone(),
            });
            self.has_emitted_visible_text = true;
        }
        outputs
    }

    fn emit_reasoning(&mut self, outputs: &mut Vec<ChatGenerationOutput>, reasoning: String) {
        if reasoning.is_empty() {
            return;
        }
        self.accumulated_reasoning.push_str(&reasoning);
        outputs.push(ChatGenerationOutput::Reasoning { text: reasoning });
    }

    fn emit_visible_text(&mut self, outputs: &mut Vec<ChatGenerationOutput>, text: String) {
        if text.is_empty() {
            return;
        }
        self.has_emitted_visible_text = true;
        outputs.push(ChatGenerationOutput::Text { text });
    }

    fn drain_reasoning(&mut self, outputs: &mut Vec<ChatGenerationOutput>) -> bool {
        if let Some(close_offset) = find_think_close(&self.buffer) {
            let reasoning = self.buffer[..close_offset].to_owned();
            let close_len = think_close_len(&self.buffer[close_offset..]);
            self.buffer = self.buffer[close_offset + close_len..].to_owned();
            self.in_reasoning = false;
            self.emit_reasoning(outputs, reasoning);
            return true;
        }
        if let Some(tool_offset) = self
            .buffer
            .find(TOOL_CALLS_OPEN)
            .or_else(|| self.buffer.find(TOOL_CALL_OPEN))
        {
            let opener_len = if self.buffer[tool_offset..].starts_with(TOOL_CALLS_OPEN) {
                TOOL_CALLS_OPEN.len()
            } else {
                0
            };
            let reasoning = self.buffer[..tool_offset].to_owned();
            self.buffer = self.buffer[tool_offset + opener_len..].to_owned();
            self.in_reasoning = false;
            self.in_tool_calls = true;
            self.emit_reasoning(outputs, reasoning);
            return true;
        }
        if let Some((reasoning, call, remainder)) = take_declared_tool_call(
            &self.buffer,
            &self.declared_tool_names,
            self.completed_tool_call_count,
        ) {
            self.buffer = remainder;
            self.in_reasoning = false;
            self.completed_tool_call_count = self.completed_tool_call_count.saturating_add(1);
            self.emit_reasoning(outputs, reasoning);
            outputs.push(call);
            return true;
        }
        if let Some(hold_start) = compact_tool_prefix_start(&self.buffer, &self.declared_tool_names)
        {
            let reasoning = self.buffer[..hold_start].to_owned();
            self.buffer = self.buffer[hold_start..].to_owned();
            self.emit_reasoning(outputs, reasoning);
            return false;
        }
        if let Some(held) = hold_incomplete_tag(&self.buffer) {
            let reasoning = self.buffer[..self.buffer.len() - held.len()].to_owned();
            self.buffer = held.to_owned();
            self.emit_reasoning(outputs, reasoning);
            return false;
        }
        if !self.buffer.is_empty() {
            let reasoning = std::mem::take(&mut self.buffer);
            self.emit_reasoning(outputs, reasoning);
        }
        false
    }

    fn drain_tool_calls(&mut self, outputs: &mut Vec<ChatGenerationOutput>) -> bool {
        while let Some(call) = take_complete_tool_call(&mut self.buffer) {
            outputs.push(with_tool_call_index(call, self.completed_tool_call_count));
            self.completed_tool_call_count = self.completed_tool_call_count.saturating_add(1);
        }
        if let Some(close_offset) = self.buffer.find(TOOL_CALLS_CLOSE) {
            self.buffer = self.buffer[close_offset + TOOL_CALLS_CLOSE.len()..].to_owned();
            self.in_tool_calls = false;
            return true;
        }
        false
    }
}

fn take_complete_tool_call(buffer: &mut String) -> Option<ChatGenerationOutput> {
    let start = buffer.find(TOOL_CALL_OPEN)?;
    let body_start = start + TOOL_CALL_OPEN.len();
    let close = buffer[body_start..].find(TOOL_CALL_CLOSE)?;
    let body = buffer[body_start..body_start + close].trim().to_owned();
    let after = body_start + close + TOOL_CALL_CLOSE.len();
    *buffer = buffer[after..].to_owned();
    parse_tool_call_body(&body)
}

fn parse_tool_call_body(body: &str) -> Option<ChatGenerationOutput> {
    let trimmed = body.trim();
    if trimmed.starts_with('{') {
        return parse_json_tool_call(trimmed);
    }
    parse_xml_tool_call(trimmed)
}

fn parse_json_tool_call(body: &str) -> Option<ChatGenerationOutput> {
    let value: Value = serde_json::from_str(body).ok()?;
    let name = value.get("name")?.as_str()?.to_owned();
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    Some(ChatGenerationOutput::ToolCall {
        tool_call_index: 0,
        function_name: name,
        arguments_json: arguments.to_string(),
    })
}

fn parse_xml_tool_call(body: &str) -> Option<ChatGenerationOutput> {
    let mut remaining = body;
    let name_end = remaining
        .find(ARG_KEY_OPEN)
        .unwrap_or_else(|| remaining.find('\n').unwrap_or(remaining.len()));
    let function_name = remaining[..name_end].trim().to_owned();
    if function_name.is_empty() {
        return None;
    }
    remaining = remaining[name_end..].trim_start();
    let mut arguments = Map::new();
    while let Some(key_start) = remaining.find(ARG_KEY_OPEN) {
        remaining = &remaining[key_start + ARG_KEY_OPEN.len()..];
        let key_end = remaining.find(ARG_KEY_CLOSE)?;
        let key = remaining[..key_end].trim().to_owned();
        remaining = remaining[key_end + ARG_KEY_CLOSE.len()..].trim_start();
        if remaining.starts_with("<ifm|arg_type>") {
            let type_end = remaining.find("</ifm|arg_type>")?;
            remaining = remaining[type_end + "</ifm|arg_type>".len()..].trim_start();
        }
        if !remaining.starts_with(ARG_VALUE_OPEN) {
            return None;
        }
        remaining = &remaining[ARG_VALUE_OPEN.len()..];
        let value_end = remaining.find(ARG_VALUE_CLOSE)?;
        let raw_value = remaining[..value_end].trim();
        remaining = remaining[value_end + ARG_VALUE_CLOSE.len()..].trim_start();
        arguments.insert(key, parse_argument_value(raw_value));
    }
    Some(ChatGenerationOutput::ToolCall {
        tool_call_index: 0,
        function_name,
        arguments_json: Value::Object(arguments).to_string(),
    })
}

fn with_tool_call_index(call: ChatGenerationOutput, tool_call_index: u16) -> ChatGenerationOutput {
    match call {
        ChatGenerationOutput::ToolCall {
            function_name,
            arguments_json,
            ..
        } => ChatGenerationOutput::ToolCall {
            tool_call_index,
            function_name,
            arguments_json,
        },
        other => other,
    }
}

fn compact_tool_prefix_start(buffer: &str, declared_tool_names: &[String]) -> Option<usize> {
    declared_tool_names
        .iter()
        .filter_map(|tool_name| {
            buffer.rfind(tool_name.as_str()).and_then(|start| {
                let after = &buffer[start + tool_name.len()..];
                (after.is_empty() || after.starts_with('{')).then_some(start)
            })
        })
        .min()
}

fn take_declared_tool_call(
    buffer: &str,
    declared_tool_names: &[String],
    tool_call_index: u16,
) -> Option<(String, ChatGenerationOutput, String)> {
    take_declared_compact_tool_call(buffer, declared_tool_names, tool_call_index)
        .or_else(|| take_declared_json_tool_call(buffer, declared_tool_names, tool_call_index))
}

fn take_declared_compact_tool_call(
    buffer: &str,
    declared_tool_names: &[String],
    tool_call_index: u16,
) -> Option<(String, ChatGenerationOutput, String)> {
    for tool_name in declared_tool_names {
        let compact_prefix = format!("{tool_name}{{");
        let Some(start) = buffer.find(&compact_prefix) else {
            continue;
        };
        let json_start = start + tool_name.len();
        let Some((arguments, json_len)) = parse_json_object_at(&buffer[json_start..]) else {
            continue;
        };
        let after = json_start + json_len;
        return Some((
            buffer[..start].to_owned(),
            ChatGenerationOutput::ToolCall {
                tool_call_index,
                function_name: tool_name.clone(),
                arguments_json: arguments.to_string(),
            },
            buffer.get(after..).unwrap_or("").to_owned(),
        ));
    }
    None
}

fn take_declared_json_tool_call(
    buffer: &str,
    declared_tool_names: &[String],
    tool_call_index: u16,
) -> Option<(String, ChatGenerationOutput, String)> {
    let mut search_from = 0;
    while let Some(relative_start) = buffer[search_from..].find('{') {
        let start = search_from + relative_start;
        let Some((value, json_len)) = parse_json_object_at(&buffer[start..]) else {
            search_from = start + 1;
            continue;
        };
        let function_name = value
            .get("name")
            .or_else(|| value.get("function"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let Some(function_name) = function_name else {
            search_from = start + 1;
            continue;
        };
        if !declared_tool_names
            .iter()
            .any(|tool_name| tool_name == &function_name)
        {
            search_from = start + 1;
            continue;
        }
        let arguments = value
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        return Some((
            buffer[..start].to_owned(),
            ChatGenerationOutput::ToolCall {
                tool_call_index,
                function_name,
                arguments_json: arguments.to_string(),
            },
            buffer.get(start + json_len..).unwrap_or("").to_owned(),
        ));
    }
    None
}

fn parse_json_object_at(source: &str) -> Option<(Value, usize)> {
    if !source.starts_with('{') {
        return None;
    }
    let mut depth = 0_i32;
    for (index, character) in source.char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = index + character.len_utf8();
                    let value = serde_json::from_str(&source[..end]).ok()?;
                    return Some((value, end));
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_argument_value(raw_value: &str) -> Value {
    serde_json::from_str(raw_value).unwrap_or_else(|_| Value::String(raw_value.to_owned()))
}

fn find_think_close(buffer: &str) -> Option<usize> {
    [THINK_CLOSE, THINK_FAST_CLOSE, THINK_FASTER_CLOSE]
        .into_iter()
        .filter_map(|marker| buffer.find(marker))
        .min()
}

fn think_close_len(from_close: &str) -> usize {
    if from_close.starts_with(THINK_CLOSE) {
        THINK_CLOSE.len()
    } else if from_close.starts_with(THINK_FAST_CLOSE) {
        THINK_FAST_CLOSE.len()
    } else {
        THINK_FASTER_CLOSE.len()
    }
}

fn cut_completed_turn_end(buffer: &mut String) {
    let Some(turn_end_offset) = TURN_END_MARKERS
        .into_iter()
        .filter_map(|marker| buffer.find(marker))
        .min()
    else {
        return;
    };
    buffer.truncate(turn_end_offset);
}

fn hold_incomplete_tag(buffer: &str) -> Option<&str> {
    let tag_start = buffer.rfind('<')?;
    let tail = &buffer[tag_start..];
    if tail.contains('>') { None } else { Some(tail) }
}
