use std::collections::BTreeSet;

use serde_json::Value;
use thiserror::Error;

use crate::{ChatGenerationCommand, ChatMessage, ChatToolChoice, ChatToolDefinition};

const MAX_CHAT_TOOL_SCHEMA_NESTING_DEPTH: usize = 32;
const MAX_CHAT_OUTPUT_TOKENS: u16 = u16::MAX;
const MAX_CHAT_TEMPERATURE_THOUSANDTHS: u16 = 2_000;
const MAX_CHAT_TOP_P_THOUSANDTHS: u16 = 1_000;

impl ChatGenerationCommand {
    /// Independently validates structured chat history after it crosses the worker boundary.
    pub fn validate(&self) -> Result<(), ChatGenerationValidationError> {
        if self.model.is_empty() {
            return Err(ChatGenerationValidationError::EmptyModelId);
        }
        if self.messages.is_empty() {
            return Err(ChatGenerationValidationError::EmptyMessages);
        }
        if self.settings.max_output_tokens == 0 {
            return Err(ChatGenerationValidationError::OutputTokenCountOutOfRange {
                actual_output_tokens: self.settings.max_output_tokens,
                maximum_output_tokens: MAX_CHAT_OUTPUT_TOKENS,
            });
        }
        if let Some(temperature_thousandths) = self.settings.temperature_thousandths
            && temperature_thousandths > MAX_CHAT_TEMPERATURE_THOUSANDTHS
        {
            return Err(ChatGenerationValidationError::TemperatureOutOfRange {
                actual_temperature_thousandths: temperature_thousandths,
                maximum_temperature_thousandths: MAX_CHAT_TEMPERATURE_THOUSANDTHS,
            });
        }
        if let Some(top_p_thousandths) = self.settings.top_p_thousandths
            && top_p_thousandths > MAX_CHAT_TOP_P_THOUSANDTHS
        {
            return Err(ChatGenerationValidationError::TopPOutOfRange {
                actual_top_p_thousandths: top_p_thousandths,
                maximum_top_p_thousandths: MAX_CHAT_TOP_P_THOUSANDTHS,
            });
        }
        let declared_tool_names = self.validate_tools()?;
        self.validate_messages()?;
        match &self.tool_choice {
            ChatToolChoice::Auto | ChatToolChoice::None => {}
            ChatToolChoice::Required => {
                return Err(ChatGenerationValidationError::UnsupportedToolChoice {
                    mode: "required",
                });
            }
            ChatToolChoice::Function { name } if !declared_tool_names.contains(name) => {
                return Err(
                    ChatGenerationValidationError::ToolChoiceNamesUnknownFunction {
                        function_name: name.clone(),
                    },
                );
            }
            ChatToolChoice::Function { .. } => {
                return Err(ChatGenerationValidationError::UnsupportedToolChoice {
                    mode: "function",
                });
            }
        }
        Ok(())
    }

    fn validate_tools(&self) -> Result<BTreeSet<&String>, ChatGenerationValidationError> {
        let mut declared_tool_names = BTreeSet::new();
        for tool_definition in &self.tools {
            if tool_definition.name.is_empty() {
                return Err(ChatGenerationValidationError::EmptyToolDefinitionName);
            }
            validate_tool_schema(tool_definition)?;
            if !declared_tool_names.insert(&tool_definition.name) {
                return Err(ChatGenerationValidationError::DuplicateToolDefinitionName {
                    function_name: tool_definition.name.clone(),
                });
            }
        }
        Ok(declared_tool_names)
    }

    fn validate_messages(&self) -> Result<(), ChatGenerationValidationError> {
        let mut active_tool_call_ids = BTreeSet::new();
        let mut completed_tool_result_ids = BTreeSet::new();
        for (message_index, chat_message) in self.messages.iter().enumerate() {
            if matches!(chat_message, ChatMessage::System { .. }) && message_index != 0 {
                return Err(ChatGenerationValidationError::SystemMessageMustBeFirst {
                    message_index,
                });
            }
            match chat_message {
                ChatMessage::Assistant { tool_calls, .. } => {
                    for tool_call in tool_calls {
                        validate_assistant_tool_call(tool_call)?;
                        let argument_value = serde_json::from_str::<Value>(
                            &tool_call.function.arguments_json,
                        )
                        .map_err(|_| {
                            ChatGenerationValidationError::InvalidAssistantToolCallArguments {
                                tool_call_id: tool_call.id.clone(),
                            }
                        })?;
                        if !argument_value.is_object() {
                            return Err(
                                ChatGenerationValidationError::AssistantToolCallArgumentsMustBeObject {
                                    tool_call_id: tool_call.id.clone(),
                                },
                            );
                        }
                        if !active_tool_call_ids.insert(&tool_call.id) {
                            return Err(
                                ChatGenerationValidationError::DuplicateAssistantToolCallId {
                                    tool_call_id: tool_call.id.clone(),
                                },
                            );
                        }
                        completed_tool_result_ids.remove(&tool_call.id);
                    }
                }
                ChatMessage::Tool { tool_call_id, .. } => {
                    if tool_call_id.is_empty() {
                        return Err(ChatGenerationValidationError::EmptyToolResultId);
                    }
                    if !active_tool_call_ids.remove(tool_call_id) {
                        if completed_tool_result_ids.contains(tool_call_id) {
                            return Err(ChatGenerationValidationError::DuplicateToolResultId {
                                tool_call_id: tool_call_id.clone(),
                            });
                        }
                        return Err(ChatGenerationValidationError::UnknownToolResultId {
                            tool_call_id: tool_call_id.clone(),
                        });
                    }
                    if !completed_tool_result_ids.insert(tool_call_id) {
                        return Err(ChatGenerationValidationError::DuplicateToolResultId {
                            tool_call_id: tool_call_id.clone(),
                        });
                    }
                }
                ChatMessage::System { .. } | ChatMessage::User { .. } => {}
            }
        }
        Ok(())
    }
}

/// A bounded semantic validation failure in one structured chat command.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ChatGenerationValidationError {
    #[error("model ID must not be empty")]
    EmptyModelId,
    #[error("messages must not be empty")]
    EmptyMessages,
    #[error(
        "output token count is {actual_output_tokens}, outside the 1..={maximum_output_tokens} token range"
    )]
    OutputTokenCountOutOfRange {
        actual_output_tokens: u16,
        maximum_output_tokens: u16,
    },
    #[error(
        "temperature is {actual_temperature_thousandths} thousandths, exceeding the {maximum_temperature_thousandths}-thousandths limit"
    )]
    TemperatureOutOfRange {
        actual_temperature_thousandths: u16,
        maximum_temperature_thousandths: u16,
    },
    #[error(
        "top_p is {actual_top_p_thousandths} thousandths, exceeding the {maximum_top_p_thousandths}-thousandths limit"
    )]
    TopPOutOfRange {
        actual_top_p_thousandths: u16,
        maximum_top_p_thousandths: u16,
    },
    #[error("system message at position {message_index} must be the first message")]
    SystemMessageMustBeFirst { message_index: usize },
    #[error("tool result refers to unknown assistant tool-call ID '{tool_call_id}'")]
    UnknownToolResultId { tool_call_id: String },
    #[error("tool result for assistant tool-call ID '{tool_call_id}' appears more than once")]
    DuplicateToolResultId { tool_call_id: String },
    #[error("tool function name '{function_name}' appears more than once")]
    DuplicateToolDefinitionName { function_name: String },
    #[error("tool function name must not be empty")]
    EmptyToolDefinitionName,
    #[error("tool choice names undeclared function '{function_name}'")]
    ToolChoiceNamesUnknownFunction { function_name: String },
    #[error("tool choice mode '{mode}' is unsupported by the current worker")]
    UnsupportedToolChoice { mode: &'static str },
    #[error("tool schema for function '{function_name}' is invalid JSON")]
    InvalidToolSchema { function_name: String },
    #[error("tool schema for function '{function_name}' must be a JSON object")]
    ToolSchemaMustBeObject { function_name: String },
    #[error(
        "tool schema for function '{function_name}' has nesting depth {actual_schema_nesting_depth}, exceeding {maximum_schema_nesting_depth}"
    )]
    ToolSchemaNestingTooDeep {
        function_name: String,
        actual_schema_nesting_depth: usize,
        maximum_schema_nesting_depth: usize,
    },
    #[error("assistant tool-call ID '{tool_call_id}' appears more than once")]
    DuplicateAssistantToolCallId { tool_call_id: String },
    #[error("assistant tool-call ID '{tool_call_id}' has invalid JSON arguments")]
    InvalidAssistantToolCallArguments { tool_call_id: String },
    #[error("assistant tool-call ID '{tool_call_id}' arguments must be a JSON object")]
    AssistantToolCallArgumentsMustBeObject { tool_call_id: String },
    #[error("assistant tool-call ID must not be empty")]
    EmptyAssistantToolCallId,
    #[error("assistant tool-call function name must not be empty")]
    EmptyAssistantToolCallFunctionName,
    #[error("tool result ID must not be empty")]
    EmptyToolResultId,
}

fn validate_assistant_tool_call(
    tool_call: &crate::ChatAssistantToolCall,
) -> Result<(), ChatGenerationValidationError> {
    if tool_call.id.is_empty() {
        return Err(ChatGenerationValidationError::EmptyAssistantToolCallId);
    }
    if tool_call.function.name.is_empty() {
        return Err(ChatGenerationValidationError::EmptyAssistantToolCallFunctionName);
    }
    Ok(())
}

fn validate_tool_schema(
    tool_definition: &ChatToolDefinition,
) -> Result<(), ChatGenerationValidationError> {
    let schema_value =
        serde_json::from_str::<Value>(&tool_definition.parameters_json).map_err(|_| {
            ChatGenerationValidationError::InvalidToolSchema {
                function_name: tool_definition.name.clone(),
            }
        })?;
    if !schema_value.is_object() {
        return Err(ChatGenerationValidationError::ToolSchemaMustBeObject {
            function_name: tool_definition.name.clone(),
        });
    }
    let schema_nesting_depth = json_nesting_depth(&schema_value);
    if schema_nesting_depth > MAX_CHAT_TOOL_SCHEMA_NESTING_DEPTH {
        return Err(ChatGenerationValidationError::ToolSchemaNestingTooDeep {
            function_name: tool_definition.name.clone(),
            actual_schema_nesting_depth: schema_nesting_depth,
            maximum_schema_nesting_depth: MAX_CHAT_TOOL_SCHEMA_NESTING_DEPTH,
        });
    }
    Ok(())
}

fn json_nesting_depth(json_value: &Value) -> usize {
    match json_value {
        Value::Array(values) => 1 + values.iter().map(json_nesting_depth).max().unwrap_or(0),
        Value::Object(values) => 1 + values.values().map(json_nesting_depth).max().unwrap_or(0),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => 0,
    }
}
