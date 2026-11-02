//! The Anthropic Messages request (`POST /v1/messages`) accepted by the local
//! server, validated into protocol-neutral parts before worker admission.
//!
//! Unknown top-level fields are rejected, matching the Responses surface: a
//! field this server silently ignored would change model behavior without the
//! client knowing. Fields that carry no instruction the local model can act
//! on are accepted and dropped: `metadata`, `service_tier`, and
//! `context_management`, whose edits ask the server to clear stale thinking
//! or tool results from a cached context the local worker does not keep in
//! that form. Claude Code sends `context_management` on every request.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::anthropic_messages_content::AnthropicImageBudget;
use crate::json_nesting_depth::json_nesting_depth;
use crate::tool_name_rule::is_valid_function_name;
use crate::{
    AnthropicInputMessage, AnthropicMessageParts, AnthropicSystemPrompt, MAX_OUTPUT_TOKENS,
    MAX_TOOL_SCHEMA_NESTING_DEPTH, OpenAiChatCompletionValidationError, ThinkingControlsError,
    ThinkingControlsInputs,
};

/// One bounded request to the local Anthropic-compatible Messages endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicMessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicInputMessage>,
    #[serde(default)]
    system: Option<AnthropicSystemPrompt>,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    context_management: Option<Value>,
    #[serde(default)]
    output_config: Option<AnthropicOutputConfiguration>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    tools: Vec<AnthropicToolDefinition>,
    #[serde(default)]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    thinking: Option<AnthropicThinkingConfiguration>,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

/// One client-declared tool. Server tools (`web_search_*`, `bash_*`, and the
/// like) carry a versioned `type` and run on Anthropic's side, so only custom
/// tools, whose `type` is absent or `custom`, can be served locally.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicToolDefinition {
    #[serde(rename = "type", default)]
    tool_type: Option<String>,
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: Option<Value>,
}

/// The tool-selection object.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    choice_type: String,
    #[serde(default)]
    disable_parallel_tool_use: Option<bool>,
}

/// The extended-thinking switch. `budget_tokens` applies to `enabled` only.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicThinkingConfiguration {
    #[serde(rename = "type")]
    thinking_type: String,
    #[serde(default)]
    budget_tokens: Option<u32>,
    /// `summarized` (the default) streams the thinking text; `omitted` keeps
    /// the thinking blocks but leaves their text empty, as Anthropic does.
    #[serde(default)]
    display: Option<String>,
}

/// Output controls. `effort` sets how hard the model thinks; any other
/// output control (such as a structured `format`) is rejected rather than
/// silently ignored.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicOutputConfiguration {
    #[serde(default)]
    effort: Option<String>,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

/// Validated Messages request data ready for daemon translation.
#[derive(Clone, Debug, PartialEq)]
pub struct AnthropicMessagesRequestParts {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<AnthropicMessageParts>,
    pub tools: Vec<AnthropicToolDefinitionParts>,
    pub tool_choice: AnthropicToolChoiceParts,
    pub maximum_output_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: bool,
    /// `Some(0)` disables thinking; `None` leaves the model default in force.
    pub thinking_budget: Option<u32>,
    /// The model still thinks and the tokens still count in usage; only the
    /// thinking text is withheld from the response.
    pub is_thinking_display_omitted: bool,
}

/// One validated custom tool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnthropicToolDefinitionParts {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Tool-selection behavior the current Qwen3.5-MoE prompt can enforce; the
/// worker rejects forced selection (`any`, `tool`), so it is rejected here
/// with the Anthropic option name instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnthropicToolChoiceParts {
    Auto,
    None,
}

impl AnthropicMessagesRequest {
    /// Validates and consumes this public request into protocol-neutral parts.
    pub fn into_parts(
        self,
    ) -> Result<AnthropicMessagesRequestParts, AnthropicMessagesValidationError> {
        if let Some((field_name, _)) = self.unknown_fields.first_key_value() {
            return Err(AnthropicMessagesValidationError::UnknownField {
                field_name: field_name.clone(),
            });
        }
        validate_request_scalars(&self)?;
        let thinking_budget =
            resolve_thinking_budget(self.thinking.as_ref(), self.output_config.as_ref())?;
        let is_thinking_display_omitted = resolve_thinking_display(self.thinking.as_ref())?;
        let tool_choice = resolve_tool_choice(self.tool_choice.as_ref())?;
        let system = self
            .system
            .map(AnthropicSystemPrompt::into_text)
            .transpose()?;
        let messages = message_parts(self.messages)?;
        let tools = with_added_tools(tool_definition_parts(self.tools)?, &messages);
        validate_tool_references(&messages, &tools)?;
        Ok(AnthropicMessagesRequestParts {
            model: self.model,
            system,
            messages,
            tools,
            tool_choice,
            maximum_output_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stream: self.stream,
            thinking_budget,
            is_thinking_display_omitted,
        })
    }
}

pub(crate) fn tool_definition_parts(
    tool_definitions: Vec<AnthropicToolDefinition>,
) -> Result<Vec<AnthropicToolDefinitionParts>, AnthropicMessagesValidationError> {
    let mut tools = Vec::with_capacity(tool_definitions.len());
    for tool_definition in tool_definitions {
        tools.push(tool_definition.into_parts()?);
    }
    Ok(tools)
}

/// Validates every turn against one shared image budget, because the
/// image-count and inline-media bounds apply to the request as a whole.
pub(crate) fn message_parts(
    input_messages: Vec<AnthropicInputMessage>,
) -> Result<Vec<AnthropicMessageParts>, AnthropicMessagesValidationError> {
    let mut image_budget = AnthropicImageBudget::default();
    let mut messages = Vec::with_capacity(input_messages.len());
    for input_message in input_messages {
        messages.push(input_message.into_parts(&mut image_budget)?);
    }
    Ok(messages)
}

fn validate_request_scalars(
    request: &AnthropicMessagesRequest,
) -> Result<(), AnthropicMessagesValidationError> {
    if request.model.is_empty() {
        return Err(AnthropicMessagesValidationError::EmptyModel);
    }
    if request.messages.is_empty() {
        return Err(AnthropicMessagesValidationError::EmptyMessages);
    }
    if request.max_tokens == 0 || request.max_tokens > MAX_OUTPUT_TOKENS {
        return Err(
            AnthropicMessagesValidationError::OutputTokenCountOutOfRange {
                actual_output_tokens: request.max_tokens,
                maximum_output_tokens: MAX_OUTPUT_TOKENS,
            },
        );
    }
    // Anthropic documents temperature on 0..=1, narrower than OpenAI's 0..=2.
    validate_sampling_parameter("temperature", request.temperature)?;
    validate_sampling_parameter("top_p", request.top_p)?;
    // The worker has no stop-sequence matcher or top-k sampler; rejecting
    // keeps a client from trusting a stop or a sampling limit never applied.
    if !request.stop_sequences.is_empty() {
        return Err(AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "stop_sequences",
        });
    }
    if request.top_k.is_some() {
        return Err(AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "top_k",
        });
    }
    Ok(())
}

fn validate_sampling_parameter(
    parameter_name: &'static str,
    parameter_value: Option<f32>,
) -> Result<(), AnthropicMessagesValidationError> {
    match parameter_value {
        Some(parameter_value)
            if !(parameter_value.is_finite() && (0.0..=1.0).contains(&parameter_value)) =>
        {
            Err(AnthropicMessagesValidationError::SamplingParameterOutOfRange { parameter_name })
        }
        _ => Ok(()),
    }
}

/// An explicit thinking switch wins: `disabled` is the zero budget the worker
/// renders as a closed thinking channel, and `enabled` carries its own
/// budget. For `adaptive` or an absent object, `output_config.effort` picks
/// the budget through the same level table the OpenAI surfaces use, so the
/// effort a Claude Code user selects is the effort the local model spends;
/// with no effort either, the model's own default stays in force.
pub(crate) fn resolve_thinking_budget(
    thinking: Option<&AnthropicThinkingConfiguration>,
    output_config: Option<&AnthropicOutputConfiguration>,
) -> Result<Option<u32>, AnthropicMessagesValidationError> {
    let effort_budget = resolve_effort_budget(output_config)?;
    let Some(thinking) = thinking else {
        return Ok(effort_budget);
    };
    match (thinking.thinking_type.as_str(), thinking.budget_tokens) {
        ("enabled", Some(budget_tokens)) if budget_tokens > 0 => Ok(Some(budget_tokens)),
        ("enabled", _) => Err(AnthropicMessagesValidationError::ThinkingBudgetRequired),
        ("disabled", _) => Ok(Some(0)),
        ("adaptive", _) => Ok(effort_budget),
        _ => Err(AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "thinking.type",
        }),
    }
}

fn resolve_effort_budget(
    output_config: Option<&AnthropicOutputConfiguration>,
) -> Result<Option<u32>, AnthropicMessagesValidationError> {
    let Some(output_config) = output_config else {
        return Ok(None);
    };
    if !output_config.unknown_fields.is_empty() {
        return Err(AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "output_config",
        });
    }
    let thinking_controls = ThinkingControlsInputs {
        reasoning_effort: output_config.effort.as_deref(),
        ..ThinkingControlsInputs::default()
    }
    .resolve()?;
    Ok(thinking_controls.budget)
}

pub(crate) fn resolve_thinking_display(
    thinking: Option<&AnthropicThinkingConfiguration>,
) -> Result<bool, AnthropicMessagesValidationError> {
    let Some(thinking) = thinking else {
        return Ok(false);
    };
    match thinking.display.as_deref() {
        None | Some("summarized") => Ok(false),
        Some("omitted") => Ok(true),
        Some(_) => Err(AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "thinking.display",
        }),
    }
}

/// Pure: appends the tool definitions system turns added mid-conversation
/// to the declared tools, skipping any name already declared, because the
/// worker takes one flat tool list and a duplicate name would be ambiguous.
pub(crate) fn with_added_tools(
    mut declared_tools: Vec<AnthropicToolDefinitionParts>,
    messages: &[AnthropicMessageParts],
) -> Vec<AnthropicToolDefinitionParts> {
    for message_parts in messages {
        let AnthropicMessageParts::System { added_tools, .. } = message_parts else {
            continue;
        };
        for added_tool in added_tools {
            if !is_declared_tool(&added_tool.name, &declared_tools) {
                declared_tools.push(added_tool.clone());
            }
        }
    }
    declared_tools
}

/// Every system-turn tool reference must name a declared tool.
pub(crate) fn validate_tool_references(
    messages: &[AnthropicMessageParts],
    tools: &[AnthropicToolDefinitionParts],
) -> Result<(), AnthropicMessagesValidationError> {
    for message_parts in messages {
        let AnthropicMessageParts::System {
            referenced_tool_names,
            ..
        } = message_parts
        else {
            continue;
        };
        for referenced_tool_name in referenced_tool_names {
            if !is_declared_tool(referenced_tool_name, tools) {
                return Err(AnthropicMessagesValidationError::UndeclaredToolReference {
                    tool_name: referenced_tool_name.clone(),
                });
            }
        }
    }
    Ok(())
}

fn is_declared_tool(tool_name: &str, tools: &[AnthropicToolDefinitionParts]) -> bool {
    for tool_definition in tools {
        if tool_definition.name == tool_name {
            return true;
        }
    }
    false
}

pub(crate) fn resolve_tool_choice(
    tool_choice: Option<&AnthropicToolChoice>,
) -> Result<AnthropicToolChoiceParts, AnthropicMessagesValidationError> {
    let Some(tool_choice) = tool_choice else {
        return Ok(AnthropicToolChoiceParts::Auto);
    };
    if tool_choice.disable_parallel_tool_use == Some(true) {
        return Err(AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "tool_choice.disable_parallel_tool_use",
        });
    }
    match tool_choice.choice_type.as_str() {
        "auto" => Ok(AnthropicToolChoiceParts::Auto),
        "none" => Ok(AnthropicToolChoiceParts::None),
        _ => Err(AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "tool_choice.type",
        }),
    }
}

impl AnthropicToolDefinition {
    /// Shared by top-level tools and the tool definitions Claude Code adds
    /// mid-conversation, so both pass the same name and schema rules.
    pub(crate) fn into_parts(
        self,
    ) -> Result<AnthropicToolDefinitionParts, AnthropicMessagesValidationError> {
        if !matches!(self.tool_type.as_deref(), None | Some("custom")) {
            return Err(AnthropicMessagesValidationError::UnsupportedToolType {
                tool_type: self.tool_type.unwrap_or_default(),
            });
        }
        if !is_valid_function_name(&self.name) {
            return Err(AnthropicMessagesValidationError::InvalidToolName {
                tool_name: self.name,
            });
        }
        let input_schema = self.input_schema.unwrap_or_else(default_tool_input_schema);
        let schema_nesting_depth = json_nesting_depth(&input_schema);
        if schema_nesting_depth > MAX_TOOL_SCHEMA_NESTING_DEPTH {
            return Err(AnthropicMessagesValidationError::ToolSchemaNestingTooDeep {
                actual_schema_nesting_depth: schema_nesting_depth,
                maximum_schema_nesting_depth: MAX_TOOL_SCHEMA_NESTING_DEPTH,
            });
        }
        Ok(AnthropicToolDefinitionParts {
            name: self.name,
            description: self.description,
            input_schema,
        })
    }
}

fn default_tool_input_schema() -> Value {
    serde_json::json!({ "type": "object" })
}

/// A request rejected before worker admission by the Messages contract.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum AnthropicMessagesValidationError {
    #[error("model must not be empty")]
    EmptyModel,
    #[error("messages must not be empty")]
    EmptyMessages,
    #[error("message content must not be empty")]
    EmptyMessageContent,
    #[error("message role '{role}' is unsupported; expected user, assistant, or system")]
    UnsupportedRole { role: String },
    #[error("this content block type is unsupported in {location}")]
    UnsupportedContentBlock { location: &'static str },
    #[error("redacted thinking from another provider cannot be replayed locally")]
    UnsupportedReasoningReplay,
    #[error("tool_use input could not be re-serialized as JSON")]
    UnserializableToolInput,
    #[error("image source type '{source_type}' is unsupported; send a base64 image source")]
    UnsupportedImageSource { source_type: String },
    #[error("invalid image input: {0}")]
    ImageInput(#[source] OpenAiChatCompletionValidationError),
    #[error("the request carries more than {maximum_image_part_count} images")]
    ImagePartCountExceedsBudget { maximum_image_part_count: usize },
    #[error("the request's decoded images exceed the {maximum_inline_media_bytes}-byte budget")]
    InlineMediaExceedsBudget { maximum_inline_media_bytes: usize },
    #[error("tool type '{tool_type}' is a server tool and cannot run locally")]
    UnsupportedToolType { tool_type: String },
    #[error("tool name '{tool_name}' is invalid")]
    InvalidToolName { tool_name: String },
    #[error("tool_addition references '{tool_name}', which is not declared in tools")]
    UndeclaredToolReference { tool_name: String },
    #[error(
        "tool schema nesting depth is {actual_schema_nesting_depth}, exceeding {maximum_schema_nesting_depth}"
    )]
    ToolSchemaNestingTooDeep {
        actual_schema_nesting_depth: usize,
        maximum_schema_nesting_depth: usize,
    },
    /// `output_config.effort` named a level outside the shared level table.
    #[error(transparent)]
    ThinkingControls(#[from] ThinkingControlsError),
    #[error("thinking.type 'enabled' requires a positive budget_tokens")]
    ThinkingBudgetRequired,
    #[error("request option '{option_name}' is unsupported")]
    UnsupportedOption { option_name: &'static str },
    #[error(
        "max_tokens is {actual_output_tokens}, outside the 1..={maximum_output_tokens} token range"
    )]
    OutputTokenCountOutOfRange {
        actual_output_tokens: u32,
        maximum_output_tokens: u32,
    },
    #[error("{parameter_name} is outside the supported range 0..=1")]
    SamplingParameterOutOfRange { parameter_name: &'static str },
    #[error("request field '{field_name}' is unknown")]
    UnknownField { field_name: String },
}

impl AnthropicMessagesValidationError {
    /// Whether this rejection is a request-size rejection (HTTP 413) rather
    /// than a malformed-request one (HTTP 400).
    #[must_use]
    pub fn is_request_too_large(&self) -> bool {
        matches!(
            self,
            Self::ImagePartCountExceedsBudget { .. } | Self::InlineMediaExceedsBudget { .. }
        ) || matches!(self, Self::ImageInput(image_error) if image_error.is_request_too_large())
    }
}

#[cfg(test)]
mod tests;
