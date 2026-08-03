use serde::Deserialize;
use serde_json::Value;

use crate::{MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH, OpenAiResponsesValidationError};

/// One Responses-native tool declaration accepted by the local model.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAiResponseToolDefinition {
    #[serde(rename = "type")]
    tool_type: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
    #[serde(default)]
    strict: Option<bool>,
    #[serde(flatten)]
    additional_fields: std::collections::BTreeMap<String, Value>,
}

/// One validated local function declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiResponseToolDefinitionParts {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: bool,
}

impl OpenAiResponseToolDefinition {
    pub(super) fn into_parts(
        self,
    ) -> Result<OpenAiResponseToolDefinitionParts, OpenAiResponsesValidationError> {
        if self.tool_type != "function" {
            return Err(OpenAiResponsesValidationError::UnsupportedOption {
                option_name: "tools[].type",
            });
        }
        if !self.additional_fields.is_empty() {
            return Err(OpenAiResponsesValidationError::UnsupportedOption {
                option_name: "tools[]",
            });
        }
        let name = self.name.unwrap_or_default();
        validate_function_name(&name)?;
        if self.strict == Some(true) {
            return Err(OpenAiResponsesValidationError::UnsupportedOption {
                option_name: "tools[].strict=true",
            });
        }
        if let Some(parameters) = &self.parameters {
            let schema_nesting_depth = json_nesting_depth(parameters);
            if schema_nesting_depth > MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH {
                return Err(OpenAiResponsesValidationError::ToolSchemaNestingTooDeep {
                    actual_schema_nesting_depth: schema_nesting_depth,
                    maximum_schema_nesting_depth: MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH,
                });
            }
        }
        Ok(OpenAiResponseToolDefinitionParts {
            name,
            description: self.description,
            parameters: self.parameters.unwrap_or_else(|| serde_json::json!({})),
            strict: false,
        })
    }
}

/// Responses tool-selection input, retained broadly for precise validation.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OpenAiResponseToolChoice {
    Mode(String),
    Selection(Value),
}

/// Tool-selection behavior the current Qwen3.5-MoE prompt can enforce.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiResponseToolChoiceParts {
    Auto,
    None,
}

impl OpenAiResponseToolChoiceParts {
    #[must_use]
    pub const fn kind_name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
        }
    }
}

impl OpenAiResponseToolChoice {
    pub(super) fn into_parts(
        self,
    ) -> Result<OpenAiResponseToolChoiceParts, OpenAiResponsesValidationError> {
        match self {
            Self::Mode(mode) if mode == "auto" => Ok(OpenAiResponseToolChoiceParts::Auto),
            Self::Mode(mode) if mode == "none" => Ok(OpenAiResponseToolChoiceParts::None),
            Self::Mode(_) | Self::Selection(_) => {
                Err(OpenAiResponsesValidationError::UnsupportedOption {
                    option_name: "tool_choice",
                })
            }
        }
    }
}

fn validate_function_name(function_name: &str) -> Result<(), OpenAiResponsesValidationError> {
    if !function_name.is_empty()
        && function_name
            .bytes()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, b'_' | b'-'))
    {
        return Ok(());
    }
    Err(OpenAiResponsesValidationError::InvalidToolName {
        tool_name: function_name.to_owned(),
    })
}

fn json_nesting_depth(json_value: &Value) -> usize {
    match json_value {
        Value::Array(array_values) => {
            1 + array_values
                .iter()
                .map(json_nesting_depth)
                .max()
                .unwrap_or(0)
        }
        Value::Object(object_values) => {
            1 + object_values
                .values()
                .map(json_nesting_depth)
                .max()
                .unwrap_or(0)
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => 0,
    }
}
