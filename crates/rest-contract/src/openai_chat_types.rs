use serde::Deserialize;
use serde_json::Value;

use crate::image_input::{decode_image_url, validate_image_url_scheme};
use crate::{
    MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH, OpenAiChatCompletionValidationError, OpenAiImageInput,
};

/// A chat history message supported by the initial local text-only endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
pub enum OpenAiChatMessage {
    /// An initial system instruction.
    System { content: OpenAiMessageContent },
    /// A user message.
    User { content: OpenAiMessageContent },
    /// A prior assistant response, possibly containing tool calls.
    Assistant {
        #[serde(default)]
        content: Option<OpenAiMessageContent>,
        #[serde(default)]
        reasoning_content: Option<String>,
        #[serde(default)]
        tool_calls: Vec<OpenAiAssistantToolCall>,
    },
    /// A result returned by one prior assistant tool call.
    Tool {
        content: OpenAiMessageContent,
        tool_call_id: String,
    },
}

impl OpenAiChatMessage {
    pub(crate) fn validate(&self) -> Result<(), OpenAiChatCompletionValidationError> {
        match self {
            Self::System { content } | Self::User { content } => content.validate(),
            Self::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                if content.is_none() && reasoning_content.is_none() && tool_calls.is_empty() {
                    return Err(OpenAiChatCompletionValidationError::EmptyAssistantMessage);
                }
                if let Some(content) = content {
                    content.validate()?;
                }
                for assistant_tool_call in tool_calls {
                    assistant_tool_call.validate()?;
                }
                Ok(())
            }
            Self::Tool {
                content,
                tool_call_id,
            } => {
                validate_non_empty_string("tool_call_id", tool_call_id)?;
                content.validate()
            }
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> Result<OpenAiChatMessageParts, OpenAiChatCompletionValidationError> {
        self.validate()?;
        match self {
            Self::System { content } => Ok(OpenAiChatMessageParts::System {
                content: content.into_text()?,
            }),
            Self::User { content } => {
                let (text_content, images) = content.into_user_content()?;
                Ok(OpenAiChatMessageParts::User {
                    content: text_content,
                    images,
                })
            }
            Self::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => Ok(OpenAiChatMessageParts::Assistant {
                content: content.map(OpenAiMessageContent::into_text).transpose()?,
                reasoning_content,
                tool_calls: tool_calls
                    .into_iter()
                    .map(OpenAiAssistantToolCall::into_parts)
                    .collect(),
            }),
            Self::Tool {
                content,
                tool_call_id,
            } => Ok(OpenAiChatMessageParts::Tool {
                tool_call_id,
                content: content.into_text()?,
            }),
        }
    }
}

/// One validated, text-or-image chat message ready for protocol translation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpenAiChatMessageParts {
    /// An initial system instruction.
    System { content: String },
    /// A user message, possibly carrying decoded image inputs.
    User {
        content: String,
        /// Decoded data-URI image payloads in document order. Empty for text-only users.
        images: Vec<OpenAiImageInput>,
    },
    /// A prior assistant answer, reasoning, and function calls.
    Assistant {
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<OpenAiAssistantToolCallParts>,
    },
    /// A result produced by an earlier function call.
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// Text-only message content accepted by the initial endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OpenAiMessageContent {
    /// A compact string message.
    Text(String),
    /// A list of explicitly typed content parts.
    Parts(Vec<OpenAiContentPart>),
}

impl OpenAiMessageContent {
    fn validate(&self) -> Result<(), OpenAiChatCompletionValidationError> {
        match self {
            Self::Text(_content) => Ok(()),
            Self::Parts(content_parts) => {
                if content_parts.is_empty() {
                    return Err(OpenAiChatCompletionValidationError::EmptyContentParts);
                }
                for content_part in content_parts {
                    content_part.validate()?;
                }
                Ok(())
            }
        }
    }

    fn into_text(self) -> Result<String, OpenAiChatCompletionValidationError> {
        self.validate()?;
        match self {
            Self::Text(content) => Ok(content),
            Self::Parts(content_parts) => Ok(content_parts.into_iter().fold(
                String::new(),
                |mut combined_content, content_part| {
                    if let OpenAiContentPart::Text { text } = content_part {
                        combined_content.push_str(&text);
                    }
                    combined_content
                },
            )),
        }
    }

    /// Decodes user message content into concatenated text and ordered image inputs.
    fn into_user_content(
        self,
    ) -> Result<(String, Vec<OpenAiImageInput>), OpenAiChatCompletionValidationError> {
        self.validate()?;
        match self {
            Self::Text(content) => Ok((content, Vec::new())),
            Self::Parts(content_parts) => {
                let mut combined_text = String::new();
                let mut decoded_images = Vec::new();
                for content_part in content_parts {
                    match content_part {
                        OpenAiContentPart::Text { text } => combined_text.push_str(&text),
                        OpenAiContentPart::ImageUrl { image_url } => {
                            decoded_images.push(decode_image_url(&image_url.url)?);
                        }
                        OpenAiContentPart::InputAudio { .. }
                        | OpenAiContentPart::VideoUrl { .. } => {
                            // Already rejected by validate(); unreachable here.
                        }
                    }
                }
                Ok((combined_text, decoded_images))
            }
        }
    }
}

/// A typed content part. Text and data-URI images are supported by the local endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum OpenAiContentPart {
    /// A text fragment.
    Text { text: String },
    /// An image input encoded as a `data:image/...;base64,...` URI.
    ImageUrl { image_url: OpenAiImageUrl },
    /// An audio input, intentionally unsupported by the initial endpoint.
    InputAudio { input_audio: Value },
    /// A video input, intentionally unsupported by the initial endpoint.
    VideoUrl { video_url: Value },
}

/// The `image_url` object inside an `image_url` content part.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiImageUrl {
    /// The image URL. Only `data:image/...;base64,...` URIs are accepted.
    pub(crate) url: String,
}

impl OpenAiContentPart {
    fn validate(&self) -> Result<(), OpenAiChatCompletionValidationError> {
        match self {
            Self::Text { text: _ } => Ok(()),
            Self::ImageUrl { image_url } => validate_image_url_scheme(&image_url.url),
            Self::InputAudio { .. } => Err(
                OpenAiChatCompletionValidationError::UnsupportedContentPart {
                    content_part_type: "input_audio",
                },
            ),
            Self::VideoUrl { .. } => Err(
                OpenAiChatCompletionValidationError::UnsupportedContentPart {
                    content_part_type: "video_url",
                },
            ),
        }
    }
}

/// A function tool made available to the model.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiToolDefinition {
    #[serde(rename = "type")]
    tool_type: OpenAiToolType,
    function: OpenAiFunctionDefinition,
}

impl OpenAiToolDefinition {
    /// Returns the declared function name.
    pub fn name(&self) -> &str {
        &self.function.name
    }

    pub(crate) fn validate(&self) -> Result<usize, OpenAiChatCompletionValidationError> {
        if self.tool_type != OpenAiToolType::Function {
            return Err(OpenAiChatCompletionValidationError::UnsupportedToolType);
        }
        validate_function_name(&self.function.name)?;
        let Some(parameters) = &self.function.parameters else {
            return Ok(0);
        };
        let schema_nesting_depth = json_nesting_depth(parameters);
        if schema_nesting_depth > MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH {
            return Err(
                OpenAiChatCompletionValidationError::ToolSchemaNestingTooDeep {
                    actual_schema_nesting_depth: schema_nesting_depth,
                    maximum_schema_nesting_depth: MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH,
                },
            );
        }
        let serialized_schema = serde_json::to_vec(parameters)
            .map_err(|_| OpenAiChatCompletionValidationError::ToolSchemaSerializationFailed)?;
        Ok(serialized_schema.len())
    }

    pub(crate) fn into_parts(
        self,
    ) -> Result<OpenAiToolDefinitionParts, OpenAiChatCompletionValidationError> {
        self.validate()?;
        let parameters_json = self.function.parameters.map_or_else(
            || Ok("{}".to_owned()),
            |parameters| {
                serde_json::to_string(&parameters)
                    .map_err(|_| OpenAiChatCompletionValidationError::ToolSchemaSerializationFailed)
            },
        )?;
        Ok(OpenAiToolDefinitionParts {
            name: self.function.name,
            description: self.function.description,
            parameters_json,
        })
    }
}

/// One validated function declaration ready for protocol translation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiToolDefinitionParts {
    /// The declared function name.
    pub name: String,
    /// Optional function description.
    pub description: Option<String>,
    /// Canonical JSON Schema for the function parameters.
    pub parameters_json: String,
}

/// OpenAI tool type accepted by this endpoint.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiToolType {
    /// A JSON-schema function tool.
    Function,
}

/// One function declaration in a tool definition.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiFunctionDefinition {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
}

/// A selected automatic tool mode or one named forced function.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OpenAiToolChoice {
    /// `auto`, `none`, or `required`.
    Mode(String),
    /// A forced function choice.
    Function {
        #[serde(rename = "type")]
        tool_type: OpenAiToolType,
        function: OpenAiFunctionChoice,
    },
}

/// The named function inside a forced tool choice.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiFunctionChoice {
    name: String,
}

impl OpenAiFunctionChoice {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

/// The validated tool-selection mode that a worker can implement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpenAiToolChoiceMode {
    /// The model chooses whether to call a declared function.
    Auto,
    /// The model must not call a function.
    None,
    /// The model must call a function.
    Required,
    /// The model must call one specific declared function.
    Function { name: String },
}

impl OpenAiToolChoice {
    pub(crate) fn into_mode(self) -> OpenAiToolChoiceMode {
        match self {
            Self::Mode(mode) if mode == "auto" => OpenAiToolChoiceMode::Auto,
            Self::Mode(mode) if mode == "none" => OpenAiToolChoiceMode::None,
            Self::Mode(_) => OpenAiToolChoiceMode::Required,
            Self::Function { function, .. } => OpenAiToolChoiceMode::Function {
                name: function.name,
            },
        }
    }
}

/// One assistant tool call retained in chat history.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiAssistantToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: OpenAiToolType,
    function: OpenAiAssistantToolFunction,
}

impl OpenAiAssistantToolCall {
    fn validate(&self) -> Result<(), OpenAiChatCompletionValidationError> {
        validate_non_empty_string("assistant tool-call ID", &self.id)?;
        if self.tool_type != OpenAiToolType::Function {
            return Err(OpenAiChatCompletionValidationError::UnsupportedToolType);
        }
        validate_function_name(&self.function.name)?;
        Ok(())
    }

    fn into_parts(self) -> OpenAiAssistantToolCallParts {
        OpenAiAssistantToolCallParts {
            id: self.id,
            name: self.function.name,
            arguments_json: self.function.arguments,
        }
    }
}

/// One validated assistant function call ready for protocol translation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiAssistantToolCallParts {
    /// Client-visible call ID used to correlate a later tool response.
    pub id: String,
    /// Declared function name.
    pub name: String,
    /// JSON-encoded function arguments.
    pub arguments_json: String,
}

/// One JSON-encoded function invocation retained in assistant history.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiAssistantToolFunction {
    name: String,
    arguments: String,
}

/// Stream-specific OpenAI options accepted by this endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiStreamOptions {
    #[serde(default)]
    pub(crate) include_usage: bool,
}

/// A single stop sequence or a bounded sequence list.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OpenAiStopSequences {
    /// One stop sequence.
    Single(String),
    /// Multiple stop sequences.
    Multiple(Vec<String>),
}

fn validate_function_name(function_name: &str) -> Result<(), OpenAiChatCompletionValidationError> {
    validate_non_empty_string("tool name", function_name)?;
    if function_name
        .bytes()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, b'_' | b'-'))
    {
        return Ok(());
    }
    Err(OpenAiChatCompletionValidationError::InvalidToolName {
        tool_name: function_name.to_owned(),
    })
}

fn validate_non_empty_string(
    field_name: &'static str,
    string_value: &str,
) -> Result<(), OpenAiChatCompletionValidationError> {
    if string_value.is_empty() {
        return Err(OpenAiChatCompletionValidationError::EmptyString { field_name });
    }
    Ok(())
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
