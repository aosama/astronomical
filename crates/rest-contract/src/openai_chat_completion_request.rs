use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::{
    OpenAiChatMessage, OpenAiChatMessageParts, OpenAiStopSequences, OpenAiStreamOptions,
    OpenAiToolChoice, OpenAiToolChoiceMode, OpenAiToolDefinition, OpenAiToolDefinitionParts,
};

/// The maximum accepted nesting depth of a function JSON Schema.
pub const MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH: usize = 32;
/// Maximum generated-token budget representable by the current worker protocol.
/// The model-serving layer still validates prompt plus output against model context.
pub const MAX_OPENAI_OUTPUT_TOKENS: u32 = u16::MAX as u32;
/// The fallback generated-token budget when a client does not send one.
pub const DEFAULT_OPENAI_OUTPUT_TOKENS: u32 = 1_024;

/// One bounded OpenAI-compatible Chat Completions request.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAiChatCompletionRequest {
    model: String,
    messages: Vec<OpenAiChatMessage>,
    #[serde(default)]
    tools: Vec<OpenAiToolDefinition>,
    #[serde(default)]
    tool_choice: Option<OpenAiToolChoice>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// Maximum tokens the model may spend inside the thinking block before being
    /// forced to close it and start the visible response. When `None`, the model
    /// thinks freely up to `max_tokens`.
    #[serde(default)]
    thinking_budget: Option<u32>,
    #[serde(default)]
    stop: Option<OpenAiStopSequences>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<OpenAiStreamOptions>,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

impl OpenAiChatCompletionRequest {
    /// Validates request bounds before the supervisor sends structured data to the worker.
    pub fn validate(&self) -> Result<(), OpenAiChatCompletionValidationError> {
        if let Some((field_name, _)) = self.unknown_fields.first_key_value() {
            return Err(OpenAiChatCompletionValidationError::UnknownField {
                field_name: field_name.clone(),
            });
        }
        validate_non_empty_string("model", &self.model)?;
        if self.messages.is_empty() {
            return Err(OpenAiChatCompletionValidationError::EmptyMessages);
        }
        for chat_message in &self.messages {
            chat_message.validate()?;
        }

        for tool_definition in &self.tools {
            tool_definition.validate()?;
        }

        self.validate_tool_choice()?;
        self.validate_output_token_budget()?;
        validate_sampling_parameter("temperature", self.temperature, 0.0, 2.0)?;
        validate_sampling_parameter("top_p", self.top_p, 0.0, 1.0)?;
        self.validate_unsupported_options()?;
        if self.stop.is_some() {
            return Err(OpenAiChatCompletionValidationError::UnsupportedStopSequences);
        }

        Ok(())
    }

    /// Returns the requested model ID after request validation.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns ordered conversation history after request validation.
    pub fn messages(&self) -> &[OpenAiChatMessage] {
        &self.messages
    }

    /// Returns declared callable tools after request validation.
    pub fn tools(&self) -> &[OpenAiToolDefinition] {
        &self.tools
    }

    /// Returns whether the caller requested an SSE response.
    pub fn stream(&self) -> bool {
        self.stream
    }

    /// Returns whether the final streamed chunk must contain usage information.
    pub fn includes_usage_in_stream(&self) -> bool {
        self.stream_options
            .as_ref()
            .is_some_and(|stream_options| stream_options.include_usage)
    }

    /// Returns the selected generated-token budget after request validation.
    pub fn maximum_output_tokens(&self) -> u32 {
        self.max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(DEFAULT_OPENAI_OUTPUT_TOKENS)
    }

    /// Returns the optional deterministic sampling seed after request validation.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Returns the optional thinking-token budget after request validation.
    pub fn thinking_budget(&self) -> Option<u32> {
        self.thinking_budget
    }

    /// Validates and consumes this REST DTO into protocol-neutral request parts.
    pub fn into_parts(
        self,
    ) -> Result<OpenAiChatCompletionRequestParts, OpenAiChatCompletionValidationError> {
        self.validate()?;
        let maximum_output_tokens = self.maximum_output_tokens();
        let includes_usage_in_stream = self.includes_usage_in_stream();
        Ok(OpenAiChatCompletionRequestParts {
            model: self.model,
            messages: self
                .messages
                .into_iter()
                .map(OpenAiChatMessage::into_parts)
                .collect::<Result<Vec<_>, _>>()?,
            tools: self
                .tools
                .into_iter()
                .map(OpenAiToolDefinition::into_parts)
                .collect::<Result<Vec<_>, _>>()?,
            tool_choice: self
                .tool_choice
                .map(OpenAiToolChoice::into_mode)
                .unwrap_or(OpenAiToolChoiceMode::Auto),
            maximum_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            thinking_budget: self.thinking_budget,
            stream: self.stream,
            includes_usage_in_stream,
        })
    }

    fn validate_output_token_budget(&self) -> Result<(), OpenAiChatCompletionValidationError> {
        if let (Some(max_tokens), Some(max_completion_tokens)) =
            (self.max_tokens, self.max_completion_tokens)
            && max_tokens != max_completion_tokens
        {
            return Err(
                OpenAiChatCompletionValidationError::ConflictingOutputTokenLimits {
                    max_tokens,
                    max_completion_tokens,
                },
            );
        }

        let actual_output_tokens = self.maximum_output_tokens();
        if actual_output_tokens == 0 || actual_output_tokens > MAX_OPENAI_OUTPUT_TOKENS {
            return Err(
                OpenAiChatCompletionValidationError::OutputTokenCountOutOfRange {
                    actual_output_tokens,
                    maximum_output_tokens: MAX_OPENAI_OUTPUT_TOKENS,
                },
            );
        }
        Ok(())
    }

    fn validate_tool_choice(&self) -> Result<(), OpenAiChatCompletionValidationError> {
        let Some(tool_choice) = &self.tool_choice else {
            return Ok(());
        };
        let declared_tool_names = self
            .tools
            .iter()
            .map(OpenAiToolDefinition::name)
            .collect::<Vec<_>>();
        match tool_choice {
            OpenAiToolChoice::Mode(mode) if matches!(mode.as_str(), "auto" | "none") => Ok(()),
            OpenAiToolChoice::Mode(mode) => {
                Err(OpenAiChatCompletionValidationError::UnsupportedToolChoice {
                    mode: mode.clone(),
                })
            }
            OpenAiToolChoice::Function { function, .. }
                if declared_tool_names.contains(&function.name()) =>
            {
                Err(
                    OpenAiChatCompletionValidationError::UnsupportedForcedToolChoice {
                        function_name: function.name().to_owned(),
                    },
                )
            }
            OpenAiToolChoice::Function { function, .. } => Err(
                OpenAiChatCompletionValidationError::ToolChoiceNamesUnknownFunction {
                    function_name: function.name().to_owned(),
                },
            ),
        }
    }

    fn validate_unsupported_options(&self) -> Result<(), OpenAiChatCompletionValidationError> {
        if self.frequency_penalty.is_some() {
            return Err(OpenAiChatCompletionValidationError::UnsupportedOption {
                option_name: "frequency_penalty",
            });
        }
        if self.presence_penalty.is_some() {
            return Err(OpenAiChatCompletionValidationError::UnsupportedOption {
                option_name: "presence_penalty",
            });
        }
        if self.store.is_some() {
            return Err(OpenAiChatCompletionValidationError::UnsupportedOption {
                option_name: "store",
            });
        }
        Ok(())
    }
}

/// Validated OpenAI request data ready for translation at the supervisor boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiChatCompletionRequestParts {
    /// Exact model ID targeted by the client.
    pub model: String,
    /// Ordered text-only chat history.
    pub messages: Vec<OpenAiChatMessageParts>,
    /// Declared function tools.
    pub tools: Vec<OpenAiToolDefinitionParts>,
    /// Requested tool-selection policy.
    pub tool_choice: OpenAiToolChoiceMode,
    /// Bounded generated-token budget.
    pub maximum_output_tokens: u32,
    /// Optional OpenAI-compatible temperature.
    pub temperature: Option<f32>,
    /// Optional OpenAI-compatible nucleus threshold.
    pub top_p: Option<f32>,
    /// Optional deterministic sampler seed.
    pub seed: Option<u64>,
    /// Maximum tokens the model may spend inside the thinking block.
    pub thinking_budget: Option<u32>,
    /// Whether the client requested SSE streaming.
    pub stream: bool,
    /// Whether the stream's terminal event must carry usage.
    pub includes_usage_in_stream: bool,
}

/// A request rejected before worker admission by the public OpenAI contract.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum OpenAiChatCompletionValidationError {
    /// A required string was empty.
    #[error("{field_name} must not be empty")]
    EmptyString { field_name: &'static str },
    /// The request did not contain any messages.
    #[error("messages must not be empty")]
    EmptyMessages,
    /// A multipart content list was empty.
    #[error("content parts must not be empty")]
    EmptyContentParts,
    /// The text-only endpoint received another modality.
    #[error("content part type '{content_part_type}' is not supported by the text-only endpoint")]
    UnsupportedContentPart { content_part_type: &'static str },
    /// An image URL used a scheme other than `data:image/...;base64,...`.
    #[error("only data:image base64 URIs are supported for image input")]
    UnsupportedImageUrlScheme,
    /// An image URL had a non-image MIME type.
    #[error("image MIME type '{actual_mime_type}' is not an image type")]
    UnsupportedImageMimeType { actual_mime_type: String },
    /// A data URI was malformed (missing comma or metadata).
    #[error("the data URI is malformed")]
    MalformedDataUri,
    /// A data URI base64 payload could not be decoded.
    #[error("the data URI base64 payload is invalid")]
    InvalidBase64,
    /// A decoded image exceeded the maximum accepted byte size.
    #[error("decoded image is {actual_bytes} bytes, exceeding the {maximum_bytes} byte limit")]
    ImageTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    /// An assistant history record contained no answer, reasoning, or tool call.
    #[error("assistant messages must contain content, reasoning, or tool calls")]
    EmptyAssistantMessage,
    /// A tool used an unsupported type.
    #[error("only function tools are supported")]
    UnsupportedToolType,
    /// A tool name did not match the strict portable function-name grammar.
    #[error("tool name '{tool_name}' is invalid")]
    InvalidToolName { tool_name: String },
    /// A tool schema was too deeply nested.
    #[error(
        "tool schema nesting depth is {actual_schema_nesting_depth}, exceeding {maximum_schema_nesting_depth}"
    )]
    ToolSchemaNestingTooDeep {
        actual_schema_nesting_depth: usize,
        maximum_schema_nesting_depth: usize,
    },
    /// A locally decoded schema unexpectedly could not serialize for its byte check.
    #[error("tool schema could not be serialized for bounded validation")]
    ToolSchemaSerializationFailed,
    /// The client selected an unsupported automatic tool mode.
    #[error("tool choice mode '{mode}' is unsupported")]
    UnsupportedToolChoice { mode: String },
    /// A forced function was not among the declared tools.
    #[error("tool choice names undeclared function '{function_name}'")]
    ToolChoiceNamesUnknownFunction { function_name: String },
    /// A declared function cannot yet be deterministically forced.
    #[error("forcing function '{function_name}' is unsupported")]
    UnsupportedForcedToolChoice { function_name: String },
    /// `max_tokens` and `max_completion_tokens` disagreed.
    #[error(
        "max_tokens ({max_tokens}) conflicts with max_completion_tokens ({max_completion_tokens})"
    )]
    ConflictingOutputTokenLimits {
        max_tokens: u32,
        max_completion_tokens: u32,
    },
    /// The output token budget was zero or too large for the worker representation.
    #[error(
        "output token count is {actual_output_tokens}, outside the 1..={maximum_output_tokens} token range"
    )]
    OutputTokenCountOutOfRange {
        actual_output_tokens: u32,
        maximum_output_tokens: u32,
    },
    /// A sampling setting fell outside its OpenAI-compatible range.
    #[error("{parameter_name} is outside the supported range {minimum}..={maximum}")]
    SamplingParameterOutOfRange {
        parameter_name: &'static str,
        minimum: String,
        maximum: String,
    },
    /// Caller-defined stop sequences are not implemented by the initial endpoint.
    #[error("caller-supplied stop sequences are unsupported")]
    UnsupportedStopSequences,
    /// A recognized OpenAI-compatible request option is not implemented yet.
    #[error("request option '{option_name}' is unsupported")]
    UnsupportedOption { option_name: &'static str },
    /// An unrecognized request field was supplied.
    #[error("request field '{field_name}' is unknown")]
    UnknownField { field_name: String },
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

fn validate_sampling_parameter(
    parameter_name: &'static str,
    parameter_value: Option<f32>,
    minimum: f32,
    maximum: f32,
) -> Result<(), OpenAiChatCompletionValidationError> {
    let Some(parameter_value) = parameter_value else {
        return Ok(());
    };
    if parameter_value.is_finite() && parameter_value >= minimum && parameter_value <= maximum {
        return Ok(());
    }
    Err(
        OpenAiChatCompletionValidationError::SamplingParameterOutOfRange {
            parameter_name,
            minimum: minimum.to_string(),
            maximum: maximum.to_string(),
        },
    )
}
