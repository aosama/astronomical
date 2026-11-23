use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::request_bounds::{
    RequestBoundsError, validate_image_part_count, validate_inline_media_total,
    validate_output_token_count, validate_sampling_parameter,
};
use crate::structured_generation_fields::StructuredGenerationFields;
use crate::thinking_control_fields::ThinkingControlFields;
use crate::{
    EnforcedStructuredGeneration, OpenAiChatMessage, OpenAiChatMessageParts, OpenAiStopSequences,
    OpenAiStreamOptions, OpenAiStructuredOutput, OpenAiStructuredOutputValidationError,
    OpenAiStructuredOutputsValidationError, OpenAiToolChoice, OpenAiToolChoiceMode,
    OpenAiToolDefinition, OpenAiToolDefinitionParts, ThinkingControlsError,
};

/// The fallback generated-token budget when an OpenAI client does not send
/// one. The Anthropic surface has no fallback: `max_tokens` is required.
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
    stop: Option<OpenAiStopSequences>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<OpenAiStreamOptions>,
    /// Declared before `unknown_fields`, so their spellings are taken first
    /// and never reach the catch-all map.
    #[serde(flatten)]
    thinking_controls: ThinkingControlFields,
    #[serde(flatten)]
    structured_generation: StructuredGenerationFields,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

impl OpenAiChatCompletionRequest {
    /// Validates request bounds before the daemon sends structured data to the worker.
    /// Unknown top-level fields are deliberately absorbed and ignored: callers replay
    /// provider-specific options that the local endpoint has no use for, and one
    /// unsupported option must not reject the whole request (#772).
    pub fn validate(&self) -> Result<(), OpenAiChatCompletionValidationError> {
        validate_non_empty_string("model", &self.model)?;
        if self.messages.is_empty() {
            return Err(OpenAiChatCompletionValidationError::EmptyMessages);
        }
        for chat_message in &self.messages {
            chat_message.validate()?;
        }
        validate_image_part_count(
            self.messages
                .iter()
                .map(OpenAiChatMessage::image_url_part_count)
                .sum(),
        )?;

        for tool_definition in &self.tools {
            tool_definition.validate()?;
        }

        self.validate_tool_choice()?;
        self.validate_output_token_budget()?;
        self.thinking_controls.resolve()?;
        validate_sampling_parameter("temperature", self.temperature, 0.0, 2.0)?;
        validate_sampling_parameter("top_p", self.top_p, 0.0, 1.0)?;
        self.validate_unsupported_options()?;
        if self.stop.is_some() {
            return Err(OpenAiChatCompletionValidationError::UnsupportedStopSequences);
        }
        self.structured_generation.response_format_output()?;
        self.structured_generation.enforced_generation()?;

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

    /// Validates and consumes this REST DTO into protocol-neutral request parts.
    pub fn into_parts(
        self,
    ) -> Result<OpenAiChatCompletionRequestParts, OpenAiChatCompletionValidationError> {
        self.validate()?;
        let maximum_output_tokens = self.maximum_output_tokens();
        let requested_maximum_output_tokens = self.max_completion_tokens.or(self.max_tokens);
        let includes_usage_in_stream = self.includes_usage_in_stream();
        let structured_output = self.structured_generation.response_format_output()?;
        let enforced_structured_generation = self.structured_generation.enforced_generation()?;
        let thinking_controls = self.thinking_controls.resolve()?;
        // Inline media accumulates across every user and tool message so the
        // portable media budget is per request, not per message. Each image was already
        // size-checked individually during decode; this accumulation rejects a
        // request whose combined media exceeds the budget the mainstream
        // providers enforce on the whole request.
        let mut decoded_inline_media_bytes = 0usize;
        let mut messages = Vec::with_capacity(self.messages.len());
        for chat_message in self.messages {
            let message_parts = chat_message.into_parts()?;
            if let OpenAiChatMessageParts::User { images, .. }
            | OpenAiChatMessageParts::Tool { images, .. } = &message_parts
            {
                for decoded_image in images {
                    decoded_inline_media_bytes += decoded_image.decoded_bytes().len();
                }
            }
            messages.push(message_parts);
        }
        validate_inline_media_total(decoded_inline_media_bytes)?;
        Ok(OpenAiChatCompletionRequestParts {
            model: self.model,
            messages,
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
            requested_maximum_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            thinking_budget: thinking_controls.budget,
            reasoning_excluded: thinking_controls.reasoning_excluded,
            stream: self.stream,
            includes_usage_in_stream,
            structured_output,
            enforced_structured_generation,
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

        validate_output_token_count(self.maximum_output_tokens())?;
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
        // `store: false` is the documented protocol default ("do not store output
        // for distillation or retrieval") and describes exactly what this local
        // server does anyway, so a protocol-faithful client that spells out the
        // default must not be rejected — OpenCode sends `store: false` on every
        // request, and rejecting it failed their entire conversation. `store:
        // true` requests an unimplemented server-side store, so it stays an
        // explicit rejection, mirroring the Responses surface.
        if self.store == Some(true) {
            return Err(OpenAiChatCompletionValidationError::UnsupportedOption {
                option_name: "store=true",
            });
        }
        Ok(())
    }
}

/// Validated OpenAI request data ready for translation at the daemon boundary.
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
    /// Caller-supplied generated-token budget, preserving omission for model defaults.
    pub requested_maximum_output_tokens: Option<u32>,
    /// Optional OpenAI-compatible temperature.
    pub temperature: Option<f32>,
    /// Optional OpenAI-compatible nucleus threshold.
    pub top_p: Option<f32>,
    /// Optional deterministic sampler seed.
    pub seed: Option<u64>,
    /// Maximum tokens the model may spend inside the thinking block; `Some(0)`
    /// is an explicit disable that renders the thinking channel closed.
    pub thinking_budget: Option<u32>,
    /// Whether reasoning deltas must be withheld from the response stream.
    pub reasoning_excluded: bool,
    /// Whether the client requested SSE streaming.
    pub stream: bool,
    /// Whether the stream's terminal event must carry usage.
    pub includes_usage_in_stream: bool,
    /// Validated OpenAI structured-output request, when the client asked for JSON.
    pub structured_output: Option<OpenAiStructuredOutput>,
    /// Extra-body constraint that must be token-masked or the request fails.
    pub enforced_structured_generation: Option<EnforcedStructuredGeneration>,
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
    #[error(
        "decoded image is {actual_bytes} bytes, exceeding the {maximum_bytes}-byte per-image budget by {excess_bytes} bytes (mirrors the 7 MB inline per-file limit of Google Cloud Gemini Enterprise)"
    )]
    ImageTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
        excess_bytes: usize,
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
    /// A reasoning-control spelling was contradictory or unrecognized.
    #[error(transparent)]
    ThinkingControls(#[from] ThinkingControlsError),
    /// The request fell outside a bound every generation request shares.
    #[error(transparent)]
    RequestBounds(#[from] RequestBoundsError),
    /// Caller-defined stop sequences are not implemented by the initial endpoint.
    #[error("caller-supplied stop sequences are unsupported")]
    UnsupportedStopSequences,
    /// A recognized OpenAI-compatible request option is not implemented yet.
    #[error("request option '{option_name}' is unsupported")]
    UnsupportedOption { option_name: &'static str },
    /// `response_format` failed public structured-output validation.
    #[error(transparent)]
    StructuredOutput(#[from] OpenAiStructuredOutputValidationError),
    #[error(transparent)]
    StructuredOutputs(#[from] OpenAiStructuredOutputsValidationError),
}

impl OpenAiChatCompletionValidationError {
    /// Whether this rejection is a request-size rejection (HTTP 413 under the
    /// portable request budget) rather than a malformed-request one (HTTP 400).
    #[must_use]
    pub fn is_request_too_large(&self) -> bool {
        is_chat_size_rejection(self)
    }
}

/// The size rejections: one image over its own budget, or a shared bound on
/// the whole request's images.
fn is_chat_size_rejection(validation_error: &OpenAiChatCompletionValidationError) -> bool {
    match validation_error {
        OpenAiChatCompletionValidationError::ImageTooLarge { .. } => true,
        OpenAiChatCompletionValidationError::RequestBounds(bounds_error) => {
            bounds_error.is_request_too_large()
        }
        _ => false,
    }
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

#[cfg(test)]
mod tests;
