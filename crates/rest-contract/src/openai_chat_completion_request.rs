use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::openai_request_budget::{
    MAX_OPENAI_IMAGE_PARTS_PER_REQUEST, MAX_OPENAI_INLINE_MEDIA_BYTES,
};
use crate::{
    ChatTemplateKwargsRequestObject, EnforcedStructuredGeneration, OpenAiChatMessage,
    OpenAiChatMessageParts, OpenAiResponseFormat, OpenAiStopSequences, OpenAiStreamOptions,
    OpenAiStructuredOutput, OpenAiStructuredOutputValidationError, OpenAiStructuredOutputs,
    OpenAiStructuredOutputsValidationError, OpenAiToolChoice, OpenAiToolChoiceMode,
    OpenAiToolDefinition, OpenAiToolDefinitionParts, ReasoningRequestObject, ThinkingControls,
    ThinkingControlsError, ThinkingControlsInputs, enforced_generation_from_extra_body,
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
    /// Coding-agent spelling of the thinking budget: the agent resolves this
    /// field name from its provider compatibility configuration and falls back
    /// to it when the server declares thinking-budget support.
    #[serde(default)]
    thinking_token_budget: Option<u32>,
    /// Documented OpenAI-compatible alias for the thinking budget used by
    /// coding-agent traffic alongside `thinking_token_budget`.
    #[serde(default)]
    thinking_budget_tokens: Option<u32>,
    /// OpenRouter-style reasoning object carrying an effort level, a direct
    /// token limit, an on/off switch, and a stream-exclusion preference.
    #[serde(default)]
    reasoning: Option<ReasoningRequestObject>,
    /// Qwen-style flat switch for the thinking channel.
    #[serde(default)]
    enable_thinking: Option<bool>,
    /// vLLM-style template-kwarg block carrying the thinking toggle.
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargsRequestObject>,
    #[serde(default)]
    stop: Option<OpenAiStopSequences>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<OpenAiStreamOptions>,
    #[serde(default)]
    response_format: Option<OpenAiResponseFormat>,
    #[serde(default)]
    structured_outputs: Option<OpenAiStructuredOutputs>,
    #[serde(default)]
    guided_grammar: Option<String>,
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
        self.resolve_thinking_controls()?;
        validate_sampling_parameter("temperature", self.temperature, 0.0, 2.0)?;
        validate_sampling_parameter("top_p", self.top_p, 0.0, 1.0)?;
        self.validate_unsupported_options()?;
        if self.stop.is_some() {
            return Err(OpenAiChatCompletionValidationError::UnsupportedStopSequences);
        }
        if let Some(response_format) = &self.response_format {
            response_format.clone().into_structured_output()?;
        }
        enforced_generation_from_extra_body(
            self.structured_outputs.clone(),
            self.guided_grammar.as_deref(),
        )?;

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

    /// Resolves every submitted thinking-control spelling into one hard budget
    /// plus the stream exclusion preference, mirroring the output-token budget
    /// rule: several spellings are fine while they agree, and disagreement is
    /// a caller bug that must fail loudly instead of silently picking one.
    fn resolve_thinking_controls(
        &self,
    ) -> Result<ThinkingControls, OpenAiChatCompletionValidationError> {
        ThinkingControlsInputs {
            thinking_budget: self.thinking_budget,
            thinking_token_budget: self.thinking_token_budget,
            thinking_budget_tokens: self.thinking_budget_tokens,
            reasoning: self.reasoning.as_ref(),
            reasoning_effort: self.reasoning_effort.as_deref(),
            enable_thinking: self.enable_thinking,
            chat_template_kwargs: self.chat_template_kwargs.as_ref(),
        }
        .resolve()
        .map_err(OpenAiChatCompletionValidationError::ThinkingControls)
    }

    /// Validates and consumes this REST DTO into protocol-neutral request parts.
    pub fn into_parts(
        self,
    ) -> Result<OpenAiChatCompletionRequestParts, OpenAiChatCompletionValidationError> {
        self.validate()?;
        let maximum_output_tokens = self.maximum_output_tokens();
        let requested_maximum_output_tokens = self.max_completion_tokens.or(self.max_tokens);
        let includes_usage_in_stream = self.includes_usage_in_stream();
        let structured_output = self
            .response_format
            .clone()
            .map(OpenAiResponseFormat::into_structured_output)
            .transpose()?
            .flatten();
        let enforced_structured_generation = crate::enforced_generation_from_extra_body(
            self.structured_outputs.clone(),
            self.guided_grammar.as_deref(),
        )?;
        let thinking_controls = self.resolve_thinking_controls()?;
        // Inline media accumulates across every user message so the portable
        // media budget is per request, not per message. Each image was already
        // size-checked individually during decode; this accumulation rejects a
        // request whose combined media exceeds the budget the mainstream
        // providers enforce on the whole request.
        let mut decoded_inline_media_bytes = 0usize;
        let mut messages = Vec::with_capacity(self.messages.len());
        for chat_message in self.messages {
            let message_parts = chat_message.into_parts()?;
            if let OpenAiChatMessageParts::User { images, .. } = &message_parts {
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
    /// The request carried more image parts than the portable image-count budget.
    #[error(
        "the request carries {actual_image_part_count} image parts, exceeding the {maximum_image_part_count}-image budget by {excess_image_part_count} (mirrors the 100-image-per-request limit Anthropic documents for 200k-context models)"
    )]
    ImagePartCountExceedsBudget {
        actual_image_part_count: usize,
        maximum_image_part_count: usize,
        excess_image_part_count: usize,
    },
    /// The request's total decoded inline media exceeded the portable media budget.
    #[error(
        "the request carries {actual_inline_media_bytes} decoded image bytes, exceeding the {maximum_inline_media_bytes}-byte inline-media budget by {excess_inline_media_bytes} bytes (mirrors the 20 MB total inline request size of the Gemini API)"
    )]
    InlineMediaExceedsBudget {
        actual_inline_media_bytes: usize,
        maximum_inline_media_bytes: usize,
        excess_inline_media_bytes: usize,
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
        matches!(
            self,
            Self::ImageTooLarge { .. }
                | Self::ImagePartCountExceedsBudget { .. }
                | Self::InlineMediaExceedsBudget { .. }
        )
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

fn validate_image_part_count(
    image_part_count: usize,
) -> Result<(), OpenAiChatCompletionValidationError> {
    if image_part_count > MAX_OPENAI_IMAGE_PARTS_PER_REQUEST {
        return Err(
            OpenAiChatCompletionValidationError::ImagePartCountExceedsBudget {
                actual_image_part_count: image_part_count,
                maximum_image_part_count: MAX_OPENAI_IMAGE_PARTS_PER_REQUEST,
                excess_image_part_count: image_part_count - MAX_OPENAI_IMAGE_PARTS_PER_REQUEST,
            },
        );
    }
    Ok(())
}

fn validate_inline_media_total(
    decoded_inline_media_bytes: usize,
) -> Result<(), OpenAiChatCompletionValidationError> {
    if decoded_inline_media_bytes > MAX_OPENAI_INLINE_MEDIA_BYTES {
        return Err(
            OpenAiChatCompletionValidationError::InlineMediaExceedsBudget {
                actual_inline_media_bytes: decoded_inline_media_bytes,
                maximum_inline_media_bytes: MAX_OPENAI_INLINE_MEDIA_BYTES,
                excess_inline_media_bytes: decoded_inline_media_bytes
                    - MAX_OPENAI_INLINE_MEDIA_BYTES,
            },
        );
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
