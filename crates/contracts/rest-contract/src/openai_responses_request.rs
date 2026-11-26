use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::request_bounds::{
    RequestBoundsError, validate_output_token_count, validate_sampling_parameter,
};
use crate::structured_generation_fields::StructuredGenerationFields;
use crate::thinking_control_fields::ThinkingControlFields;
use crate::{
    DEFAULT_OPENAI_OUTPUT_TOKENS, EnforcedStructuredGeneration, OpenAiResponseFunctionTool,
    OpenAiResponseInput, OpenAiResponseInputParts, OpenAiResponseRequestConfiguration,
    OpenAiResponseToolChoice, OpenAiResponseToolChoiceParts, OpenAiResponseToolDefinition,
    OpenAiResponseToolDefinitionParts, OpenAiStructuredOutput,
    OpenAiStructuredOutputValidationError, OpenAiStructuredOutputsValidationError,
    ThinkingControlsError, merge_structured_output_requests,
    structured_output_from_responses_text_format,
};

/// One bounded request to the local OpenAI-compatible Responses endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAiResponsesRequest {
    model: String,
    input: OpenAiResponseInput,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    tools: Vec<OpenAiResponseToolDefinition>,
    #[serde(default)]
    tool_choice: Option<OpenAiResponseToolChoice>,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    background: Option<bool>,
    #[serde(default)]
    truncation: Option<String>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    safety_identifier: Option<String>,
    /// OpenAI's own request field names, kept exactly as OpenAI spells them:
    /// clients such as GitHub Copilot send `prompt_cache_key` on every
    /// request, and any other spelling makes them fail as an unknown field.
    /// The key is accepted and not used; `astro_ssd_prompt_cache` matches
    /// prompts by their content instead.
    #[serde(default)]
    prompt_cache_key: Option<String>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    conversation: Option<Value>,
    #[serde(default)]
    context_management: Option<Value>,
    #[serde(default)]
    include: Option<Value>,
    #[serde(default)]
    moderation: Option<Value>,
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    prompt_cache_options: Option<Value>,
    #[serde(default)]
    prompt_cache_retention: Option<String>,
    #[serde(default)]
    stream_options: Option<Value>,
    #[serde(default)]
    text: Option<Value>,
    #[serde(default)]
    top_logprobs: Option<u8>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
    /// Declared before `unknown_fields`, so their spellings are taken first
    /// and never reach the catch-all map, which `into_parts` refuses.
    #[serde(flatten)]
    thinking_controls: ThinkingControlFields,
    #[serde(flatten)]
    structured_generation: StructuredGenerationFields,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

impl OpenAiResponsesRequest {
    /// Validates and consumes this public request into protocol-neutral parts.
    pub fn into_parts(self) -> Result<OpenAiResponsesRequestParts, OpenAiResponsesValidationError> {
        if let Some((field_name, _)) = self.unknown_fields.first_key_value() {
            return Err(OpenAiResponsesValidationError::UnknownField {
                field_name: field_name.clone(),
            });
        }
        if self.model.is_empty() {
            return Err(OpenAiResponsesValidationError::EmptyModel);
        }
        let maximum_output_tokens = self
            .max_output_tokens
            .unwrap_or(DEFAULT_OPENAI_OUTPUT_TOKENS);
        let requested_maximum_output_tokens = self.max_output_tokens;
        validate_output_token_count(maximum_output_tokens)?;
        validate_sampling_parameter("temperature", self.temperature, 0.0, 2.0)?;
        validate_sampling_parameter("top_p", self.top_p, 0.0, 1.0)?;
        let thinking_controls = self.thinking_controls.resolve()?;
        let structured_output = merge_structured_output_requests(
            self.structured_generation.response_format_output()?,
            structured_output_from_responses_text_format(self.text.as_ref())?,
        )?;
        let enforced_structured_generation = self.structured_generation.enforced_generation()?;
        validate_compatibility_fields(&self)?;
        let tools = self
            .tools
            .into_iter()
            .map(OpenAiResponseToolDefinition::into_parts)
            .collect::<Result<Vec<_>, _>>()?;
        let tool_choice = self
            .tool_choice
            .map(OpenAiResponseToolChoice::into_parts)
            .transpose()?
            .unwrap_or(OpenAiResponseToolChoiceParts::Auto);
        Ok(OpenAiResponsesRequestParts {
            model: self.model,
            input: self.input.into_parts()?,
            instructions: self.instructions,
            tools,
            tool_choice,
            metadata: self.metadata,
            maximum_output_tokens,
            requested_maximum_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stream: self.stream,
            thinking_budget: thinking_controls.budget,
            reasoning_excluded: thinking_controls.reasoning_excluded,
            structured_output,
            enforced_structured_generation,
        })
    }
}

/// Validated Responses request data ready for daemon translation.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiResponsesRequestParts {
    pub model: String,
    pub input: OpenAiResponseInputParts,
    pub instructions: Option<String>,
    pub tools: Vec<OpenAiResponseToolDefinitionParts>,
    pub tool_choice: OpenAiResponseToolChoiceParts,
    pub metadata: BTreeMap<String, String>,
    pub maximum_output_tokens: u32,
    pub requested_maximum_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: bool,
    pub thinking_budget: Option<u32>,
    /// Whether reasoning deltas must be withheld from the response stream.
    pub reasoning_excluded: bool,
    pub structured_output: Option<OpenAiStructuredOutput>,
    pub enforced_structured_generation: Option<EnforcedStructuredGeneration>,
}

impl OpenAiResponsesRequestParts {
    /// Copies only the bounded settings required in returned Response objects.
    #[must_use]
    pub fn response_configuration(&self) -> OpenAiResponseRequestConfiguration {
        OpenAiResponseRequestConfiguration {
            metadata: self.metadata.clone(),
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.requested_maximum_output_tokens,
            tool_choice: self.tool_choice.kind_name(),
            tools: self
                .tools
                .iter()
                .map(|function_tool| {
                    OpenAiResponseFunctionTool::new(
                        function_tool.name.clone(),
                        function_tool.description.clone(),
                        function_tool.parameters.clone(),
                        function_tool.strict,
                    )
                })
                .collect(),
        }
    }
}

/// A request rejected before worker admission by the Responses contract.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OpenAiResponsesValidationError {
    #[error("model must not be empty")]
    EmptyModel,
    #[error("input items must not be empty")]
    EmptyInputItems,
    #[error("message content parts must not be empty")]
    EmptyContentParts,
    #[error("image input is supported only in user messages and tool results")]
    ImageInputOutsideUserMessage,
    /// A reasoning-control spelling was contradictory or unrecognized.
    #[error(transparent)]
    ThinkingControls(#[from] ThinkingControlsError),
    /// The request fell outside a bound every generation request shares.
    #[error(transparent)]
    RequestBounds(#[from] RequestBoundsError),
    #[error("invalid image input: {0}")]
    ImageInput(#[source] crate::OpenAiChatCompletionValidationError),
    #[error("encrypted foreign reasoning cannot be replayed locally")]
    UnsupportedReasoningReplay,
    #[error("response input item type '{input_item_type}' is unsupported")]
    UnsupportedInputItem { input_item_type: String },
    /// An `input_image` part arrived without the inline image it describes.
    #[error("an input_image part needs an image_url holding a data:image/... URI")]
    MissingImageUrl,
    #[error("tool name '{tool_name}' is invalid")]
    InvalidToolName { tool_name: String },
    #[error(
        "tool schema nesting depth is {actual_schema_nesting_depth}, exceeding {maximum_schema_nesting_depth}"
    )]
    ToolSchemaNestingTooDeep {
        actual_schema_nesting_depth: usize,
        maximum_schema_nesting_depth: usize,
    },
    #[error("request option '{option_name}' is unsupported")]
    UnsupportedOption { option_name: &'static str },
    #[error("metadata exceeds the supported 16-entry limit")]
    MetadataEntryCountExceeded,
    #[error("metadata key or value exceeds the supported length")]
    MetadataTextTooLong,
    #[error("request field '{field_name}' is unknown")]
    UnknownField { field_name: String },
    #[error(transparent)]
    StructuredOutput(#[from] OpenAiStructuredOutputValidationError),
    #[error(transparent)]
    StructuredOutputs(#[from] OpenAiStructuredOutputsValidationError),
}

impl OpenAiResponsesValidationError {
    /// Whether this rejection is a request-size rejection (HTTP 413 under the
    /// portable request budget) rather than a malformed-request one (HTTP 400).
    #[must_use]
    pub fn is_request_too_large(&self) -> bool {
        is_responses_size_rejection(self)
    }
}

/// The size rejections: a shared bound on the whole request's images, or one
/// image over its own budget, reported through the image-input error.
fn is_responses_size_rejection(validation_error: &OpenAiResponsesValidationError) -> bool {
    match validation_error {
        OpenAiResponsesValidationError::RequestBounds(bounds_error) => {
            bounds_error.is_request_too_large()
        }
        OpenAiResponsesValidationError::ImageInput(chat_error) => chat_error.is_request_too_large(),
        _ => false,
    }
}

fn validate_compatibility_fields(
    request: &OpenAiResponsesRequest,
) -> Result<(), OpenAiResponsesValidationError> {
    if let Some(option_name) = first_unsupported_option_name(request) {
        return Err(OpenAiResponsesValidationError::UnsupportedOption { option_name });
    }
    validate_metadata_limits(&request.metadata)
}

/// The first request option this server refuses, checked in a fixed order so
/// a request with several unsupported options always names the same one.
fn first_unsupported_option_name(request: &OpenAiResponsesRequest) -> Option<&'static str> {
    if request.store == Some(true) {
        return Some("store=true");
    }
    for (option_name, is_present) in [
        (
            "previous_response_id",
            request.previous_response_id.is_some(),
        ),
        ("conversation", request.conversation.is_some()),
        ("context_management", request.context_management.is_some()),
        ("moderation", request.moderation.is_some()),
        ("prompt", request.prompt.is_some()),
        (
            "prompt_cache_options",
            request.prompt_cache_options.is_some(),
        ),
        (
            "prompt_cache_retention",
            request.prompt_cache_retention.is_some(),
        ),
    ] {
        if is_present {
            return Some(option_name);
        }
    }
    first_unsupported_option_setting(request)
}

/// Options that are accepted when absent or at their default and refused once
/// they are set to a value this server does not support.
fn first_unsupported_option_setting(request: &OpenAiResponsesRequest) -> Option<&'static str> {
    if request.parallel_tool_calls == Some(false) {
        return Some("parallel_tool_calls=false");
    }
    if request
        .top_logprobs
        .is_some_and(|top_logprobs| top_logprobs > 0)
    {
        return Some("top_logprobs");
    }
    if request
        .text
        .as_ref()
        .is_some_and(|text_configuration| text_configuration.get("verbosity").is_some())
    {
        return Some("text.verbosity");
    }
    if request.background == Some(true) {
        return Some("background=true");
    }
    if request
        .truncation
        .as_deref()
        .is_some_and(|truncation| truncation != "disabled")
    {
        return Some("truncation");
    }
    if request
        .service_tier
        .as_deref()
        .is_some_and(|service_tier| !matches!(service_tier, "auto" | "default"))
    {
        return Some("service_tier");
    }
    None
}

/// Rejects metadata over the Responses API limits: at most 16 entries, with keys of at most 64 bytes and values of at most 512 bytes.
fn validate_metadata_limits(
    metadata: &BTreeMap<String, String>,
) -> Result<(), OpenAiResponsesValidationError> {
    if metadata.len() > 16 {
        return Err(OpenAiResponsesValidationError::MetadataEntryCountExceeded);
    }
    if metadata
        .iter()
        .any(|(metadata_name, metadata_text)| metadata_name.len() > 64 || metadata_text.len() > 512)
    {
        return Err(OpenAiResponsesValidationError::MetadataTextTooLong);
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod prompt_cache_fields;

#[cfg(test)]
mod tool_output_images;

#[cfg(test)]
mod compatibility_fields;
