use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::{
    DEFAULT_OPENAI_OUTPUT_TOKENS, MAX_OPENAI_OUTPUT_TOKENS, OpenAiResponseFormat,
    OpenAiResponseFunctionTool, OpenAiResponseInput, OpenAiResponseInputParts,
    OpenAiResponseRequestConfiguration, OpenAiResponseToolChoice, OpenAiResponseToolChoiceParts,
    OpenAiResponseToolDefinition, OpenAiResponseToolDefinitionParts, OpenAiStructuredOutput,
    OpenAiStructuredOutputValidationError, merge_structured_output_requests,
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
    reasoning: Option<Value>,
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
    #[serde(default)]
    thinking_budget: Option<u32>,
    #[serde(default)]
    response_format: Option<OpenAiResponseFormat>,
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
        if maximum_output_tokens == 0 || maximum_output_tokens > MAX_OPENAI_OUTPUT_TOKENS {
            return Err(OpenAiResponsesValidationError::OutputTokenCountOutOfRange {
                actual_output_tokens: maximum_output_tokens,
                maximum_output_tokens: MAX_OPENAI_OUTPUT_TOKENS,
            });
        }
        validate_sampling_parameter("temperature", self.temperature, 0.0, 2.0)?;
        validate_sampling_parameter("top_p", self.top_p, 0.0, 1.0)?;
        let structured_output = merge_structured_output_requests(
            self.response_format
                .clone()
                .map(OpenAiResponseFormat::into_structured_output)
                .transpose()?
                .flatten(),
            structured_output_from_responses_text_format(self.text.as_ref())?,
        )?;
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
            thinking_budget: self.thinking_budget,
            structured_output,
        })
    }
}

/// Validated Responses request data ready for supervisor translation.
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
    pub structured_output: Option<OpenAiStructuredOutput>,
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
    #[error("image input is supported only in user messages")]
    ImageInputOutsideUserMessage,
    #[error("invalid image input: {0}")]
    ImageInput(#[source] crate::OpenAiChatCompletionValidationError),
    #[error("encrypted foreign reasoning cannot be replayed locally")]
    UnsupportedReasoningReplay,
    #[error("response input item type '{input_item_type}' is unsupported")]
    UnsupportedInputItem { input_item_type: String },
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
    #[error(
        "output token count is {actual_output_tokens}, outside the 1..={maximum_output_tokens} token range"
    )]
    OutputTokenCountOutOfRange {
        actual_output_tokens: u32,
        maximum_output_tokens: u32,
    },
    #[error("{parameter_name} is outside the supported range {minimum}..={maximum}")]
    SamplingParameterOutOfRange {
        parameter_name: &'static str,
        minimum: String,
        maximum: String,
    },
    #[error("request field '{field_name}' is unknown")]
    UnknownField { field_name: String },
    #[error(transparent)]
    StructuredOutput(#[from] OpenAiStructuredOutputValidationError),
}

fn validate_compatibility_fields(
    request: &OpenAiResponsesRequest,
) -> Result<(), OpenAiResponsesValidationError> {
    if request.store == Some(true) {
        return Err(OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "store=true",
        });
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
            return Err(OpenAiResponsesValidationError::UnsupportedOption { option_name });
        }
    }
    if request.parallel_tool_calls == Some(false) {
        return Err(OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "parallel_tool_calls=false",
        });
    }
    if request
        .top_logprobs
        .is_some_and(|top_logprobs| top_logprobs > 0)
    {
        return Err(OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "top_logprobs",
        });
    }
    if request
        .text
        .as_ref()
        .is_some_and(|text_configuration| text_configuration.get("verbosity").is_some())
    {
        return Err(OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "text.verbosity",
        });
    }
    if request.background == Some(true) {
        return Err(OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "background=true",
        });
    }
    if request
        .truncation
        .as_deref()
        .is_some_and(|truncation| truncation != "disabled")
    {
        return Err(OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "truncation",
        });
    }
    if request
        .service_tier
        .as_deref()
        .is_some_and(|service_tier| !matches!(service_tier, "auto" | "default"))
    {
        return Err(OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "service_tier",
        });
    }
    if request.metadata.len() > 16 {
        return Err(OpenAiResponsesValidationError::MetadataEntryCountExceeded);
    }
    if request
        .metadata
        .iter()
        .any(|(metadata_name, metadata_text)| metadata_name.len() > 64 || metadata_text.len() > 512)
    {
        return Err(OpenAiResponsesValidationError::MetadataTextTooLong);
    }
    Ok(())
}

fn validate_sampling_parameter(
    parameter_name: &'static str,
    parameter_value: Option<f32>,
    minimum: f32,
    maximum: f32,
) -> Result<(), OpenAiResponsesValidationError> {
    let Some(parameter_value) = parameter_value else {
        return Ok(());
    };
    if parameter_value.is_finite() && parameter_value >= minimum && parameter_value <= maximum {
        return Ok(());
    }
    Err(
        OpenAiResponsesValidationError::SamplingParameterOutOfRange {
            parameter_name,
            minimum: minimum.to_string(),
            maximum: maximum.to_string(),
        },
    )
}
