use serde::Serialize;
use thiserror::Error;

/// Validated input used to construct one OpenAI-compatible model description.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiModelParts {
    /// Model identifier advertised by the local server.
    pub model_id: String,
    /// Unix timestamp associated with this server's model advertisement.
    pub created: u64,
    /// Local owner label for the advertised model.
    pub owned_by: String,
    /// Total prompt plus generation position capacity of the model.
    pub context_window: u32,
    /// Maximum prompt tokens a client may send.
    pub max_input_tokens: u32,
    /// Maximum output tokens a client may request.
    pub max_output_tokens: u32,
    /// Modalities accepted in request input.
    pub input_modalities: Vec<String>,
    /// Modalities emitted in response output.
    pub output_modalities: Vec<String>,
    /// Whether the advertised generation endpoints support streaming.
    pub supports_streaming: bool,
    /// Whether the model emits reasoning separately from visible assistant text.
    pub supports_reasoning: bool,
    /// Public wire format used when reasoning is supported.
    pub reasoning_format: Option<String>,
    /// Whether the model emits validated function calls.
    pub supports_tool_calls: bool,
    /// Public wire format used when function calls are supported.
    pub tool_call_format: Option<String>,
    /// Generation endpoints supported by this model.
    pub supported_endpoints: Vec<String>,
}

impl OpenAiModelParts {
    /// Validates relationships that must hold between advertised model capabilities.
    pub fn validate(&self) -> Result<(), OpenAiModelValidationError> {
        if self.context_window == 0 {
            return Err(OpenAiModelValidationError::ContextWindowMustBePositive);
        }
        if self.max_input_tokens > self.context_window {
            return Err(
                OpenAiModelValidationError::InputTokenBudgetExceedsContextWindow {
                    max_input_tokens: self.max_input_tokens,
                    context_window: self.context_window,
                },
            );
        }
        if self.max_output_tokens > self.context_window {
            return Err(
                OpenAiModelValidationError::OutputTokenBudgetExceedsContextWindow {
                    max_output_tokens: self.max_output_tokens,
                    context_window: self.context_window,
                },
            );
        }
        if self
            .max_input_tokens
            .checked_add(self.max_output_tokens)
            .is_none_or(|total_token_budget| total_token_budget > self.context_window)
        {
            return Err(
                OpenAiModelValidationError::CombinedTokenBudgetsExceedContextWindow {
                    max_input_tokens: self.max_input_tokens,
                    max_output_tokens: self.max_output_tokens,
                    context_window: self.context_window,
                },
            );
        }
        if !self
            .input_modalities
            .iter()
            .any(|modality| modality == "text")
        {
            return Err(OpenAiModelValidationError::InputModalitiesMustContainText);
        }
        if !self
            .output_modalities
            .iter()
            .any(|modality| modality == "text")
        {
            return Err(OpenAiModelValidationError::OutputModalitiesMustContainText);
        }
        if self.supports_reasoning != self.reasoning_format.is_some() {
            return Err(OpenAiModelValidationError::ReasoningFormatMustMatchSupport);
        }
        if self.supports_tool_calls != self.tool_call_format.is_some() {
            return Err(OpenAiModelValidationError::ToolCallFormatMustMatchSupport);
        }
        if self.supported_endpoints.is_empty() {
            return Err(OpenAiModelValidationError::SupportedEndpointsMustNotBeEmpty);
        }
        Ok(())
    }
}

/// Rejection reason for internally inconsistent advertised model capabilities.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum OpenAiModelValidationError {
    /// A model must have at least one context position.
    #[error("context window must be positive")]
    ContextWindowMustBePositive,
    /// The advertised prompt budget cannot exceed the full context window.
    #[error("maximum input tokens {max_input_tokens} exceed context window {context_window}")]
    InputTokenBudgetExceedsContextWindow {
        max_input_tokens: u32,
        context_window: u32,
    },
    /// The advertised output budget cannot exceed the full context window.
    #[error("maximum output tokens {max_output_tokens} exceed context window {context_window}")]
    OutputTokenBudgetExceedsContextWindow {
        max_output_tokens: u32,
        context_window: u32,
    },
    /// The prompt and output budgets cannot together exceed the context window.
    #[error(
        "maximum input tokens {max_input_tokens} plus maximum output tokens {max_output_tokens} exceed context window {context_window}"
    )]
    CombinedTokenBudgetsExceedContextWindow {
        max_input_tokens: u32,
        max_output_tokens: u32,
        context_window: u32,
    },
    /// Input must always support text for the current API contract.
    #[error("input modalities must contain text")]
    InputModalitiesMustContainText,
    /// Output must always support text for the current API contract.
    #[error("output modalities must contain text")]
    OutputModalitiesMustContainText,
    /// A reasoning format must be present exactly when reasoning is supported.
    #[error("reasoning format must be present exactly when reasoning is supported")]
    ReasoningFormatMustMatchSupport,
    /// A tool-call format must be present exactly when tool calling is supported.
    #[error("tool-call format must be present exactly when tool calling is supported")]
    ToolCallFormatMustMatchSupport,
    /// At least one generation endpoint must be advertised.
    #[error("supported endpoints must not be empty")]
    SupportedEndpointsMustNotBeEmpty,
}

/// A standard OpenAI-compatible list of the exact models ready in the local worker.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiModelList {
    #[serde(rename = "object")]
    object_kind: &'static str,
    data: Vec<OpenAiModel>,
}

impl OpenAiModelList {
    /// Builds an empty response while no local worker can safely advertise a model.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            object_kind: "list",
            data: Vec::new(),
        }
    }

    /// Builds a one-model response from validated model capabilities.
    pub fn single_model(model_parts: OpenAiModelParts) -> Result<Self, OpenAiModelValidationError> {
        Ok(Self {
            object_kind: "list",
            data: vec![OpenAiModel::from_parts(model_parts)?],
        })
    }

    /// Builds a multi-model response listing all discovered models.
    #[must_use]
    pub fn from_models(models: Vec<OpenAiModel>) -> Self {
        Self {
            object_kind: "list",
            data: models,
        }
    }
}

/// One model visible through the standard OpenAI models endpoint.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiModel {
    id: String,
    #[serde(rename = "object")]
    object_kind: &'static str,
    created: u64,
    owned_by: String,
    /// Total prompt plus generation position capacity of the loaded model.
    #[serde(skip_serializing_if = "Option::is_none")]
    context_window: Option<u32>,
    /// Maximum prompt tokens a client may send.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_input_tokens: Option<u32>,
    /// Advertised per-request output-token ceiling.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    /// Input modalities accepted by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_modalities: Option<Vec<String>>,
    /// Output modalities emitted by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    output_modalities: Option<Vec<String>>,
    /// Whether the worker supports SSE streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    supports_streaming: Option<bool>,
    /// Whether the worker emits reasoning separately from assistant-visible text.
    #[serde(skip_serializing_if = "Option::is_none")]
    supports_reasoning: Option<bool>,
    /// Public wire format for separately emitted reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_format: Option<String>,
    /// Whether the worker emits complete validated function calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    supports_tool_calls: Option<bool>,
    /// Public wire format for model-generated function calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_format: Option<String>,
    /// Generation endpoints that support this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    supported_endpoints: Option<Vec<String>>,
}

impl OpenAiModel {
    /// Returns the exact model identifier advertised through the REST API.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Builds one model entry from validated advertised model capabilities.
    pub fn from_parts(model_parts: OpenAiModelParts) -> Result<Self, OpenAiModelValidationError> {
        model_parts.validate()?;
        Ok(Self {
            id: model_parts.model_id,
            object_kind: "model",
            created: model_parts.created,
            owned_by: model_parts.owned_by,
            context_window: Some(model_parts.context_window),
            max_input_tokens: Some(model_parts.max_input_tokens),
            max_output_tokens: Some(model_parts.max_output_tokens),
            input_modalities: Some(model_parts.input_modalities),
            output_modalities: Some(model_parts.output_modalities),
            supports_streaming: Some(model_parts.supports_streaming),
            supports_reasoning: Some(model_parts.supports_reasoning),
            reasoning_format: model_parts.reasoning_format,
            supports_tool_calls: Some(model_parts.supports_tool_calls),
            tool_call_format: model_parts.tool_call_format,
            supported_endpoints: Some(model_parts.supported_endpoints),
        })
    }
}

/// A standard OpenAI-compatible error response.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiErrorResponse {
    error: OpenAiError,
}

impl OpenAiErrorResponse {
    /// Builds an invalid-request response with optional public field and stable code context.
    pub fn invalid_request(
        message: impl Into<String>,
        parameter: Option<&str>,
        code: Option<&str>,
    ) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                error_type: "invalid_request_error",
                param: parameter.map(str::to_owned),
                code: code.map(str::to_owned),
            },
        }
    }

    /// Builds a service-unavailable response when no safe worker can serve the request.
    pub fn service_unavailable(message: impl Into<String>, code: Option<&str>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                error_type: "server_error",
                param: None,
                code: code.map(str::to_owned),
            },
        }
    }

    /// Builds a service-unavailable response for a rejected model load.
    pub fn model_load_failed(model_load_failure_reason: String) -> Self {
        Self::service_unavailable(
            format!("the requested model could not be loaded: {model_load_failure_reason}"),
            Some("model_load_failed"),
        )
    }

    /// Builds a capacity response when the one-worker scheduler cannot admit another request.
    pub fn capacity_unavailable(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                error_type: "server_error",
                param: None,
                code: Some("server_capacity".to_owned()),
            },
        }
    }
}

/// Error content nested inside an OpenAI-compatible error response.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiError {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
}
