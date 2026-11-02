//! The Anthropic Messages response, streaming events, and error envelope the
//! local server returns from `POST /v1/messages`.

use serde::Serialize;
use serde_json::Value;

/// One assistant message, returned whole or carried by `message_start`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnthropicMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub object_type: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<AnthropicOutputContentBlock>,
    pub stop_reason: Option<AnthropicStopReason>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

impl AnthropicMessage {
    /// A message before any content or stop reason exists.
    #[must_use]
    pub fn empty(id: String, model: String) -> Self {
        Self {
            id,
            object_type: "message",
            role: "assistant",
            model,
            content: Vec::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        }
    }
}

/// One output content block.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicOutputContentBlock {
    /// Local reasoning is not signed. The empty signature keeps the block
    /// shape clients parse, and this server accepts it back unverified.
    Thinking {
        thinking: String,
        signature: String,
    },
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
}

/// Token usage. `input_tokens` excludes prompt-cache hits, which Anthropic
/// reports separately in `cache_read_input_tokens`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_creation_input_tokens: u32,
    pub cache_read_input_tokens: u32,
}

impl AnthropicUsage {
    /// Splits the worker's whole prompt count into uncached and cached input.
    #[must_use]
    pub fn from_worker_counts(
        prompt_token_count: u32,
        cached_token_count: u32,
        output_token_count: u32,
    ) -> Self {
        Self {
            input_tokens: prompt_token_count.saturating_sub(cached_token_count),
            output_tokens: output_token_count,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: cached_token_count,
        }
    }
}

/// One server-sent event of a streamed Messages response.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicMessageStreamEvent {
    MessageStart {
        message: AnthropicMessage,
    },
    ContentBlockStart {
        index: usize,
        content_block: AnthropicOutputContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: AnthropicContentBlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: AnthropicMessageDeltaBody,
        usage: AnthropicUsage,
    },
    MessageStop,
    Error {
        error: AnthropicError,
    },
}

impl AnthropicMessageStreamEvent {
    /// The SSE `event:` name, which Anthropic sets equal to the payload type.
    #[must_use]
    pub const fn event_type(&self) -> &'static str {
        match self {
            Self::MessageStart { .. } => "message_start",
            Self::ContentBlockStart { .. } => "content_block_start",
            Self::ContentBlockDelta { .. } => "content_block_delta",
            Self::ContentBlockStop { .. } => "content_block_stop",
            Self::MessageDelta { .. } => "message_delta",
            Self::MessageStop => "message_stop",
            Self::Error { .. } => "error",
        }
    }
}

/// An incremental change to the open content block.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockDelta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    InputJsonDelta { partial_json: String },
}

/// The top-level message fields that change once generation stops.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct AnthropicMessageDeltaBody {
    pub stop_reason: AnthropicStopReason,
    pub stop_sequence: Option<&'static str>,
}

/// The `{"type":"error","error":{...}}` envelope of a rejected request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub envelope_type: &'static str,
    pub error: AnthropicError,
}

/// One Anthropic error; `error_type` uses Anthropic's fixed error-type names.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub message: String,
}

impl AnthropicError {
    #[must_use]
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::with_type("invalid_request_error", message)
    }

    #[must_use]
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::with_type("not_found_error", message)
    }

    #[must_use]
    pub fn request_too_large(message: impl Into<String>) -> Self {
        Self::with_type("request_too_large", message)
    }

    #[must_use]
    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::with_type("rate_limit_error", message)
    }

    #[must_use]
    pub fn api_error(message: impl Into<String>) -> Self {
        Self::with_type("api_error", message)
    }

    #[must_use]
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::with_type("overloaded_error", message)
    }

    fn with_type(error_type: &'static str, message: impl Into<String>) -> Self {
        Self {
            error_type,
            message: message.into(),
        }
    }

    /// Wraps this error in the top-level response envelope.
    #[must_use]
    pub fn into_response_body(self) -> AnthropicErrorResponse {
        AnthropicErrorResponse {
            envelope_type: "error",
            error: self,
        }
    }
}
