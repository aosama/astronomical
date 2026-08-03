use serde::Serialize;

/// One OpenAI-compatible Server-Sent Events chat completion chunk.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiChatCompletionChunk {
    id: String,
    #[serde(rename = "object")]
    object_kind: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiTokenUsage>,
}

impl OpenAiChatCompletionChunk {
    /// Creates the initial assistant-role chunk for one stream.
    pub fn assistant_role(id: impl Into<String>, created: u64, model: impl Into<String>) -> Self {
        Self::new(
            id,
            created,
            model,
            OpenAiChatCompletionDelta {
                role: Some("assistant"),
                content: None,
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
            None,
        )
    }

    /// Creates one generated text delta.
    pub fn text_delta(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self::new(
            id,
            created,
            model,
            OpenAiChatCompletionDelta {
                role: None,
                content: Some(text.into()),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
            None,
        )
    }

    /// Creates one generated reasoning delta.
    pub fn reasoning_delta(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        reasoning_content: impl Into<String>,
    ) -> Self {
        Self::new(
            id,
            created,
            model,
            OpenAiChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: Some(reasoning_content.into()),
                tool_calls: Vec::new(),
            },
            None,
        )
    }

    /// Creates one complete tool-call delta at its stable output index.
    pub fn tool_call_delta(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        tool_call_index: u16,
        tool_call_id: impl Into<String>,
        function_name: impl Into<String>,
        function_arguments: impl Into<String>,
    ) -> Self {
        Self::new(
            id,
            created,
            model,
            OpenAiChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: vec![OpenAiToolCallDelta {
                    index: tool_call_index,
                    id: tool_call_id.into(),
                    tool_type: "function",
                    function: OpenAiToolCallFunctionDelta {
                        name: function_name.into(),
                        arguments: function_arguments.into(),
                    },
                }],
            },
            None,
        )
    }

    /// Creates the terminal chunk with the public completion reason.
    pub fn finished(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        finish_reason: OpenAiFinishReason,
    ) -> Self {
        Self::new(
            id,
            created,
            model,
            OpenAiChatCompletionDelta::empty(),
            Some(finish_reason),
        )
    }

    /// Adds token usage to a terminal chunk when the client requested it.
    #[must_use]
    pub fn with_usage(mut self, usage: OpenAiTokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    fn new(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        delta: OpenAiChatCompletionDelta,
        finish_reason: Option<OpenAiFinishReason>,
    ) -> Self {
        Self {
            id: id.into(),
            object_kind: "chat.completion.chunk",
            created,
            model: model.into(),
            choices: vec![OpenAiChatCompletionChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage: None,
        }
    }
}

/// One complete non-streaming OpenAI-compatible chat completion response.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiChatCompletionResponse {
    id: String,
    #[serde(rename = "object")]
    object_kind: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChatCompletionChoice>,
    usage: OpenAiTokenUsage,
}

impl OpenAiChatCompletionResponse {
    /// Creates the single-choice response returned by the initial local endpoint.
    pub fn new(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        message: OpenAiAssistantMessage,
        finish_reason: OpenAiFinishReason,
        usage: OpenAiTokenUsage,
    ) -> Self {
        Self {
            id: id.into(),
            object_kind: "chat.completion",
            created,
            model: model.into(),
            choices: vec![OpenAiChatCompletionChoice {
                index: 0,
                message,
                finish_reason,
            }],
            usage,
        }
    }
}

/// The one complete choice in an initial non-streaming response.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiChatCompletionChoice {
    index: u8,
    message: OpenAiAssistantMessage,
    finish_reason: OpenAiFinishReason,
}

/// Complete assistant output assembled from ordered worker events.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiAssistantMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAiResponseToolCall>,
}

impl OpenAiAssistantMessage {
    /// Creates one assistant message from bounded response parts.
    pub fn new(
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<OpenAiResponseToolCall>,
    ) -> Self {
        Self {
            role: "assistant",
            content,
            reasoning_content,
            tool_calls,
        }
    }
}

/// One complete function call in a non-streaming assistant message.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiResponseToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: OpenAiToolCallFunctionDelta,
}

impl OpenAiResponseToolCall {
    /// Creates one complete OpenAI function call.
    pub fn function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            tool_type: "function",
            function: OpenAiToolCallFunctionDelta {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

/// The one supported choice in the initial single-request stream.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiChatCompletionChunkChoice {
    index: u8,
    delta: OpenAiChatCompletionDelta,
    finish_reason: Option<OpenAiFinishReason>,
}

/// The incremental assistant output carried by one chunk.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAiToolCallDelta>,
}

impl OpenAiChatCompletionDelta {
    fn empty() -> Self {
        Self {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: Vec::new(),
        }
    }
}

/// One complete function call surfaced in an OpenAI stream chunk.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiToolCallDelta {
    index: u16,
    id: String,
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: OpenAiToolCallFunctionDelta,
}

/// The function data attached to an OpenAI tool-call delta.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiToolCallFunctionDelta {
    name: String,
    arguments: String,
}

/// A terminal reason recognized by OpenAI-compatible chat clients.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiFinishReason {
    /// The model reached a normal end-of-sequence marker.
    Stop,
    /// The configured output-token cap was reached.
    Length,
    /// The model emitted one or more complete function calls.
    ToolCalls,
}

/// Token accounting returned in complete responses and optionally on terminal stream chunks.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct OpenAiTokenUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<OpenAiPromptTokenDetails>,
}

/// Breakdown of prompt token costs, following the OpenAI convention.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct OpenAiPromptTokenDetails {
    cached_tokens: u32,
}

impl OpenAiTokenUsage {
    /// Builds checked token accounting from worker-observed counts.
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Option<Self> {
        prompt_tokens
            .checked_add(completion_tokens)
            .map(|total_tokens| Self {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                prompt_tokens_details: None,
            })
    }

    /// Attaches the number of prompt tokens served from the persistent cache.
    ///
    /// When zero, the `prompt_tokens_details` field is omitted entirely
    /// so existing clients see no change in the response shape.
    #[must_use]
    pub fn with_cached_tokens(mut self, cached_tokens: u32) -> Self {
        self.prompt_tokens_details = if cached_tokens > 0 {
            Some(OpenAiPromptTokenDetails { cached_tokens })
        } else {
            None
        };
        self
    }
}
