use serde::{Deserialize, Serialize};

use crate::RequestId;

/// One bounded structured chat-generation command for the local inference worker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatGenerationCommand {
    /// The supervisor-generated request correlation identifier.
    pub request_id: RequestId,
    /// The exact worker-advertised model ID the request targets.
    pub model: String,
    /// Ordered system, user, assistant, and tool conversation history.
    pub messages: Vec<ChatMessage>,
    /// Functions that the model may request but never execute itself.
    pub tools: Vec<ChatToolDefinition>,
    /// The caller's function-selection mode.
    pub tool_choice: ChatToolChoice,
    /// Bounded sampling and output settings.
    pub settings: ChatGenerationSettings,
}

/// One chat message crossing the supervisor-to-worker trust boundary.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "role", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChatMessage {
    /// An initial model instruction.
    System {
        /// Text content that was separately validated by the supervisor.
        content: String,
    },
    /// A user message, possibly carrying decoded image inputs.
    User {
        /// Text content that was separately validated by the supervisor.
        content: String,
        /// Decoded image file payloads in document order. Empty for text-only users.
        #[serde(default)]
        images: Vec<ChatImageInput>,
    },
    /// A previous assistant message and optional tool calls.
    Assistant {
        /// Final response text, when the assistant emitted one.
        content: Option<String>,
        /// Model reasoning retained as separate client-visible content.
        reasoning_content: Option<String>,
        /// Function calls requested by the prior assistant turn.
        tool_calls: Vec<ChatAssistantToolCall>,
    },
    /// A result from a prior tool call.
    Tool {
        /// Correlates this result to the assistant tool call that requested it.
        tool_call_id: String,
        /// Text returned by the externally owned tool.
        content: String,
    },
}

/// One decoded image carried in a user chat message across the IPC boundary.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatImageInput {
    /// The MIME type parsed from the source data URI, e.g. `image/png`.
    pub mime_type: String,
    /// The raw decoded image file bytes (PNG/JPEG/WebP payload before pixel decoding).
    #[serde(with = "base64_image_file_bytes")]
    pub decoded_bytes: Vec<u8>,
}

mod base64_image_file_bytes {
    use base64::prelude::{BASE64_STANDARD, Engine as _};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<SerializerType>(
        decoded_image_file_bytes: &[u8],
        serializer: SerializerType,
    ) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.serialize_str(&BASE64_STANDARD.encode(decoded_image_file_bytes))
    }

    pub fn deserialize<'de, DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Vec<u8>, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let encoded_image_file_bytes = String::deserialize(deserializer)?;
        BASE64_STANDARD
            .decode(encoded_image_file_bytes)
            .map_err(serde::de::Error::custom)
    }
}

/// One assistant function call retained in conversation history.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatAssistantToolCall {
    /// Client-visible call ID used to correlate the later tool response.
    pub id: String,
    /// The requested function and JSON argument document.
    pub function: ChatAssistantToolFunction,
}

/// One named JSON function invocation retained in assistant history.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatAssistantToolFunction {
    /// The function name.
    pub name: String,
    /// Canonical JSON arguments serialized by the supervisor.
    pub arguments_json: String,
}

/// One callable JSON-schema function supplied to the model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatToolDefinition {
    /// The declared function name.
    pub name: String,
    /// Optional caller-provided explanation.
    pub description: Option<String>,
    /// Canonical JSON Schema serialized by the supervisor after bounded validation.
    pub parameters_json: String,
}

/// The caller's tool-selection policy after public validation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChatToolChoice {
    /// The model decides whether to call a function.
    Auto,
    /// The model must not call a function.
    None,
    /// The model must call one declared function.
    Required,
    /// The model must call this declared function.
    Function {
        /// The selected declared function name.
        name: String,
    },
}

/// Bounded generation settings that influence model execution.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatGenerationSettings {
    /// The maximum number of generated tokens for this request.
    pub max_output_tokens: u16,
    /// Optional sampling temperature in thousandths.
    pub temperature_thousandths: Option<u16>,
    /// Optional nucleus-sampling threshold in thousandths.
    pub top_p_thousandths: Option<u16>,
    /// Optional deterministic sampler seed.
    pub seed: Option<u64>,
    /// Maximum tokens the model may spend inside the thinking block before
    /// being forced to close it. `None` means no budget — the model thinks
    /// freely up to `max_output_tokens`.
    #[serde(default)]
    pub thinking_budget: Option<u16>,
}

/// Structured-chat capabilities reported by one ready worker model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatModelCapabilities {
    /// Whether the worker emits reasoning separately from assistant-visible text.
    pub supports_reasoning: bool,
    /// Whether the worker emits complete validated function calls.
    pub supports_tool_calls: bool,
    /// Whether the model supports image input (vision).
    pub has_vision: bool,
    /// Maximum prompt tokens a client may send. Reported as the context window
    /// minus the reserved output budget so clients size prompts to leave room
    /// for generation. The engine enforces the real shared prompt+generation
    /// bound at admission time.
    pub max_input_tokens: u32,
    /// Advertised per-request output-token ceiling. Well under the u16 protocol
    /// request cap and the context window.
    pub max_output_tokens: u32,
    /// Total prompt plus generation position capacity of the loaded model
    /// (Qwen3.5-MoE `max_position_embeddings`).
    pub context_window: u32,
}

/// One ordered output emitted while structured chat generation is active.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChatGenerationOutput {
    /// Assistant-visible text with model control syntax removed.
    Text { text: String },
    /// Model reasoning kept separate from assistant-visible text.
    Reasoning { text: String },
    /// One complete validated function call.
    ToolCall {
        tool_call_index: u16,
        function_name: String,
        arguments_json: String,
    },
}

/// A bounded request-scoped failure that leaves the worker process responsive.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatGenerationFailureReason {
    /// The worker independently rejected malformed structured chat input.
    InvalidRequest { reason: String },
    /// A fatal model-execution failure reported before the worker exits.
    /// The reason is bounded and safe for the local API; native details stay in logs.
    FatalExecution { reason: String },
    /// Prompt plus requested output exceeds the model-native context window.
    ContextLengthExceeded {
        actual_total_context_tokens: u32,
        maximum_context_tokens: u32,
    },
    /// A different generation request already owns the worker's bounded capacity.
    EngineBusy,
    /// Generated tokens could not be decoded or parsed into the declared output contract.
    MalformedModelOutput,
}

impl ChatGenerationFailureReason {
    /// Preserves the worker's human-readable validation explanation for diagnostics and clients.
    #[must_use]
    pub fn invalid_request(reason: impl Into<String>) -> Self {
        Self::InvalidRequest {
            reason: reason.into(),
        }
    }
}

/// A bounded reason why structured chat generation stopped.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatGenerationCompletionReason {
    /// The model emitted its configured end-of-sequence token.
    EndOfSequence,
    /// The request produced exactly its allowed output-token count.
    MaximumOutputTokens,
    /// The model emitted at least one complete validated function call.
    ToolCalls,
    /// The supervisor cancelled the active request.
    Cancelled,
}
