use serde::{Deserialize, Serialize};

use crate::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationSettings,
    ChatMessage, EmbeddingsFailureReason,
};

/// Protocol version the daemon and its local CLI clients negotiate on connect.
pub const DAEMON_PROTOCOL_VERSION: u32 = 1;

/// Application name the daemon reports during the handshake so a CLI client
/// can confirm the socket belongs to Astronomical and not to a stale file.
pub const DAEMON_APPLICATION_NAME: &str = "Astronomical";

/// Coarse daemon availability reported for status requests.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonWorkerStatus {
    /// The inference engine is still loading.
    Loading,
    /// The daemon can accept chat generation requests.
    Ready,
    /// The inference engine is absent or otherwise unavailable.
    Unavailable,
}

/// One request an ephemeral CLI process sends to the local daemon.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DaemonRequest {
    Handshake,
    /// Reports the daemon's current availability and resident model.
    Status,
    /// Streams one chat completion from the resident model. The daemon
    /// fills tools, tool choice, the Qwen thinking seed, and structured
    /// generation itself: the CLI surface never leases those capabilities.
    ChatGenerate {
        /// Exact worker-advertised model ID the request targets.
        model: String,
        /// Ordered conversation ending in the user's prompt.
        messages: Vec<ChatMessage>,
        /// Bounded sampling and output settings from the CLI.
        settings: ChatGenerationSettings,
    },
    /// Computes one embedding batch. A `None` model means "use whatever is
    /// resident"; the daemon swaps models itself when the requested model
    /// differs from the loaded one.
    EmbedGenerate {
        /// Exact worker-advertised model ID, or the resident model when absent.
        model: Option<String>,
        /// Texts to embed; the CLI sends exactly one.
        inputs: Vec<String>,
        /// Optional Matryoshka dimension truncation requested by the caller.
        dimensions: Option<u32>,
    },
}

/// One reply the daemon sends back to the local CLI process. `Eq` is
/// intentionally absent: embedding vectors are `f32`, which only supports
/// `PartialEq`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DaemonResponse {
    HandshakeAccepted {
        protocol_version: u32,
        application_name: String,
    },
    /// Answer to [`DaemonRequest::Status`].
    Status {
        worker_status: DaemonWorkerStatus,
        ready_model_id: Option<String>,
    },
    /// One visible answer fragment, streamed in generation order.
    ChatGenerationText { text: String },
    /// One reasoning-channel fragment.
    ChatGenerationReasoning { text: String },
    /// One model-requested function call.
    ChatGenerationToolCall {
        tool_call_index: u16,
        function_name: String,
        arguments_json: String,
    },
    /// Terminal frame for a finished generation.
    ChatGenerationCompleted {
        prompt_token_count: u32,
        generated_token_count: u16,
        reasoning_token_count: u16,
        cached_token_count: u32,
        reason: ChatGenerationCompletionReason,
    },
    /// Terminal frame for a failed generation.
    ChatGenerationFailed { reason: ChatGenerationFailureReason },
    /// Terminal refusal when the daemon declines the request before inference.
    GenerationRejected { reason: String },
    /// Terminal frame for a finished embedding batch.
    EmbeddingsCompleted {
        /// Worker-advertised model ID that produced the vectors.
        model: String,
        /// One vector per input, in input order.
        vectors: Vec<Vec<f32>>,
        /// Token count per input, in input order.
        input_token_counts: Vec<u32>,
    },
    /// Terminal frame for a failed embedding batch.
    EmbeddingsFailed { reason: EmbeddingsFailureReason },
}
