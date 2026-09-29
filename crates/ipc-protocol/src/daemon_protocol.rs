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
    /// Lists the models discovered on this Mac with their capability flags.
    ModelsList,
    /// Lists the release download catalog with ready/download state per entry.
    Catalog,
    /// Starts (or resumes a matching paused) download for one catalog entry.
    /// Accepts the requestable model id or the full huggingface id.
    DownloadStart {
        model_id: String,
    },
    /// Reports the active library download job, if any.
    DownloadStatus,
    /// Persists a new default model id for one-shot CLI verbs.
    DefaultModelSet {
        model_id: String,
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
        /// Effective default model id: the persisted value, falling back to
        /// the built-in default when the config sets none.
        default_model_id: Option<String>,
    },
    /// Answer to [`DaemonRequest::ModelsList`].
    ModelsList { models: Vec<DaemonListedModel> },
    /// Answer to [`DaemonRequest::Catalog`].
    Catalog { entries: Vec<DaemonCatalogEntry> },
    /// Terminal frame for an admitted [`DaemonRequest::DownloadStart`].
    DownloadStarted { huggingface_id: String },
    /// Answer to [`DaemonRequest::DownloadStatus`].
    DownloadStatus { job: Option<DaemonDownloadJob> },
    /// Terminal frame for an admitted [`DaemonRequest::DefaultModelSet`].
    DefaultModelSet { default_model_id: String },
    /// Terminal refusal when the daemon declines a model-management request.
    RequestRejected { reason: String },
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

/// One model discovered on this Mac, any capability class.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonListedModel {
    /// Requestable model id (leaf of the huggingface id).
    pub model_id: String,
    /// Model family label, e.g. `qwen`.
    pub family: String,
    /// Effective context window in tokens after policy clamping.
    /// `None` for models that are not chat-capable (image, embeddings).
    pub context_window: Option<u32>,
    /// True when this model can produce embeddings for `embed`.
    pub supports_embeddings: bool,
    /// True when this model is currently resident in the worker.
    pub is_resident: bool,
    /// Artifact size in bytes; the CLI renders decimal SI gigabytes.
    pub size_bytes: u64,
}

/// One release download-catalog entry with its local readiness.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonCatalogEntry {
    pub huggingface_id: String,
    pub display_name: String,
    pub family: String,
    /// Approximate download size in bytes; the CLI renders decimal SI GB.
    pub approximate_size_bytes: u64,
    /// True when the entry is discovered or has a validated publication here.
    pub ready_on_this_mac: bool,
    /// Requestable model id once ready; `None` while not downloaded.
    pub requestable_model_id: Option<String>,
    /// Active download job state for this entry, if one runs.
    pub download_state: Option<String>,
    pub context_window: Option<u32>,
    pub supports_reasoning: bool,
    pub supports_vision: bool,
    pub supports_tool_calls: bool,
    pub supports_image_generation: bool,
    pub supports_embeddings: bool,
}

/// The active library download job, if any.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonDownloadJob {
    pub huggingface_id: String,
    /// Durable job state, e.g. `downloading`, `verifying`, `publishing`.
    pub state: String,
    pub bytes_completed: u64,
    pub bytes_total: u64,
    /// Public error code when the job failed.
    pub error: Option<String>,
}
