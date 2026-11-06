use serde::{Deserialize, Serialize};

use crate::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationSettings,
    ChatMessage, EmbeddingsFailureReason, GeneratedImage, ImageGenerationFailureReason,
    ImageGenerationResultMetadata,
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
    /// fills tools, tool choice, and the Qwen thinking seed itself: the
    /// CLI surface never leases those capabilities. Structured generation
    /// is the exception the caller owns: the CLI sends raw schema JSON
    /// here and the daemon validates it into the enforced worker
    /// constraint before the worker ever sees it.
    ChatGenerate {
        /// Exact worker-advertised model ID the request targets.
        model: String,
        /// Ordered conversation ending in the user's prompt.
        messages: Vec<ChatMessage>,
        /// Bounded sampling and output settings from the CLI.
        settings: ChatGenerationSettings,
        /// Raw JSON schema text the reply must satisfy, when the caller
        /// supplied `--schema`. The CLI forwards the file's text; the
        /// daemon parses, bounds, and rejects invalid schemas.
        schema_json: Option<String>,
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
    /// Generates one image from a text prompt. The daemon owns everything
    /// model-specific (the capability envelope, the diffusion step count,
    /// the guidance scale) so the CLI carries only what the user chose.
    ImageGenerate(ImageGenerateRequest),
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

/// What the user chose for one image: the daemon adds the model family's own
/// step count and guidance, so neither is a field here.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerateRequest {
    /// Exact worker-advertised image model ID; the daemon swaps the resident
    /// model itself when it differs.
    pub model: String,
    /// What to draw.
    pub prompt: String,
    pub width_pixels: u32,
    pub height_pixels: u32,
    /// Reproducibility seed; the daemon picks one when absent and reports the
    /// effective seed in the completion.
    pub seed: Option<u64>,
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
        /// Which model is loading; absent when nothing is.
        /// Lets a waiting caller tell a slow load from a stuck daemon.
        model_load_in_flight: Option<crate::ModelLoadInFlight>,
        /// Changes each time a chat or image request makes progress, this caller's or
        /// another's. Embeddings report no progress, so they do not move it.
        /// Lets a waiting caller tell a long prefill or a queue from a stuck worker.
        /// Only a change means anything; the value itself can restart.
        request_progress_count: u64,
        /// What the Neural Engine is doing for the loaded model; `None` before any
        /// model has reported.
        neural_engine_prefill: Option<crate::NeuralEnginePrefillStatus>,
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
    /// Terminal frame for a finished image. The PNG travels as base64 in the
    /// frame, well inside the 32 MB frame ceiling for any supported size.
    ImageGenerationCompleted {
        /// Worker-advertised model ID that drew the image.
        model: String,
        image: GeneratedImage,
        /// Effective seed, size, steps, guidance, and elapsed time.
        metadata: ImageGenerationResultMetadata,
    },
    /// Terminal frame for an image request the worker failed after admission.
    ImageGenerationFailed {
        reason: ImageGenerationFailureReason,
    },
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
    /// True when this model can draw an image from text for `image`.
    pub supports_image_generation: bool,
    /// True when this model is currently resident in the worker.
    pub is_resident: bool,
    /// Artifact size in bytes; the CLI renders gigabytes of 2³⁰ bytes.
    pub size_bytes: u64,
}

/// One release download-catalog entry with its local readiness.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonCatalogEntry {
    pub huggingface_id: String,
    pub display_name: String,
    pub family: String,
    /// Approximate download size in bytes; the CLI renders GB of 2³⁰ bytes.
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
