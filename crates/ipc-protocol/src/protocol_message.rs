use serde::{Deserialize, Serialize};

use crate::expert_layer_seating::ExpertLayerSeatingCounts;
use crate::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationFailureReason,
    ChatGenerationOutput, EmbeddingsCommand, EmbeddingsFailureReason, GeneratedImage,
    ImageGenerationCommand, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, WorkerModelCapabilities, WorkerModelConfiguration,
    WorkerPersistentPromptCacheRequestDiagnostics, WorkerRuntimeFeatureConfiguration,
    WorkerStartupConfiguration,
};

/// Maximum serialized payload accepted inside one length-delimited worker frame.
///
/// Aligned with the portable HTTP request body budget in
/// `astronomical-rest-contract` so one validated local request always stays one
/// IPC message: the budget module in `rest-contract` is the single source of
/// every ceiling value, its sources, and its retrieval date.
pub const MAX_IPC_FRAME_BYTES: usize = astronomical_rest_contract::MAX_OPENAI_REQUEST_BODY_BYTES;

/// Daemon-local correlation identifier for one generation request.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct RequestId(u64);

impl RequestId {
    /// Creates a request identifier from a daemon-local monotonic value.
    #[must_use]
    pub const fn new(raw_request_id: u64) -> Self {
        Self(raw_request_id)
    }

    /// Returns the numeric correlation value used in diagnostics.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Per-request target-model work that was eligible for and restored from reusable prompt state.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerPromptWorkReuse {
    pub target_eligible_token_count: u64,
    pub target_restored_token_count: u64,
}

/// Current sparse-expert weight residency exposed by the local worker.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertMemoryMode {
    /// Every target layer has complete sparse experts resident.
    Resident,
    /// Some routed experts are retained while misses still page.
    Hybrid,
    /// No sparse expert payload is retained; every miss pages from storage.
    Paged,
}

/// Final concrete sparse-expert topology copied from the model owner.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerExpertResidencySnapshot {
    pub total_layer_count: u32,
    pub resident_expert_count: u32,
    pub resident_expert_payload_bytes: u64,
    /// Expert-library bytes if every sparse expert were resident. Zero means
    /// the worker could not measure the library, and Observatory must not
    /// invent a remainder from the model file size.
    pub complete_expert_payload_byte_count: u64,
    /// Experts in one sparse layer. The streaming engine uses one capacity
    /// for every layer; zero means the layer strip has nothing to draw.
    pub expert_capacity: u32,
    /// Retained counts for the layer map. Older workers omit the field;
    /// the zero seating means Observatory waits for the end-of-reply topology.
    #[serde(default)]
    pub layer_seating: ExpertLayerSeatingCounts,
}

/// One decoder layer's retained sparse-expert identities at a snapshot instant.
///
/// The identities come from the warm slot-table cache, so a layer with no
/// warm table simply does not appear in the snapshot; absence means "every
/// routed expert of this layer still streams from storage".
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerExpertLayerTopology {
    pub layer_index: u32,
    /// Expert ids currently retained in RAM by this layer, sorted ascending.
    pub retained_expert_ids: Vec<u32>,
    /// Retained payload bytes the warm table accounts for this layer.
    pub retained_payload_bytes: u64,
    /// True when the layer's table holds every expert of the layer.
    pub is_complete_layer: bool,
    /// Prefill-plus-decode demand the retained experts cover, the same
    /// coverage score the residency planner arbitrates with.
    pub covered_demand: u64,
}

/// Session-scoped expert-streaming measurements exposed to the daemon.
///
/// Counters accumulate from the worker's per-request performance-attribution
/// reports across every generation of the current engine load, so the values
/// describe the whole serving session rather than one request. The topology
/// is the warm-cache truth at the instant of the snapshot.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerExpertStreamingStats {
    pub model_id: Option<String>,
    pub expert_memory_mode: ExpertMemoryMode,
    pub expert_residency: WorkerExpertResidencySnapshot,
    /// Layers with a warm expert table right now.
    pub layer_topology: Vec<WorkerExpertLayerTopology>,
    /// Routed assignments served from retained RAM this session.
    pub retained_route_assignment_count: u64,
    /// Routed assignments read from storage this session.
    pub storage_route_assignment_count: u64,
    /// Decode layer-tokens whose routed experts were all warm. Classified once
    /// per layer per token, so these count layer-tokens, not tokens (issue
    /// #1715).
    pub hot_route_fully_covered_layer_token_count: u64,
    /// Decode layer-tokens with some warm and some cold routed experts.
    pub hot_route_partially_covered_layer_token_count: u64,
    /// Decode layer-tokens with no warm routed experts.
    pub hot_route_missed_layer_token_count: u64,
    /// Logical positional reads issued for expert payloads this session.
    pub expert_source_read_call_count: u64,
    pub expert_source_read_byte_count: u64,
    pub expert_source_read_elapsed_nanoseconds: u64,
    /// Positional-read time attributable to prefill page streams.
    pub prefill_source_read_elapsed_nanoseconds: u64,
    /// Positional-read time attributable to decode page streams; the only
    /// seconds the decode time-saved estimate may divide by decode misses.
    pub decode_source_read_elapsed_nanoseconds: u64,
    /// Physical disk bytes the process read since expert streaming began,
    /// or `None` when the platform cannot sample process I/O. Wider than
    /// expert paging: process-attributed, so other worker reads count too.
    pub expert_source_physical_disk_read_bytes: Option<u64>,
    /// Storage bytes a no-cache server would have read for the routes that
    /// seated complete layers served from RAM instead (issue #1715).
    pub avoided_complete_layer_source_bytes: u64,
    /// Expert payload bytes prefill had to stream from storage.
    pub mandatory_prefill_source_bytes: u64,
    /// Expert payload bytes decode had to stream from storage.
    pub mandatory_decode_source_bytes: u64,
    /// Session totals of expert routing and streaming behind this snapshot.
    /// Prompt and generated tokens are the finalized request's own counts:
    /// the daemon accumulates them across events for session totals.
    pub prompt_token_count: u64,
    pub generated_token_count: u64,
}

/// Model currently processing the active prompt phase.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPromptProcessingPhase {
    /// The target model is processing protected or selected prompt work.
    #[default]
    Target,
}

/// Lifecycle point at which the worker observed MLX allocator memory.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxMemorySnapshotSource {
    /// The model became resident in the worker.
    ModelLoaded,
    /// A prompt-processing chunk completed.
    Prefill,
    /// One-token-ahead decode work was submitted to MLX.
    DecodeSubmitted,
    /// Request state and reclaimable allocator memory were released.
    Finalized,
    /// The ready worker was idle when the daemon requested a refresh.
    IdlePoll,
    /// A live MLX memory-ceiling control operation completed.
    MemoryLimitAdjusted,
    /// One image render step completed while generation was still active.
    ImageGenerationStep,
}

/// One worker-owned MLX allocator observation reconciled into user-visible owners.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerMlxMemorySnapshot {
    pub source: MlxMemorySnapshotSource,
    pub active_memory_bytes: u64,
    pub allocator_cache_memory_bytes: u64,
    pub peak_memory_bytes: u64,
    pub expert_payload_bytes: u64,
    pub model_core_payload_bytes: u64,
    pub context_state_payload_bytes: u64,
    /// Reason-tagged split of the ceiling's unused headroom at this instant
    /// (issue #510), composed by the same budget owner the engine acts on.
    /// `None` from engines without a composed RAM budget.
    #[serde(default)]
    pub memory_ceiling_utilization: Option<WorkerMemoryCeilingUtilizationSnapshot>,
}

/// One utilization decomposition published beside its memory snapshot.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerMemoryCeilingUtilizationSnapshot {
    pub unused_headroom_bytes: u64,
    pub reserved_model_core_slack_bytes: u64,
    pub reserved_context_growth_bytes: u64,
    pub reserved_activation_and_workspace_bytes: u64,
    pub unseated_expert_entitlement_bytes: u64,
    pub unexplained_headroom_bytes: u64,
    pub owner_overrun_bytes: u64,
}

/// A command sent from the HTTP process to its one inference worker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerCommand {
    /// Supplies immutable startup configuration before any worker operation.
    InitializeWorker(WorkerStartupConfiguration),
    /// Starts one bounded structured-chat generation.
    Generate(ChatGenerationCommand),
    /// Starts one bounded text-to-image generation.
    GenerateImage(ImageGenerationCommand),
    /// Starts one bounded text-embedding request.
    GenerateEmbeddings(EmbeddingsCommand),
    /// Stops the active generation with this request identifier.
    Cancel { request_id: RequestId },
    /// Swaps the loaded model to a different model directory.
    /// The worker unloads the current model, validates and loads the new one,
    /// then emits a ModelSwapped event with the new model_id and capabilities.
    SwapModel {
        /// Absolute path to the new model directory.
        model_directory: String,
        /// Fully resolved policy for this selected canonical model.
        model_configuration: WorkerModelConfiguration,
    },
    /// Requests one MLX memory observation from a ready idle worker.
    SampleMlxMemory,
    /// Replaces the worker's effective MLX process memory ceiling while idle.
    UpdateMlxMemoryLimit {
        /// Validated effective MLX ceiling in exact bytes.
        effective_mlx_memory_ceiling_bytes: u64,
        /// Configuration generation that owns the accepted memory policy.
        configuration_generation: String,
    },
    /// Requests the worker delete the persistent prompt-cache footprint on SSD.
    ///
    /// When `model_id` is `None`, the entire global cache root is wiped. When
    /// `model_id` is `Some`, only the `<global_root>/<model_id>/` tree is
    /// removed. The daemon sends this from `DELETE /v1/cache` and waits
    /// for `PromptCacheCleared` before responding to the HTTP caller.
    ClearPromptCache {
        /// Optional scoped model identity. `None` clears everything.
        model_id: Option<String>,
    },
}

/// An event emitted by the inference worker.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerEvent {
    /// Confirms the feature settings applied by the currently running worker.
    ///
    /// A fresh idle worker emits this after its Idle event. A loaded model emits
    /// it again after applying the selected model's runtime configuration.
    RuntimeFeatureConfigurationApplied {
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration,
    },
    /// Reports that the worker process is running without a model loaded.
    ///
    /// The daemon must send `SwapModel` before forwarding the first
    /// generation request.
    Idle {
        /// Immutable machine-derived MLX ceiling in bytes.
        machine_mlx_memory_ceiling_bytes: u64,
        /// Worker-configured effective MLX graph-evaluation ceiling in bytes.
        effective_mlx_memory_ceiling_bytes: u64,
        /// Positive no-model MLX minimum in bytes.
        minimum_mlx_memory_ceiling_bytes: u64,
        /// GPU memory macOS lets this worker keep wired, when a wired claim
        /// was granted at startup. `None` means pageable serving with no
        /// wired pool. The effective ceiling never exceeds this budget, so
        /// status consumers can name the operating-system bound whenever it
        /// clamps the configured ceiling (issue #1282).
        wired_residency_budget_bytes: Option<u64>,
    },
    /// Delivers an explicitly requested idle or model-load MLX observation.
    ///
    /// The residency snapshot is captured at the same instant as the memory
    /// snapshot so the public status can never pair a fresh measurement with a
    /// stale residency claim (issue #337).
    MlxMemorySample {
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
        expert_residency: Option<WorkerExpertResidencySnapshot>,
    },
    /// Reports one accepted live MLX memory-ceiling adjustment.
    MlxMemoryLimitChanged {
        effective_mlx_memory_ceiling_bytes: u64,
        minimum_mlx_memory_ceiling_bytes: u64,
        expert_memory_mode: ExpertMemoryMode,
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
        /// Sparse-expert ownership after the adjustment, when the model has experts.
        expert_residency: Option<WorkerExpertResidencySnapshot>,
    },
    /// Reports that an MLX memory-ceiling adjustment was rejected without mutation.
    MlxMemoryLimitRejected {
        requested_mlx_memory_ceiling_bytes: u64,
        minimum_mlx_memory_ceiling_bytes: u64,
        machine_mlx_memory_ceiling_bytes: u64,
        reason: String,
    },
    /// Reports that the requested ceiling is valid but needs a new worker process.
    ///
    /// The loaded resident engine cannot reach the requested ceiling in place,
    /// and constructing its streaming replacement inside this process would
    /// briefly require both model runtimes, so the memory cannot be released
    /// deterministically. The worker keeps its loaded runtime and unchanged
    /// ceilings; the daemon owns a stop-first process restart under the
    /// persisted lower ceiling.
    MlxMemoryLimitRestartRequired {
        /// Requested effective ceiling in bytes that this process cannot adopt.
        requested_mlx_memory_ceiling_bytes: u64,
        /// Effective ceiling after serving-envelope clamping, as requested.
        effective_mlx_memory_ceiling_bytes: u64,
        /// Safe idle lower bound the loaded model currently reports.
        minimum_mlx_memory_ceiling_bytes: u64,
        /// Configuration generation that owns the requested memory policy.
        configuration_generation: String,
        /// Bounded explanation suitable for the worker protocol.
        reason: String,
    },
    /// Reports a change in sparse-expert residency without affecting model output.
    ExpertMemoryModeChanged {
        expert_memory_mode: ExpertMemoryMode,
    },
    /// Reports final engine residency and MLX memory after request cleanup.
    ///
    /// This is separate from prompt-progress telemetry because automatic expert
    /// retention can grow after the final prefill measurement.
    GenerationFinalized {
        request_id: RequestId,
        expert_memory_mode: Option<ExpertMemoryMode>,
        /// Present when post-cleanup MLX memory could be observed.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
        expert_residency: Option<WorkerExpertResidencySnapshot>,
    },
    /// Reports session-scoped expert-streaming statistics after each
    /// generation finalizes.
    ///
    /// Emitted by the worker whenever a streaming engine completes a
    /// generation, so the daemon can expose expert-cache benefits through
    /// `GET /v1/experts/stats` without touching worker-owned state itself.
    ExpertStreamingStats { stats: WorkerExpertStreamingStats },
    /// Reports image-generation phase and denoising-step progress.
    ImageGenerationProgress {
        request_id: RequestId,
        phase: ImageGenerationPhase,
        completed_steps: u16,
        total_steps: u16,
        elapsed_millis: u64,
        /// Present when the image engine could observe MLX memory at this step
        /// boundary, so the menu shows a measured footprint mid-render.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    },
    /// Delivers one completed encoded image and its reproducibility metadata.
    ImageGenerationCompleted {
        request_id: RequestId,
        generated_image: GeneratedImage,
        result_metadata: ImageGenerationResultMetadata,
    },
    /// Reports a request-scoped image failure that leaves the worker responsive.
    ImageGenerationFailed {
        request_id: RequestId,
        reason: ImageGenerationFailureReason,
    },
    /// Confirms that image request state has been released after any outcome.
    ImageGenerationFinalized {
        request_id: RequestId,
        elapsed_millis: u64,
        /// Present when the image engine observed MLX after releasing request arrays and cache.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    },
    /// Delivers one completed embedding vector per input, in request order.
    EmbeddingsCompleted {
        request_id: RequestId,
        embeddings: Vec<Vec<f32>>,
        input_token_counts: Vec<u32>,
        elapsed_millis: u64,
    },
    /// Reports a request-scoped embeddings failure that leaves the worker responsive.
    EmbeddingsFailed {
        request_id: RequestId,
        reason: EmbeddingsFailureReason,
    },
    /// Confirms that embeddings request state has been released after any outcome.
    EmbeddingsFinalized {
        request_id: RequestId,
        elapsed_millis: u64,
        /// Present when the embedding engine observed MLX after releasing request arrays and cache.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    },
    /// Reports that the configured model finished loading.
    Ready {
        model_id: String,
        capabilities: WorkerModelCapabilities,
    },
    /// Delivers one or more ordered model outputs in a single frame.
    Output {
        request_id: RequestId,
        sequence_number: u16,
        generated_token_count: u16,
        outputs: Vec<ChatGenerationOutput>,
        /// Present when this output follows a measured decode boundary.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
        /// Sparse-expert ownership captured with the same decode observation.
        expert_residency: Option<WorkerExpertResidencySnapshot>,
    },
    /// Reports initial prompt-processing status or one completed prompt-processing chunk.
    PrefillProgress {
        request_id: RequestId,
        prompt_processing_phase: WorkerPromptProcessingPhase,
        processed_tokens: u32,
        total_tokens: u32,
        elapsed_millis: u64,
        /// Present with completed chunks; model evaluation time before allocator cleanup.
        forward_prefill_chunk_elapsed_millis: Option<u64>,
        /// Present only after the engine completes and measures a prefill chunk.
        completed_prefill_chunk_tokens: Option<u32>,
        /// Present with completed chunks; worker-owned MLX allocator observation.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
        /// Current retained-expert ownership after this prompt-processing boundary.
        expert_residency: Option<WorkerExpertResidencySnapshot>,
    },
    /// Reports the explicit barrier between final prompt processing and first decode.
    GenerationPreparationStarted {
        request_id: RequestId,
        /// Full residency at the preparation barrier, including the expert
        /// library size. Flattening a subset here dropped those fields from
        /// the live status bar at the moment generation began.
        expert_residency: WorkerExpertResidencySnapshot,
        /// Memory observation captured at the same owner-thread instant as the residency.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    },
    /// Reports generated-token progress that has not necessarily produced public output yet.
    GenerationProgress {
        request_id: RequestId,
        generated_token_count: u16,
        maximum_output_tokens: u16,
        elapsed_millis: u64,
        /// Present when generation advanced without public model output.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
        /// Sparse-expert ownership captured with the same decode observation.
        expert_residency: Option<WorkerExpertResidencySnapshot>,
    },
    /// Reports the measured first decode forward independently from preparation and output.
    FirstDecodeCompleted {
        request_id: RequestId,
        elapsed_millis: u64,
    },
    /// Reports model-row work avoided through exact reusable prompt state.
    PromptWorkReuse {
        request_id: RequestId,
        prompt_work_reuse: WorkerPromptWorkReuse,
    },
    /// Reports normal completion, including cancellation.
    Completed {
        request_id: RequestId,
        prompt_token_count: u32,
        generated_token_count: u16,
        reasoning_token_count: u16,
        cached_token_count: u32,
        persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
        reason: ChatGenerationCompletionReason,
    },
    /// Reports a request-scoped failure that leaves the process responsive.
    Failed {
        request_id: RequestId,
        reason: ChatGenerationFailureReason,
    },
    /// Reports that a model swap completed successfully and the new model is loaded.
    /// Emitted after processing a SwapModel command, replacing the initial Ready event.
    ModelSwapped {
        model_id: String,
        capabilities: WorkerModelCapabilities,
        /// Expert-memory mode selected before the model became ready.
        expert_memory_mode: Option<ExpertMemoryMode>,
        /// Safe idle lower bound for the newly loaded model.
        minimum_mlx_memory_ceiling_bytes: u64,
    },
    /// Reports that a model swap failed while the worker process remained responsive.
    ModelSwapFailed {
        /// Whether the previously loaded model remains available after the failure.
        loaded_model_remains_ready: bool,
        /// Actionable model-load failure reason safe to return to the local API caller.
        model_load_failure_reason: String,
    },
    /// Reports cumulative persistent prompt-cache observability counters and disk footprint.
    ///
    /// Emitted by the worker after the model finishes loading and after each
    /// generation completes, so the daemon can expose cache health through
    /// `GET /v1/cache/stats` without reading the cache directory itself.
    PersistentPromptCacheStats {
        persistent_prompt_cache_hits: u64,
        persistent_prompt_cache_misses: u64,
        persistent_prompt_cache_tokens_saved: u64,
        persistent_prompt_cache_partial_tail_hits: u64,
        persistent_prompt_cache_block_token_count: u64,
        persistent_prompt_cache_sequence_state_block_count: u64,
        persistent_prompt_cache_boundary_state_snapshot_count: u64,
        persistent_prompt_cache_visual_embedding_count: u64,
        persistent_prompt_cache_total_size_bytes: u64,
        persistent_prompt_cache_visual_embedding_total_size_bytes: u64,
        persistent_prompt_cache_maximum_size_bytes: u64,
        persistent_prompt_cache_visual_embedding_hits: u64,
        persistent_prompt_cache_visual_embedding_misses: u64,
        persistent_prompt_cache_visual_embedding_rows_loaded: u64,
    },
    /// Confirms a persistent prompt-cache deletion completed.
    ///
    /// Emitted by the worker after processing `ClearPromptCache`. The daemon
    /// waits for this event before responding to the HTTP caller with the
    /// deletion outcome.
    PromptCacheCleared {
        /// The model identity scoped by the request. `None` means global clear.
        model_id: Option<String>,
        /// Number of prompt-cache block directories removed.
        blocks_removed: u64,
        /// Total bytes removed across all cache file types.
        bytes_freed: u64,
    },
}
