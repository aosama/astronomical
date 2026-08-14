use serde::{Deserialize, Serialize};

use crate::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationFailureReason,
    ChatGenerationOutput, ChatModelCapabilities, WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};

/// Maximum serialized payload accepted inside one length-delimited worker frame.
/// Matches the HTTP chat transport ceiling so one validated local request stays one IPC message.
pub const MAX_IPC_FRAME_BYTES: usize = 32 * 1024 * 1024;

/// Supervisor-local correlation identifier for one generation request.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct RequestId(u64);

impl RequestId {
    /// Creates a request identifier from a supervisor-local monotonic value.
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

/// Per-request model-row work that was eligible for and restored from reusable prompt state.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerPromptWorkReuse {
    pub target_eligible_token_count: u64,
    pub target_restored_token_count: u64,
    pub drafter_eligible_token_count: u64,
    pub drafter_restored_token_count: u64,
}

/// Current sparse-expert weight residency exposed by the local worker.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertMemoryMode {
    /// Every target and optional MTP layer has complete sparse experts resident.
    Resident,
    /// Some complete layers or routed pages are retained while misses still page.
    Hybrid,
    /// No sparse expert payload is retained; every layer pages operation-locally.
    Paged,
}

/// Final concrete sparse-expert topology copied from the model owner.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerExpertResidencySnapshot {
    pub total_layer_count: u32,
    pub complete_layer_count: u32,
    pub complete_layer_payload_bytes: u64,
    pub partial_layer_count: u32,
    pub partial_layer_payload_bytes: u64,
}

/// Runtime execution state of native multi-token prediction (MTP).
///
/// Disabled: the user preference is false.
/// TargetOnly: preference is true and the selected model has no compatible MTP inventory.
/// Active: preference is true, the head is compatible, and native MTP decode
/// is available.
/// Unavailable: preference is true but MTP inventory or initialization failed.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MtpRuntimeState {
    #[default]
    Disabled,
    TargetOnly,
    Active,
    Unavailable,
}

/// Runtime execution state of optional draft-assisted speculative prefill.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativePrefillRuntimeState {
    /// The user preference is disabled.
    #[default]
    Disabled,
    /// A validated request-scoped draft model is available for scoring.
    Active,
    /// The preference is enabled but the draft model could not be used.
    Unavailable,
}

/// Model currently processing the active prompt phase.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPromptProcessingPhase {
    /// The target model is processing protected or selected prompt work.
    #[default]
    Target,
    /// The request-scoped SpecPrefill drafter is preparing importance scores.
    Drafter,
}

/// Lifecycle point at which the worker observed MLX allocator memory.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxMemorySnapshotSource {
    /// The model became resident in the worker.
    ModelLoaded,
    /// A prompt-processing chunk completed.
    Prefill,
    /// A complete active-memory observation captured while request-scoped draft scoring ran.
    SpeculativePrefillDraftScoring,
    /// One-token-ahead decode work was submitted to MLX.
    DecodeSubmitted,
    /// Request state and reclaimable allocator memory were released.
    Finalized,
    /// The ready worker was idle when the supervisor requested a refresh.
    IdlePoll,
    /// A live MLX memory-ceiling control operation completed.
    MemoryLimitAdjusted,
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
    /// Complete active MLX memory attributed to the request-scoped drafter phase.
    pub speculative_prefill_draft_memory_bytes: u64,
}

/// Why the worker's adaptive prefill optimizer requested one candidate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPrefillOptimizerDecisionReason {
    InitialExploration,
    StaleObservationProbe,
    CumulativeLatencyPlanning,
    Fallback,
    TerminalRemainder,
}

/// Recent measured evidence for one configured prefill candidate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerPrefillOptimizerCandidateEvidence {
    pub candidate_prefill_chunck_tokens: u32,
    pub observation_count: u32,
    pub average_actual_prefill_chunck_tokens: u32,
    pub average_elapsed_millis: u64,
    pub decisions_since_last_observation: Option<u64>,
}

/// Human-readable execution context that isolates optimizer evidence.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerPrefillOptimizerContext {
    pub prompt_position_tokens: u32,
    pub has_restored_prefix: bool,
    pub is_first_chunck_after_restore: bool,
    pub has_visual_embeddings: bool,
    pub is_mtp_active: bool,
    pub are_sparse_experts_paged: bool,
    pub is_prompt_cache_capture_eligible: bool,
    pub has_prior_capacity_reduction: bool,
}

/// One optimizer decision, its measured outcome, and the evidence available afterward.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerPrefillOptimizerInsight {
    pub requested_prefill_chunck_tokens: u32,
    pub actual_prefill_chunck_tokens: u32,
    pub elapsed_millis: u64,
    pub decision_reason: WorkerPrefillOptimizerDecisionReason,
    pub has_observed_prefill_capacity_constraint: bool,
    pub has_observations_for_every_candidate: bool,
    pub context: WorkerPrefillOptimizerContext,
    pub candidate_evidence: Vec<WorkerPrefillOptimizerCandidateEvidence>,
}

/// A command sent from the HTTP process to its one inference worker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerCommand {
    /// Supplies immutable startup configuration before any worker operation.
    InitializeWorker(WorkerStartupConfiguration),
    /// Starts one bounded structured-chat generation.
    Generate(ChatGenerationCommand),
    /// Stops the active generation with this request identifier.
    Cancel { request_id: RequestId },
    /// Swaps the loaded model to a different model directory.
    /// The worker unloads the current model, validates and loads the new one,
    /// then emits a ModelSwapped event with the new model_id and capabilities.
    SwapModel {
        /// Absolute path to the new model directory.
        model_directory: String,
        /// Per-request output-token ceiling for the new model.
        max_output_tokens: u32,
    },
    /// Requests one MLX memory observation from a ready idle worker.
    SampleMlxMemory,
    /// Replaces the worker's effective MLX process memory ceiling while idle.
    UpdateMlxMemoryLimit {
        /// Validated effective MLX ceiling in exact bytes.
        effective_mlx_memory_ceiling_bytes: u64,
    },
    /// Requests the worker delete the persistent prompt-cache footprint on SSD.
    ///
    /// When `model_id` is `None`, the entire global cache root is wiped. When
    /// `model_id` is `Some`, only the `<global_root>/<model_id>/` tree is
    /// removed. The supervisor sends this from `DELETE /v1/cache` and waits
    /// for `PromptCacheCleared` before responding to the HTTP caller.
    ClearPromptCache {
        /// Optional scoped model identity. `None` clears everything.
        model_id: Option<String>,
    },
}

/// An event emitted by the inference worker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerEvent {
    /// Confirms the feature settings applied by the currently running worker.
    ///
    /// A fresh idle worker emits this after its Idle event. A loaded model emits
    /// it again after target binding determines its actual SpecPrefill state.
    RuntimeFeatureConfigurationApplied {
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration,
    },
    /// Reports that the worker process is running without a model loaded.
    ///
    /// The supervisor must send `SwapModel` before forwarding the first
    /// generation request.
    Idle {
        /// Immutable machine-derived MLX ceiling in bytes.
        machine_mlx_memory_ceiling_bytes: u64,
        /// Worker-configured effective MLX graph-evaluation ceiling in bytes.
        effective_mlx_memory_ceiling_bytes: u64,
        /// Positive no-model MLX minimum in bytes.
        minimum_mlx_memory_ceiling_bytes: u64,
    },
    /// Delivers an explicitly requested idle or model-load MLX observation.
    MlxMemorySample {
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    },
    /// Reports one accepted live MLX memory-ceiling adjustment.
    MlxMemoryLimitChanged {
        effective_mlx_memory_ceiling_bytes: u64,
        minimum_mlx_memory_ceiling_bytes: u64,
        expert_memory_mode: ExpertMemoryMode,
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    },
    /// Reports that an MLX memory-ceiling adjustment was rejected without mutation.
    MlxMemoryLimitRejected {
        requested_mlx_memory_ceiling_bytes: u64,
        minimum_mlx_memory_ceiling_bytes: u64,
        machine_mlx_memory_ceiling_bytes: u64,
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
    /// Reports that the configured model finished loading.
    Ready {
        model_id: String,
        capabilities: ChatModelCapabilities,
        /// Actual MTP runtime state reported by the worker after model load.
        mtp_runtime_state: MtpRuntimeState,
        /// Present when MTP is unavailable despite the preference being enabled.
        mtp_unavailable_reason: Option<String>,
        /// Actual optional speculative-prefill state reported after model load.
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        /// Present when speculative prefill is enabled but the draft model is unavailable.
        speculative_prefill_unavailable_reason: Option<String>,
        /// Configured draft model identity when speculative prefill is enabled.
        speculative_prefill_draft_model_id: Option<String>,
        /// Validated request-scoped draft revision when speculative prefill is active.
        speculative_prefill_draft_model_revision: Option<String>,
    },
    /// Delivers one or more ordered model outputs in a single frame.
    Output {
        request_id: RequestId,
        sequence_number: u16,
        generated_token_count: u16,
        outputs: Vec<ChatGenerationOutput>,
        /// Present when this output follows a measured decode boundary.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    },
    /// Reports initial prompt-processing status or one completed prompt-processing chunk.
    PrefillProgress {
        request_id: RequestId,
        prompt_processing_phase: WorkerPromptProcessingPhase,
        processed_tokens: u32,
        total_tokens: u32,
        elapsed_millis: u64,
        /// Present with completed chunks; model evaluation time before allocator cleanup.
        forward_prefill_chunck_elapsed_millis: Option<u64>,
        /// Present only after the engine completes and measures a prefill chunk.
        completed_prefill_chunck_tokens: Option<u32>,
        /// Present after an optimized chunk; absent for initial and fixed-size progress.
        prefill_optimizer_insight: Option<WorkerPrefillOptimizerInsight>,
        /// Present with completed chunks; worker-owned MLX allocator observation.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
        /// Current complete/partial ownership after this prompt-processing boundary.
        expert_residency: Option<WorkerExpertResidencySnapshot>,
        /// Snapshot captured while the request-scoped SpecPrefill drafter was scoring.
        speculative_prefill_draft_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    },
    /// Reports the explicit barrier between final prompt processing and first decode.
    GenerationPreparationStarted {
        request_id: RequestId,
        total_layer_count: u32,
        complete_layer_count: u32,
        complete_layer_payload_bytes: u64,
        partial_layer_count: u32,
        partial_layer_payload_bytes: u64,
    },
    /// Reports generated-token progress that has not necessarily produced public output yet.
    GenerationProgress {
        request_id: RequestId,
        generated_token_count: u16,
        maximum_output_tokens: u16,
        elapsed_millis: u64,
        /// Present when generation advanced without public model output.
        mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    },
    /// Reports the measured first decode forward independently from preparation and output.
    FirstDecodeCompleted {
        request_id: RequestId,
        elapsed_millis: u64,
    },
    /// Reports model-row work avoided through exact or SpecPrefill-specific reusable state.
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
        capabilities: ChatModelCapabilities,
        /// Expert-memory mode selected before the model became ready.
        expert_memory_mode: Option<ExpertMemoryMode>,
        /// Safe idle lower bound for the newly loaded model.
        minimum_mlx_memory_ceiling_bytes: u64,
        /// Actual MTP runtime state reported by the worker after the swap.
        mtp_runtime_state: MtpRuntimeState,
        /// Present when MTP is unavailable despite the preference being enabled.
        mtp_unavailable_reason: Option<String>,
        /// Actual optional speculative-prefill state reported after the swap.
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        /// Present when speculative prefill is enabled but the draft model is unavailable.
        speculative_prefill_unavailable_reason: Option<String>,
        /// Configured draft model identity when speculative prefill is enabled.
        speculative_prefill_draft_model_id: Option<String>,
        /// Validated request-scoped draft revision when speculative prefill is active.
        speculative_prefill_draft_model_revision: Option<String>,
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
    /// generation completes, so the supervisor can expose cache health through
    /// `GET /v1/cache/stats` without reading the cache directory itself.
    PersistentPromptCacheStats {
        persistent_prompt_cache_hits: u64,
        persistent_prompt_cache_misses: u64,
        persistent_prompt_cache_tokens_saved: u64,
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
    /// Emitted by the worker after processing `ClearPromptCache`. The supervisor
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

impl WorkerEvent {
    /// Returns a bounded diagnostic summary without exposing model-generated payloads.
    #[must_use]
    pub fn diagnostic_summary(&self) -> String {
        match self {
            Self::RuntimeFeatureConfigurationApplied { .. } => {
                "runtime_feature_configuration_applied".to_owned()
            }
            Self::Idle { .. } => "idle".to_owned(),
            Self::MlxMemorySample { .. } => "mlx_memory_sample".to_owned(),
            Self::MlxMemoryLimitChanged { .. } => "mlx_memory_limit_changed".to_owned(),
            Self::MlxMemoryLimitRejected { .. } => "mlx_memory_limit_rejected".to_owned(),
            Self::ExpertMemoryModeChanged { .. } => "expert_memory_mode_changed".to_owned(),
            Self::GenerationFinalized { request_id, .. } => {
                format!("generation_finalized request_id={}", request_id.value())
            }
            Self::Ready { .. } => "ready".to_owned(),
            Self::Output { request_id, .. } => {
                format!("output request_id={}", request_id.value())
            }
            Self::PrefillProgress { request_id, .. } => {
                format!("prefill_progress request_id={}", request_id.value())
            }
            Self::GenerationPreparationStarted { request_id, .. } => {
                format!(
                    "generation_preparation_started request_id={}",
                    request_id.value()
                )
            }
            Self::GenerationProgress { request_id, .. } => {
                format!("generation_progress request_id={}", request_id.value())
            }
            Self::FirstDecodeCompleted { request_id, .. } => {
                format!("first_decode_completed request_id={}", request_id.value())
            }
            Self::PromptWorkReuse { request_id, .. } => {
                format!("prompt_work_reuse request_id={}", request_id.value())
            }
            Self::Completed { request_id, .. } => {
                format!("completed request_id={}", request_id.value())
            }
            Self::Failed { request_id, .. } => {
                format!("failed request_id={}", request_id.value())
            }
            Self::ModelSwapped { .. } => "model_swapped".to_owned(),
            Self::ModelSwapFailed { .. } => "model_swap_failed".to_owned(),
            Self::PersistentPromptCacheStats { .. } => "persistent_prompt_cache_stats".to_owned(),
            Self::PromptCacheCleared { model_id, .. } => {
                format!("prompt_cache_cleared model_id={:?}", model_id)
            }
        }
    }
}
