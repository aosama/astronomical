use std::future::Future;

use crate::{InferenceEngineError, MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
use astronomical_ipc_protocol::{
    ExpertMemoryMode, MtpRuntimeState, RequestId, SpeculativePrefillRuntimeState, WorkerEvent,
    WorkerPromptWorkReuse,
};

/// Asynchronous inference-engine contract that keeps runtime-affine work off Tokio threads.
pub trait InferenceEngine {
    /// Architecture-specific prepared input accepted by this engine.
    type Request: PreparedInferenceRequest + Send;

    /// Loads engine resources before the worker reports readiness.
    fn load(
        &mut self,
    ) -> impl Future<Output = Result<EngineLoadResult, InferenceEngineError>> + Send;

    /// Creates one engine-side request after prompt preprocessing and capacity reservation.
    fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> impl Future<Output = Result<EngineGenerationStart, InferenceEngineError>> + Send;

    /// Advances one bounded prefill or generated-token boundary.
    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> impl Future<Output = Result<GeneratedToken, InferenceEngineError>> + Send;

    /// Adds tokenized model-visible feedback to the active request before decoding continues.
    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> impl Future<Output = Result<(), InferenceEngineError>> + Send;

    /// Cancels and releases engine-side state for one active request.
    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> impl Future<Output = Result<GenerationFinalization, InferenceEngineError>> + Send;

    /// Collects persistent prompt-cache observability stats for IPC emission.
    ///
    /// Returns `Ok(None)` when the engine has no persistent prompt cache. The
    /// worker emits `WorkerEvent::PersistentPromptCacheStats` only when `Some`.
    /// The default implementation returns `Ok(None)` so engines without persistent
    /// prompt storage incur no cost.
    fn collect_persistent_prompt_cache_stats(
        &self,
    ) -> impl Future<Output = Result<Option<WorkerEvent>, InferenceEngineError>> + Send {
        async move { Ok(None) }
    }

    /// Collects the current idle MLX memory observation when the engine supports it.
    fn collect_mlx_memory_telemetry(
        &self,
    ) -> impl Future<Output = Result<Option<MlxMemoryTelemetry>, InferenceEngineError>> + Send {
        async move { Ok(None) }
    }

    /// Updates the live MLX memory ceiling on the engine's owning execution context.
    fn update_mlx_memory_limit(
        &mut self,
        _requested_mlx_memory_ceiling_bytes: u64,
    ) -> impl Future<Output = Result<MlxMemoryLimitAdjustment, InferenceEngineError>> + Send {
        async move {
            Err(InferenceEngineError::Fatal {
                reason: "this engine does not support live MLX memory limits".to_owned(),
            })
        }
    }
}

/// Model-specific input that has passed prompt preparation and is ready for inference.
pub trait PreparedInferenceRequest {
    /// Returns the token count used for protocol progress reporting.
    fn prompt_token_count(&self) -> usize;
}

/// Synchronous architecture-specific implementation executed only on the MLX owner thread.
pub trait MlxInferenceExecution: 'static {
    /// Architecture-specific prepared input accepted by this execution owner.
    type Request: PreparedInferenceRequest + Send + 'static;

    /// Loads model resources on the owning MLX thread.
    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError>;

    /// Starts one prepared request.
    fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError>;

    /// Advances one bounded prompt-processing or token-generation boundary.
    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError>;

    /// Injects model-visible feedback into the active request.
    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError>;

    /// Releases active request state.
    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError>;

    /// Reports optional persistent prompt-cache state.
    fn collect_persistent_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(None)
    }

    /// Reports optional live MLX memory telemetry.
    fn collect_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        Ok(None)
    }

    /// Updates the active MLX memory ceiling.
    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError>;
}

/// Result of loading engine resources, including optional runtime optimizers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineLoadResult {
    mtp_runtime_state: MtpRuntimeState,
    mtp_unavailable_reason: Option<String>,
    speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
    speculative_prefill_unavailable_reason: Option<String>,
    speculative_prefill_draft_model_id: Option<String>,
    speculative_prefill_draft_model_revision: Option<String>,
    minimum_mlx_memory_ceiling_bytes: u64,
}

impl EngineLoadResult {
    /// Creates a load result with the default MTP state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
            minimum_mlx_memory_ceiling_bytes: 1,
        }
    }

    /// Sets the MTP runtime state.
    #[must_use]
    pub fn with_mtp_runtime_state(mut self, mtp_runtime_state: MtpRuntimeState) -> Self {
        self.mtp_runtime_state = mtp_runtime_state;
        self
    }

    /// Sets the MTP unavailable reason when the runtime state is Unavailable.
    #[must_use]
    pub fn with_mtp_unavailable_reason(mut self, reason: String) -> Self {
        self.mtp_unavailable_reason = Some(reason);
        self
    }

    /// Sets optional draft-assisted speculative-prefill load metadata.
    #[must_use]
    pub fn with_speculative_prefill_runtime(
        mut self,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> Self {
        self.speculative_prefill_runtime_state = speculative_prefill_runtime_state;
        self.speculative_prefill_unavailable_reason = speculative_prefill_unavailable_reason;
        self.speculative_prefill_draft_model_id = speculative_prefill_draft_model_id;
        self.speculative_prefill_draft_model_revision = speculative_prefill_draft_model_revision;
        self
    }

    /// Sets the loaded model's safe idle MLX minimum in exact bytes.
    #[must_use]
    pub const fn with_minimum_mlx_memory_ceiling_bytes(
        mut self,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> Self {
        self.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
        self
    }

    /// Returns the MTP runtime state.
    #[must_use]
    pub const fn mtp_runtime_state(&self) -> MtpRuntimeState {
        self.mtp_runtime_state
    }

    /// Returns the MTP unavailable reason, if any.
    #[must_use]
    pub fn mtp_unavailable_reason(&self) -> Option<&str> {
        self.mtp_unavailable_reason.as_deref()
    }

    /// Returns the optional draft-assisted speculative-prefill runtime state.
    #[must_use]
    pub const fn speculative_prefill_runtime_state(&self) -> SpeculativePrefillRuntimeState {
        self.speculative_prefill_runtime_state
    }

    /// Returns the optional draft-assisted speculative-prefill load failure reason.
    #[must_use]
    pub fn speculative_prefill_unavailable_reason(&self) -> Option<&str> {
        self.speculative_prefill_unavailable_reason.as_deref()
    }

    /// Returns the configured draft model identity, when present.
    #[must_use]
    pub fn speculative_prefill_draft_model_id(&self) -> Option<&str> {
        self.speculative_prefill_draft_model_id.as_deref()
    }

    /// Returns the validated resident draft model revision, when active.
    #[must_use]
    pub fn speculative_prefill_draft_model_revision(&self) -> Option<&str> {
        self.speculative_prefill_draft_model_revision.as_deref()
    }

    /// Returns the exact safe idle MLX minimum for the loaded engine.
    #[must_use]
    pub const fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.minimum_mlx_memory_ceiling_bytes
    }
}

/// Engine-side metadata reported when one generation request starts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineGenerationStart {
    cached_token_count: u32,
    expert_memory_mode: Option<ExpertMemoryMode>,
}

impl EngineGenerationStart {
    /// Builds start metadata with the prompt tokens restored from persistent cache.
    #[must_use]
    pub const fn new(cached_token_count: u32) -> Self {
        Self {
            cached_token_count,
            expert_memory_mode: None,
        }
    }

    #[must_use]
    pub const fn with_expert_memory_mode(
        cached_token_count: u32,
        expert_memory_mode: ExpertMemoryMode,
    ) -> Self {
        Self {
            cached_token_count,
            expert_memory_mode: Some(expert_memory_mode),
        }
    }

    /// Returns the number of prompt tokens restored from persistent cache.
    #[must_use]
    pub const fn cached_token_count(&self) -> u32 {
        self.cached_token_count
    }

    #[must_use]
    pub const fn expert_memory_mode(&self) -> Option<ExpertMemoryMode> {
        self.expert_memory_mode
    }
}

/// Final engine state available after a generation ends or is cancelled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenerationFinalization {
    expert_memory_mode: Option<ExpertMemoryMode>,
    mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
}

impl GenerationFinalization {
    #[must_use]
    pub const fn new(
        expert_memory_mode: Option<ExpertMemoryMode>,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    ) -> Self {
        Self {
            expert_memory_mode,
            mlx_memory_telemetry,
        }
    }

    #[must_use]
    pub const fn expert_memory_mode(self) -> Option<ExpertMemoryMode> {
        self.expert_memory_mode
    }

    #[must_use]
    pub const fn mlx_memory_telemetry(self) -> Option<MlxMemoryTelemetry> {
        self.mlx_memory_telemetry
    }

    #[must_use]
    pub const fn has_reportable_state(self) -> bool {
        self.expert_memory_mode.is_some() || self.mlx_memory_telemetry.is_some()
    }
}

/// Evidence for one candidate in the latest optimizer decision context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillChunckOptimizerCandidateInsight {
    pub candidate_prefill_chunck_tokens: usize,
    pub observation_count: usize,
    pub average_actual_prefill_chunck_tokens: usize,
    pub average_elapsed_millis: u64,
    pub decisions_since_last_observation: Option<u64>,
}

/// Execution context that isolates adaptive prefill evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillChunckOptimizerContextInsight {
    pub prompt_position_tokens: usize,
    pub has_restored_prefix: bool,
    pub is_first_chunck_after_restore: bool,
    pub has_visual_embeddings: bool,
    pub is_mtp_active: bool,
    pub are_sparse_experts_paged: bool,
    pub is_prompt_cache_capture_eligible: bool,
    pub has_prior_capacity_reduction: bool,
}

/// Latest adaptive prefill request, measured outcome, and supporting evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillChunckOptimizerInsight {
    pub requested_prefill_chunck_tokens: usize,
    pub actual_prefill_chunck_tokens: usize,
    pub elapsed_millis: u64,
    pub decision_reason: crate::PrefillChunckSizeOptimizerDecisionReason,
    pub has_observed_prefill_capacity_constraint: bool,
    pub has_observations_for_every_candidate: bool,
    pub context: PrefillChunckOptimizerContextInsight,
    pub candidate_evidence: Vec<PrefillChunckOptimizerCandidateInsight>,
}

/// One bounded progress boundary from the engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GeneratedToken {
    /// A generated token ID that must be decoded by the active model processor.
    TokenId {
        token_id: u32,
        is_reasoning_token: bool,
        expert_memory_mode: Option<ExpertMemoryMode>,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
        /// Present only after the engine has released this request. The worker
        /// reports the finalization but must not cancel the engine again.
        generation_finalization: Option<GenerationFinalization>,
    },
    /// One bounded native prefill chunk completed without producing a token yet.
    PrefillProgress {
        processed_token_count: u32,
        elapsed_millis: u64,
        /// Time spent evaluating the model, excluding allocator cleanup and telemetry.
        forward_prefill_chunck_elapsed_millis: u64,
        /// Selected `prefill_chunck_tokens` used for this completed prompt-processing chunk.
        completed_prefill_chunck_tokens: u32,
        prefill_optimizer_insight: Option<PrefillChunckOptimizerInsight>,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
        expert_memory_mode: Option<ExpertMemoryMode>,
        prompt_work_reuse: WorkerPromptWorkReuse,
    },
    /// Engine-side end-of-sequence without an explicit token ID.
    EndOfSequence,
}

impl GeneratedToken {
    /// Replaces mode metadata after request finalization changes expert residency.
    #[must_use]
    pub fn with_expert_memory_mode(
        mut self,
        final_expert_memory_mode: Option<ExpertMemoryMode>,
    ) -> Self {
        match &mut self {
            Self::TokenId {
                expert_memory_mode, ..
            }
            | Self::PrefillProgress {
                expert_memory_mode, ..
            } => *expert_memory_mode = final_expert_memory_mode,
            Self::EndOfSequence => {}
        }
        self
    }

    /// Attaches final post-cleanup state to the terminal generated token.
    #[must_use]
    pub fn with_generation_finalization(
        mut self,
        generation_finalization: GenerationFinalization,
    ) -> Self {
        if let Self::TokenId {
            expert_memory_mode,
            generation_finalization: generated_token_finalization,
            ..
        } = &mut self
        {
            *expert_memory_mode = generation_finalization.expert_memory_mode();
            *generated_token_finalization = Some(generation_finalization);
        }
        self
    }
}
