use std::future::Future;

use crate::{InferenceEngineError, MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
use astronomical_ipc_protocol::{
    ExpertMemoryMode, RequestId, WorkerEvent, WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPromptProcessingPhase, WorkerPromptWorkReuse,
};

use super::EngineLoadResult;

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

    /// Clears persistent prompt-cache storage on the engine's owning context.
    ///
    /// `Ok(None)` means this engine has no open cache store, allowing the worker
    /// to fall back to its startup-configured global cache root.
    fn clear_persistent_prompt_cache(
        &mut self,
        _model_id: Option<String>,
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

    /// Clears persistent prompt-cache storage owned by this execution context.
    fn clear_persistent_prompt_cache(
        &mut self,
        _model_id: Option<String>,
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

/// Engine-side metadata reported when one generation request starts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineGenerationStart {
    cached_token_count: u32,
    restored_prompt_prefix_token_count: u32,
    expert_memory_mode: Option<ExpertMemoryMode>,
    prompt_processing_phase: Option<WorkerPromptProcessingPhase>,
    persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
}

impl EngineGenerationStart {
    /// Builds start metadata with the prompt tokens restored from persistent cache.
    #[must_use]
    pub const fn new(cached_token_count: u32) -> Self {
        Self {
            cached_token_count,
            restored_prompt_prefix_token_count: cached_token_count,
            expert_memory_mode: None,
            prompt_processing_phase: Some(WorkerPromptProcessingPhase::Target),
            persistent_prompt_cache_diagnostics: None,
        }
    }

    #[must_use]
    pub const fn with_expert_memory_mode(
        cached_token_count: u32,
        expert_memory_mode: ExpertMemoryMode,
    ) -> Self {
        Self {
            cached_token_count,
            restored_prompt_prefix_token_count: cached_token_count,
            expert_memory_mode: Some(expert_memory_mode),
            prompt_processing_phase: Some(WorkerPromptProcessingPhase::Target),
            persistent_prompt_cache_diagnostics: None,
        }
    }

    /// Records the complete logical prompt prefix whose processing state was restored.
    #[must_use]
    pub const fn with_restored_prompt_prefix_token_count(
        mut self,
        restored_prompt_prefix_token_count: u32,
    ) -> Self {
        self.restored_prompt_prefix_token_count = restored_prompt_prefix_token_count;
        self
    }

    /// Returns the number of prompt tokens restored from persistent cache.
    #[must_use]
    pub const fn cached_token_count(&self) -> u32 {
        self.cached_token_count
    }

    /// Returns the logical prompt prefix already represented by restored engine state.
    #[must_use]
    pub const fn restored_prompt_prefix_token_count(&self) -> u32 {
        self.restored_prompt_prefix_token_count
    }

    #[must_use]
    pub const fn expert_memory_mode(&self) -> Option<ExpertMemoryMode> {
        self.expert_memory_mode
    }

    /// Records the model that will process the next prompt phase.
    #[must_use]
    pub const fn with_prompt_processing_phase(
        mut self,
        prompt_processing_phase: Option<WorkerPromptProcessingPhase>,
    ) -> Self {
        self.prompt_processing_phase = prompt_processing_phase;
        self
    }

    #[must_use]
    pub const fn prompt_processing_phase(&self) -> Option<WorkerPromptProcessingPhase> {
        self.prompt_processing_phase
    }

    #[must_use]
    pub fn with_persistent_prompt_cache_diagnostics(
        mut self,
        persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
    ) -> Self {
        self.persistent_prompt_cache_diagnostics = persistent_prompt_cache_diagnostics;
        self
    }

    #[must_use]
    pub const fn persistent_prompt_cache_diagnostics(
        &self,
    ) -> Option<&WorkerPersistentPromptCacheRequestDiagnostics> {
        self.persistent_prompt_cache_diagnostics.as_ref()
    }
}

/// Final engine state available after a generation ends or is cancelled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertResidencyTelemetry {
    pub total_layer_count: u32,
    pub complete_layer_count: u32,
    pub complete_layer_payload_bytes: u64,
    pub partial_layer_count: u32,
    pub partial_layer_payload_bytes: u64,
}

/// Final engine state available after a generation ends or is cancelled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenerationFinalization {
    expert_memory_mode: Option<ExpertMemoryMode>,
    mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    expert_residency_telemetry: Option<ExpertResidencyTelemetry>,
}

impl GenerationFinalization {
    #[must_use]
    pub const fn new(
        expert_memory_mode: Option<ExpertMemoryMode>,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
        expert_residency_telemetry: Option<ExpertResidencyTelemetry>,
    ) -> Self {
        Self {
            expert_memory_mode,
            mlx_memory_telemetry,
            expert_residency_telemetry,
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
    pub const fn expert_residency_telemetry(self) -> Option<ExpertResidencyTelemetry> {
        self.expert_residency_telemetry
    }

    #[must_use]
    pub const fn has_reportable_state(self) -> bool {
        self.expert_memory_mode.is_some()
            || self.mlx_memory_telemetry.is_some()
            || self.expert_residency_telemetry.is_some()
    }
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
        /// Wall-clock duration of the first decode forward, emitted exactly once.
        first_decode_forward_elapsed_millis: Option<u64>,
        /// Present only after the engine has released this request. The worker
        /// reports the finalization but must not cancel the engine again.
        generation_finalization: Option<GenerationFinalization>,
    },
    /// One bounded native prefill chunk completed without producing a token yet.
    PrefillProgress {
        processed_token_count: u32,
        elapsed_millis: u64,
        /// Time spent evaluating the model, excluding allocator cleanup and telemetry.
        forward_prefill_chunk_elapsed_millis: u64,
        /// Selected `prefill_chunck_tokens` used for this completed prompt-processing chunk.
        completed_prefill_chunk_tokens: u32,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
        /// Current complete/partial retained ownership after this chunk.
        expert_residency_telemetry: Option<ExpertResidencyTelemetry>,
        /// Active MLX telemetry captured during request-scoped draft scoring.
        speculative_prefill_draft_memory_telemetry: Option<MlxMemoryTelemetry>,
        expert_memory_mode: Option<ExpertMemoryMode>,
        prompt_work_reuse: WorkerPromptWorkReuse,
        persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
    },
    /// A confirmed prompt phase is about to begin before its blocking model work.
    PromptProcessingPhaseStarted {
        prompt_processing_phase: WorkerPromptProcessingPhase,
        total_token_count: u32,
    },
    /// Prefill is complete and the engine is reconciling expert ownership before decode.
    GenerationPreparationStarted {
        total_layer_count: u32,
        complete_layer_count: u32,
        complete_layer_payload_bytes: u64,
        partial_layer_count: u32,
        partial_layer_payload_bytes: u64,
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
            Self::PromptProcessingPhaseStarted { .. }
            | Self::GenerationPreparationStarted { .. }
            | Self::EndOfSequence => {}
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
