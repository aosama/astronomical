use std::future::Future;

use crate::InferenceEngineError;
use astronomical_ipc_protocol::{
    ExpertLayerSeatingCounts, ExpertMemoryMode, RequestId, WorkerEvent,
    WorkerPersistentPromptCacheRequestDiagnostics, WorkerPromptProcessingPhase,
    WorkerPromptWorkReuse,
};

use super::EngineLoadResult;
use crate::memory::telemetry::MlxMemoryLimitAdjustment;
use crate::memory::telemetry::MlxMemoryTelemetry;

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

    /// Clones only requests whose family can safely replay them through streaming.
    ///
    /// Engines that do not support an invisible retry leave the default unchanged.
    fn clone_for_streaming_retry(&self) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    /// Adds a completed resident-to-streaming replacement interval to the replay request.
    ///
    /// Request types without request-scoped performance attribution may keep the default.
    fn record_streaming_retry_interval(
        &mut self,
        _started_at: std::time::Instant,
        _ended_at: std::time::Instant,
    ) {
    }
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
    pub resident_expert_count: u32,
    pub resident_expert_payload_bytes: u64,
    /// Expert-library bytes if every sparse expert were resident. Zero means
    /// unmeasured; Observatory hides the SSD remainder in that case.
    pub complete_expert_payload_byte_count: u64,
    /// Experts in one sparse layer. Uniform across the loaded model's layers.
    pub expert_capacity: u32,
    /// Per-layer retained counts. Zero seated layers means this sample has
    /// no map yet, so Observatory keeps the end-of-reply topology.
    pub layer_seating: ExpertLayerSeatingCounts,
}

/// One decoder layer's retained expert identities captured for observatory.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExpertLayerTopology {
    pub layer_index: u32,
    pub retained_expert_ids: Vec<u32>,
    pub retained_payload_bytes: u64,
    pub is_complete_layer: bool,
    pub covered_demand: u64,
}

/// Session-scoped expert-streaming measurements captured at finalization.
///
/// Counters accumulate across every generation of the current engine load,
/// recorded always-on at the same central sites the diagnostics attribution
/// reads, so observatory value panels never depend on diagnostics being
/// enabled.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertStreamingStats {
    pub model_id: Option<String>,
    pub expert_memory_mode: ExpertMemoryMode,
    pub total_layer_count: u32,
    pub resident_expert_count: u32,
    pub resident_expert_payload_bytes: u64,
    /// Expert-library bytes if every sparse expert were resident.
    pub complete_expert_payload_byte_count: u64,
    /// Experts in one sparse layer.
    pub expert_capacity: u32,
    pub layer_topology: Vec<ExpertLayerTopology>,
    pub retained_route_assignment_count: u64,
    pub storage_route_assignment_count: u64,
    pub hot_route_fully_covered_token_count: u64,
    pub hot_route_partially_covered_token_count: u64,
    pub hot_route_missed_token_count: u64,
    pub expert_source_read_call_count: u64,
    pub expert_source_read_byte_count: u64,
    pub expert_source_read_elapsed_nanoseconds: u64,
    pub avoided_complete_layer_source_bytes: u64,
    pub mandatory_prefill_source_bytes: u64,
    pub mandatory_decode_source_bytes: u64,
    pub prompt_token_count: u64,
    pub generated_token_count: u64,
}

impl Default for ExpertStreamingStats {
    /// Zero state for engines without expert streaming: nothing retained,
    /// every layer represented as absent from the topology.
    fn default() -> Self {
        Self {
            model_id: None,
            expert_memory_mode: ExpertMemoryMode::Paged,
            total_layer_count: 0,
            resident_expert_count: 0,
            resident_expert_payload_bytes: 0,
            complete_expert_payload_byte_count: 0,
            expert_capacity: 0,
            layer_topology: Vec::new(),
            retained_route_assignment_count: 0,
            storage_route_assignment_count: 0,
            hot_route_fully_covered_token_count: 0,
            hot_route_partially_covered_token_count: 0,
            hot_route_missed_token_count: 0,
            expert_source_read_call_count: 0,
            expert_source_read_byte_count: 0,
            expert_source_read_elapsed_nanoseconds: 0,
            avoided_complete_layer_source_bytes: 0,
            mandatory_prefill_source_bytes: 0,
            mandatory_decode_source_bytes: 0,
            prompt_token_count: 0,
            generated_token_count: 0,
        }
    }
}

impl ExpertStreamingStats {
    /// Returns whether any expert-streaming evidence exists for this engine.
    #[must_use]
    pub const fn has_streaming_evidence(&self) -> bool {
        self.total_layer_count > 0 || self.resident_expert_count > 0
    }
}

/// Final engine state available after a generation ends or is cancelled.
///
/// Not `Copy` since expert-streaming stats carry a heap topology; consumers
/// receive it once by value at the emission boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GenerationFinalization {
    expert_memory_mode: Option<ExpertMemoryMode>,
    mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    expert_residency_telemetry: Option<ExpertResidencyTelemetry>,
    expert_streaming_stats: Option<Box<ExpertStreamingStats>>,
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
            expert_streaming_stats: None,
        }
    }

    /// Attaches session expert-streaming stats without disturbing `Copy`.
    #[must_use]
    pub fn with_expert_streaming_stats(
        mut self,
        expert_streaming_stats: Option<Box<ExpertStreamingStats>>,
    ) -> Self {
        self.expert_streaming_stats = expert_streaming_stats;
        self
    }

    #[must_use]
    pub fn expert_streaming_stats(&self) -> Option<&ExpertStreamingStats> {
        self.expert_streaming_stats.as_deref()
    }

    #[must_use]
    pub const fn expert_memory_mode(&self) -> Option<ExpertMemoryMode> {
        self.expert_memory_mode
    }

    #[must_use]
    pub const fn mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.mlx_memory_telemetry
    }

    #[must_use]
    pub const fn expert_residency_telemetry(&self) -> Option<ExpertResidencyTelemetry> {
        self.expert_residency_telemetry
    }

    /// Placement, memory, and residency are the pre-session facts. Streaming
    /// stats are a separate report, so they stay out of this predicate and
    /// the caller combines the two answers.
    #[must_use]
    const fn has_placement_or_memory_telemetry(&self) -> bool {
        self.expert_memory_mode.is_some()
            || self.mlx_memory_telemetry.is_some()
            || self.expert_residency_telemetry.is_some()
    }

    #[must_use]
    pub const fn has_reportable_state(&self) -> bool {
        self.has_placement_or_memory_telemetry() || self.expert_streaming_stats.is_some()
    }
}

/// One bounded progress boundary from the engine.
// The inline `Option<MlxMemoryTelemetry>` carries the seating array so a
// residency sample stays `Copy` on the hot path; boxing it would trade a
// size gap on a transient value for an allocation per decode token.
#[allow(clippy::large_enum_variant)]
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
        /// Boxed because it embeds full memory and expert-residency telemetry
        /// (including the 128-entry seating array) yet appears on a single
        /// terminal token; boxing keeps every decode token from moving it.
        generation_finalization: Option<Box<GenerationFinalization>>,
    },
    /// One bounded native prefill chunk completed without producing a token yet.
    PrefillProgress {
        /// Tokens processed by this chunk alone. The worker accumulates these
        /// per-chunk deltas into the request-wide prefill cursor; returning a
        /// cumulative count instead double-counts every earlier chunk and pins
        /// the status surface at 100% long before prefill completes.
        processed_token_count: u32,
        elapsed_millis: u64,
        /// Time spent evaluating the model, excluding allocator cleanup and telemetry.
        forward_prefill_chunk_elapsed_millis: u64,
        /// Selected `prefill_chunk_tokens` used for this completed prompt-processing chunk.
        completed_prefill_chunk_tokens: u32,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
        /// Current retained-expert ownership after this chunk.
        expert_residency_telemetry: Option<ExpertResidencyTelemetry>,
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
    ///
    /// The memory observation is captured at the same owner-thread instant as the
    /// residency so the public status never pairs fresh residency with a stale
    /// prefill-era measurement (issue #337).
    GenerationPreparationStarted {
        expert_residency: ExpertResidencyTelemetry,
        mlx_memory_telemetry: Option<crate::memory::telemetry::MlxMemoryTelemetry>,
    },
    /// Engine-side end-of-sequence without an explicit token ID.
    EndOfSequence,
}

impl GeneratedToken {
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
            *generated_token_finalization = Some(Box::new(generation_finalization));
        }
        self
    }
}
