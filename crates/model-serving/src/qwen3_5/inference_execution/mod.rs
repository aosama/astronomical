mod advance_generation;
mod completed_forward_memory;
/// Prefill-to-decode no-I/O reconciliation: lift pressure and preserve topology.
mod decode_expert_memory_handoff;
mod decoder_state_reuse;
pub(crate) mod engine_request;
pub(in crate::qwen3_5) mod generated_token_emission;
mod generation_finalization;
mod generation_request_validation;
mod initial_context_memory_admission;
mod inject_input_tokens;
pub(in crate::qwen3_5) mod memory_admission;
mod memory_limit;
mod model_loading;
mod model_loading_finalization;
mod mtp_decode_attempt;
mod persistent_prompt_cache_capture;
mod persistent_prompt_cache_startup_logging;
mod persistent_prompt_cache_visual_identity;
mod prefill_advance;
mod prefill_capacity_recovery;
mod prefill_chunk_planning;
mod prefill_execution_context;
mod prefill_forward_dispatch;
mod prefill_memory_admission;
mod prompt_prefill;
mod prompt_prefill_counters;
mod prompt_prefill_errors;
mod prompt_processing_chunk_sizer;
mod request_memory_release;
mod resident_memory_pressure;
mod speculative_prefill;
mod start_generation;
mod terminal_prefill_seed;
mod test_controls;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use astronomical_ipc_protocol::{
    MtpDepthStatus, RequestId, SpeculativePrefillRuntimeState, WorkerChunkingConfiguration,
    WorkerEvent, WorkerSpeculativePrefillConfiguration,
};
use astronomical_runtime_integration::MlxMemoryLimits;

use crate::{
    AdaptiveRamGrowthGuard, EngineGenerationStart, EngineLoadResult, GeneratedToken,
    GenerationFinalization, InferenceEngineError, MlxInferenceEngine, MlxInferenceExecution,
    MlxMemoryLimitAdjustment, MlxMemoryTelemetry, PerformanceAttribution,
    PerformanceAttributionLog, PerformanceAttributionOutcome, PersistentPromptCacheCounters,
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheModelContract, PersistentVisualEmbeddingModelContract,
    Qwen3_5InferenceRequest, build_persistent_prompt_cache_stats_event,
};

use self::engine_request::Qwen3_5EngineRequest;
pub use self::engine_request::Qwen3_5SpeculativePrefillFailureStageForTests;
pub use self::speculative_prefill::{
    Qwen3_5SpeculativePrefillChunkMode, Qwen3_5SpeculativePrefillSelectionError,
    qwen3_5_prefill_chunk_end_at_ordinary_target_control_span_boundary,
    qwen3_5_prompt_prefill_end_exclusive, qwen3_5_select_speculative_prefill_token_positions,
    qwen3_5_selected_speculative_prefill_positions_for_range,
    qwen3_5_speculative_prefill_chunk_mode, qwen3_5_speculative_prefill_sparse_target_is_active,
};
use super::model::Qwen3_5Model;
use super::{MtpDraftDepth, ValidatedQwen3_5Artifact};

pub use crate::qwen3_5::multi_token_prediction::Qwen3_5MtpRuntimeState;
pub use crate::qwen3_5::multi_token_prediction::qwen3_5_depth_one_mtp_window_fits;
pub use crate::qwen3_5::multi_token_prediction::{
    qwen3_5_mtp_runtime_configuration_after_load, qwen3_5_mtp_runtime_state_after_load,
};
pub use memory_limit::safe_minimum_mlx_memory_ceiling_bytes;
pub use persistent_prompt_cache_capture::persistent_prompt_cache_publication_advances_parent_chain;
pub use prefill_execution_context::Qwen3_5PrefillExecutionContext;
pub use prompt_processing_chunk_sizer::{
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5PromptProcessingChunkSizerError,
};

/// Qwen3.5 inference engine backed by the architecture-neutral MLX owner driver.
pub type Qwen3_5Engine = MlxInferenceEngine<Qwen3_5InferenceExecution>;

impl MlxInferenceEngine<Qwen3_5InferenceExecution> {
    /// Starts the owner thread with an explicit `prefill_chunk_tokens` sizer.
    ///
    /// The model directory remains required for sparse expert paging.
    // Construction dependencies remain explicit to avoid another configuration facade.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_prompt_processing_chunk_sizer(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
        mtp_enabled: bool,
        speculative_prefill: WorkerSpeculativePrefillConfiguration,
    ) -> Result<Qwen3_5Engine, InferenceEngineError> {
        Self::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
            validated_artifact,
            active_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
            persistent_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            chunking,
            true,
            mtp_enabled,
            speculative_prefill,
            PerformanceAttribution::disabled(),
            PerformanceAttributionLog::disabled(),
        )
    }
    /// Starts the owner thread with every model-serving work boundary resolved.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
        adaptive_ram_growth_guard_enabled: bool,
        mtp_enabled: bool,
        speculative_prefill: WorkerSpeculativePrefillConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Qwen3_5Engine, InferenceEngineError> {
        let maximum_context_tokens = validated_artifact.config().maximum_position_count();
        Self::new_with_effective_context_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
            validated_artifact,
            active_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
            persistent_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            maximum_context_tokens,
            chunking,
            adaptive_ram_growth_guard_enabled,
            mtp_enabled,
            None,
            speculative_prefill,
            model_loading_performance_attribution,
            performance_attribution_log,
        )
    }

    /// Starts the owner thread with explicit fixed MTP depth selection metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
        adaptive_ram_growth_guard_enabled: bool,
        mtp_enabled: bool,
        mtp_draft_depth: Option<u8>,
        speculative_prefill: WorkerSpeculativePrefillConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Qwen3_5Engine, InferenceEngineError> {
        let maximum_context_tokens = validated_artifact.config().maximum_position_count();
        Self::new_with_effective_context_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
            validated_artifact,
            active_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
            persistent_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            maximum_context_tokens,
            chunking,
            adaptive_ram_growth_guard_enabled,
            mtp_enabled,
            mtp_draft_depth,
            speculative_prefill,
            model_loading_performance_attribution,
            performance_attribution_log,
        )
    }

    /// Starts the owner thread with a config-resolved context no larger than the artifact.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_effective_context_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        maximum_context_tokens: u32,
        chunking: WorkerChunkingConfiguration,
        adaptive_ram_growth_guard_enabled: bool,
        mtp_enabled: bool,
        mtp_draft_depth: Option<u8>,
        speculative_prefill: WorkerSpeculativePrefillConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Qwen3_5Engine, InferenceEngineError> {
        let configured_mtp_draft_depth = mtp_draft_depth
            .map(MtpDraftDepth::new)
            .transpose()
            .map_err(|_| fatal_engine_error("MTP draft depth must be between 1 and 3"))?;
        let full_attention_kv_state_growth_tokens =
            i32::try_from(chunking.full_attention_key_value_growth_tokens).map_err(|_| {
                fatal_engine_error("full-attention growth tokens exceed Int32 range")
            })?;
        let end_of_sequence_token_ids = validated_artifact
            .config()
            .end_of_sequence_token_ids()
            .to_vec();
        let artifact_maximum_position_count = validated_artifact.config().maximum_position_count();
        if maximum_context_tokens == 0 || maximum_context_tokens > artifact_maximum_position_count {
            return Err(fatal_engine_error(
                "effective Qwen3.5 context must be positive and not exceed the artifact context",
            ));
        }
        let maximum_position_count = maximum_context_tokens as usize;
        let vocabulary_size = validated_artifact.config().vocabulary_size();
        let context_memory_reservation_bytes_per_token = validated_artifact
            .config()
            .context_memory_reservation_bytes(1)
            .ok_or_else(|| fatal_engine_error("Qwen3.5 context memory reservation overflowed"))?;
        let memory_limits = MlxMemoryLimits::new(
            active_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
        )
        .map_err(qwen3_5_runtime_error)?;
        let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(active_memory_limit_bytes)
            .map_err(|adaptive_ram_growth_guard_error| {
                fatal_engine_error(format!(
                    "failed to configure adaptive RAM growth: {adaptive_ram_growth_guard_error}"
                ))
            })?;
        let initial_speculative_prefill_runtime_state = if speculative_prefill.enabled {
            SpeculativePrefillRuntimeState::Unavailable
        } else {
            SpeculativePrefillRuntimeState::Disabled
        };
        MlxInferenceEngine::new(move || Qwen3_5InferenceExecution {
            active_request: None,
            adaptive_ram_growth_guard,
            adaptive_ram_growth_guard_enabled,
            persistent_prompt_cache_disk_store_config,
            persistent_prompt_cache_counters: PersistentPromptCacheCounters::default(),
            context_memory_reservation_bytes_per_token,
            end_of_sequence_token_ids,
            think_end_token_id,
            full_attention_kv_state_growth_tokens,
            memory_limits,
            model_directory,
            model_id: None,
            model_revision: None,
            speculative_prefill_draft_model_revision: None,
            speculative_prefill_draft_is_available: false,
            speculative_prefill_draft_supports_processed_visual_images: false,
            speculative_prefill_token_identifier_mapping_digest: None,
            model_loading_performance_attribution: Some(model_loading_performance_attribution),
            performance_attribution_log,
            maximum_position_count,
            model: None,
            speculative_prefill_draft_model: None,
            speculative_prefill_selection_store: RefCell::new(HashMap::new()),
            speculative_prefill_draft_prefix_store: RefCell::new(HashMap::new()),
            persistent_prompt_cache_model_contract: None,
            persistent_visual_embedding_model_contract: None,
            persistent_prompt_cache: None,
            speculative_prefill_draft_persistent_prompt_cache: None,
            prompt_processing_chunk_sizer,
            chunking,
            validated_artifact: Some(validated_artifact),
            vocabulary_size,
            mtp_enabled,
            configured_mtp_draft_depth,
            speculative_prefill,
            mtp_runtime_state: if mtp_enabled {
                Qwen3_5MtpRuntimeState::Unavailable
            } else {
                Qwen3_5MtpRuntimeState::Disabled
            },
            mtp_unavailable_reason: None,
            mtp_depth_status: MtpDepthStatus {
                configured_draft_depth: mtp_draft_depth,
                ..MtpDepthStatus::default()
            },
            speculative_prefill_runtime_state: initial_speculative_prefill_runtime_state,
            speculative_prefill_unavailable_reason: None,
        })
    }
}

pub struct Qwen3_5InferenceExecution {
    pub(super) active_request: Option<Qwen3_5EngineRequest>,
    pub(super) adaptive_ram_growth_guard: AdaptiveRamGrowthGuard,
    pub(super) adaptive_ram_growth_guard_enabled: bool,
    persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
    pub(in super::super) persistent_prompt_cache_counters: PersistentPromptCacheCounters,
    pub(super) context_memory_reservation_bytes_per_token: usize,
    end_of_sequence_token_ids: Vec<u32>,
    /// Token ID that closes the thinking block. Used to enforce thinking_budget.
    think_end_token_id: u32,
    /// Capacity-growth granularity for full-attention KV slabs.
    full_attention_kv_state_growth_tokens: i32,
    pub(super) memory_limits: MlxMemoryLimits,
    /// Directory containing the safetensors model shards. Needed for
    /// expert paging to read safetensors headers at startup.
    model_directory: PathBuf,
    model_id: Option<String>,
    model_revision: Option<String>,
    pub(super) speculative_prefill_draft_model_revision: Option<String>,
    /// Startup validation outcome for the configured request-scoped draft model.
    pub(super) speculative_prefill_draft_is_available: bool,
    /// Whether the validated request-scoped draft accepts target-processed images.
    pub(super) speculative_prefill_draft_supports_processed_visual_images: bool,
    /// Canonical token-to-identifier mapping shared by target and draft artifacts.
    pub(super) speculative_prefill_token_identifier_mapping_digest: Option<[u8; 32]>,
    model_loading_performance_attribution: Option<PerformanceAttribution>,
    performance_attribution_log: PerformanceAttributionLog,
    maximum_position_count: usize,
    pub(super) model: Option<Qwen3_5Model>,
    /// Request-scoped draft model, present only while scoring an eligible prompt.
    pub(super) speculative_prefill_draft_model: Option<Qwen3_5Model>,
    /// Bounded worker-local selection store keyed by the exact draft-scored prompt.
    pub(super) speculative_prefill_selection_store:
        RefCell<HashMap<speculative_prefill::Qwen3_5SpeculativePrefillStoreKey, Vec<usize>>>,
    /// Bounded worker-local draft decoder checkpoints isolated from target state.
    pub(super) speculative_prefill_draft_prefix_store: RefCell<
        HashMap<
            speculative_prefill::Qwen3_5SpeculativePrefillStoreKey,
            speculative_prefill::Qwen3_5SpeculativePrefillDraftPrefixStoreEntry,
        >,
    >,
    pub(crate) persistent_prompt_cache_model_contract: Option<PersistentPromptCacheModelContract>,
    pub(crate) persistent_visual_embedding_model_contract:
        Option<PersistentVisualEmbeddingModelContract>,
    pub(in super::super) persistent_prompt_cache: Option<Arc<PersistentPromptCacheDiskStore>>,
    /// SSD-backed dense decoder state owned by the configured SpecPrefill drafter.
    pub(in super::super) speculative_prefill_draft_persistent_prompt_cache:
        Option<Arc<PersistentPromptCacheDiskStore>>,
    prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
    chunking: WorkerChunkingConfiguration,
    validated_artifact: Option<ValidatedQwen3_5Artifact>,
    vocabulary_size: u32,
    /// User preference: whether MTP is enabled.
    /// Defaults to false until the worker passes the real config value.
    mtp_enabled: bool,
    configured_mtp_draft_depth: Option<MtpDraftDepth>,
    /// Resolved optional draft-assisted speculative-prefill configuration.
    pub(super) speculative_prefill: WorkerSpeculativePrefillConfiguration,
    /// Actual MTP runtime state after model loading.
    mtp_runtime_state: Qwen3_5MtpRuntimeState,
    /// Concise reason when MTP runtime state is Unavailable.
    mtp_unavailable_reason: Option<String>,
    mtp_depth_status: MtpDepthStatus,
    /// Actual optional draft-assisted speculative-prefill state after model loading.
    speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
    /// Concise reason when speculative prefill is Unavailable.
    speculative_prefill_unavailable_reason: Option<String>,
}

pub(in crate::qwen3_5) type Qwen3_5EngineState = Qwen3_5InferenceExecution;

impl MlxInferenceExecution for Qwen3_5InferenceExecution {
    type Request = Qwen3_5InferenceRequest;

    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Qwen3_5EngineState::load(self)
    }

    fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Qwen3_5EngineState::start_generation(self, inference_request)
    }

    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Qwen3_5EngineState::advance_generation(self, request_id)
    }

    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        Qwen3_5EngineState::inject_input_tokens(self, request_id, input_token_ids)
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Qwen3_5EngineState::cancel_generation(self, request_id)
    }

    fn collect_persistent_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(Qwen3_5EngineState::collect_persistent_prompt_cache_stats(
            self,
        ))
    }

    fn clear_persistent_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let Some(persistent_prompt_cache) = self.persistent_prompt_cache.as_ref() else {
            return Ok(None);
        };
        let clear_outcome = persistent_prompt_cache
            .clear_prompt_cache(model_id.as_deref())
            .map_err(|clear_error| fatal_engine_error(clear_error.to_string()))?;
        Ok(Some(WorkerEvent::PromptCacheCleared {
            model_id: clear_outcome.model_id,
            blocks_removed: clear_outcome.blocks_removed,
            bytes_freed: clear_outcome.bytes_freed,
        }))
    }

    fn collect_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        Qwen3_5EngineState::collect_current_mlx_memory_telemetry(self)
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        Qwen3_5EngineState::update_mlx_memory_limit(self, requested_mlx_memory_ceiling_bytes)
    }
}

impl Qwen3_5EngineState {
    fn collect_persistent_prompt_cache_stats(&self) -> Option<WorkerEvent> {
        let persistent_prompt_cache = self.persistent_prompt_cache.as_ref()?;
        let global_prompt_cache_maximum_size_bytes = self
            .persistent_prompt_cache_disk_store_config
            .as_ref()
            .map(|disk_store_config| disk_store_config.global_prompt_cache_maximum_size_bytes())?;
        Some(build_persistent_prompt_cache_stats_event(
            &self.persistent_prompt_cache_counters,
            u64::try_from(
                persistent_prompt_cache
                    .model_contract_ref()
                    .block_token_count(),
            )
            .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.sequence_state_block_count()).unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.boundary_state_snapshot_count())
                .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.visual_embedding_count()).unwrap_or(u64::MAX),
            persistent_prompt_cache.total_size_bytes(),
            persistent_prompt_cache.visual_embedding_total_size_bytes(),
            global_prompt_cache_maximum_size_bytes,
        ))
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        let Some(active_request) = self.active_request.take() else {
            return Ok(GenerationFinalization::default());
        };
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            Ok(GenerationFinalization::default())
        } else {
            Ok(self.finalize_generation_request(
                active_request,
                PerformanceAttributionOutcome::Cancelled,
                None,
            ))
        }
    }
}

pub(crate) fn qwen3_5_runtime_error(runtime_error: impl std::fmt::Display) -> InferenceEngineError {
    fatal_engine_error(runtime_error.to_string())
}

pub(super) fn fatal_engine_error(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: reason.into(),
    }
}
