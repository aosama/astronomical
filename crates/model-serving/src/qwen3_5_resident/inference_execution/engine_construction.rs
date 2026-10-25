//! Construction of the resident Qwen3.5 inference engine from a validated artifact.

use std::path::PathBuf;

use astronomical_ipc_protocol::WorkerChunkingConfiguration;
use astronomical_runtime_integration::MlxMemoryLimits;

use crate::persistent_cache::counters::PersistentPromptCacheCounters;
use crate::{
    AdaptiveRamGrowthGuard, InferenceEngineError, MlxInferenceEngine, PerformanceAttribution,
    PerformanceAttributionLog, PersistentPromptCacheDiskStoreConfig,
};

use super::super::ValidatedQwen3_5Artifact;
use super::Qwen3_5InferenceExecution;
use super::Qwen3_5ResidentEngine;
use super::Qwen3_5ResidentPromptProcessingChunkSizer;
use super::engine_error_helpers::fatal_engine_error;
use super::engine_error_helpers::qwen3_5_runtime_error;

/// Wraps one adaptive-RAM-growth configuration failure as a fatal engine error.
fn adaptive_ram_growth_configuration_error(
    adaptive_ram_growth_guard_error: impl std::fmt::Display,
) -> InferenceEngineError {
    super::fatal_engine_error(format!(
        "failed to configure adaptive RAM growth: {adaptive_ram_growth_guard_error}"
    ))
}

impl MlxInferenceEngine<Qwen3_5InferenceExecution> {
    /// Starts the owner thread with an explicit `prefill_chunk_tokens` sizer.
    ///
    /// The model directory contains the complete resident expert tensors.
    // Construction dependencies remain explicit to avoid another configuration facade.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_prompt_processing_chunk_sizer(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5ResidentPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
    ) -> Result<Qwen3_5ResidentEngine, InferenceEngineError> {
        Self::new_with_runtime_chunking_and_performance_attribution(
            validated_artifact,
            active_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
            persistent_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            chunking,
            true,
            PerformanceAttribution::disabled(),
            PerformanceAttributionLog::disabled(),
        )
    }
    /// Starts the owner thread with every model-serving work boundary resolved.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_runtime_chunking_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5ResidentPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
        adaptive_ram_growth_guard_enabled: bool,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Qwen3_5ResidentEngine, InferenceEngineError> {
        let maximum_context_tokens = validated_artifact.config().maximum_position_count();
        Self::new_with_effective_context_runtime_chunking_and_performance_attribution(
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
            model_loading_performance_attribution,
            performance_attribution_log,
            None,
        )
    }

    /// Starts the owner thread with a config-resolved context no larger than the artifact.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_effective_context_runtime_chunking_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5ResidentPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        maximum_context_tokens: u32,
        chunking: WorkerChunkingConfiguration,
        adaptive_ram_growth_guard_enabled: bool,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
        diagnostic_prefill_activation_floor_bytes: Option<u64>,
    ) -> Result<Qwen3_5ResidentEngine, InferenceEngineError> {
        let full_attention_kv_state_growth_tokens =
            i32::try_from(chunking.full_attention_key_value_growth_tokens)
            .map_err(super::generation_session_lifecycle::generation_error_constructors::full_attention_growth_tokens_exceed_i32_range_error)?;
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
        let hard_maximum_position_count = artifact_maximum_position_count as usize;
        let vocabulary_size = validated_artifact.config().vocabulary_size();
        let context_memory_reservation_bytes_per_token = validated_artifact
            .config()
            .context_memory_reservation_bytes(1)
            .ok_or_else(super::generation_session_lifecycle::generation_error_constructors::context_memory_reservation_overflowed_error)?;
        let memory_limits = MlxMemoryLimits::new(
            active_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
        )
        .map_err(qwen3_5_runtime_error)?;
        let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(active_memory_limit_bytes)
            .map_err(adaptive_ram_growth_configuration_error)?;
        MlxInferenceEngine::new(move || Qwen3_5InferenceExecution {
            active_request: None,
            adaptive_ram_growth_guard,
            adaptive_ram_growth_guard_enabled,
            diagnostic_prefill_activation_floor_bytes,
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
            model_loading_performance_attribution: Some(model_loading_performance_attribution),
            performance_attribution_log,
            maximum_position_count,
            hard_maximum_position_count,
            model: None,
            persistent_prompt_cache_model_contract: None,
            persistent_visual_embedding_model_contract: None,
            persistent_prompt_cache: None,
            prompt_processing_chunk_sizer,
            chunking,
            retained_live_prefix: None,
            validated_artifact: Some(validated_artifact),
            vocabulary_size,
        })
    }
}
