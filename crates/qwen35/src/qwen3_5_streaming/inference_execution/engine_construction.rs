//! Construction of the streaming Qwen3.5 inference engine from a validated artifact.

use std::path::PathBuf;

use ipc_protocol::{WorkerChunkingConfiguration, WorkerDecodeAstroExpertCachePolicy};
use runtime_integration::MlxMemoryLimits;

use crate::astro_ssd_prompt_cache::counters::AstroSsdPromptCacheCounters;
use crate::{
    AstroSsdPromptCacheDiskStoreConfig, InferenceEngineError, MlxInferenceEngine,
    PerformanceAttribution, PerformanceAttributionLog,
};

use super::super::ValidatedQwen3_5Artifact;
use super::{
    Qwen3_5InferenceExecution, Qwen3_5StreamingEngine, Qwen3_5StreamingPromptProcessingChunkSizer,
    fatal_engine_error, qwen3_5_runtime_error,
};

impl MlxInferenceEngine<Qwen3_5InferenceExecution> {
    /// Starts the owner thread with an explicit `prefill_chunk_tokens` sizer.
    ///
    /// The model directory remains required for sparse expert paging.
    // Construction dependencies remain explicit to avoid another configuration facade.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_prompt_processing_chunk_sizer(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5StreamingPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
    ) -> Result<Qwen3_5StreamingEngine, InferenceEngineError> {
        Self::new_with_runtime_chunking_and_performance_attribution(
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            chunking,
            PerformanceAttribution::disabled(),
            PerformanceAttributionLog::disabled(),
        )
    }
    /// Starts the owner thread with every model-serving work boundary resolved.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_runtime_chunking_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5StreamingPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Qwen3_5StreamingEngine, InferenceEngineError> {
        let maximum_context_tokens = validated_artifact.config().maximum_position_count();
        Self::new_with_effective_context_runtime_chunking_and_performance_attribution(
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            maximum_context_tokens,
            chunking,
            model_loading_performance_attribution,
            performance_attribution_log,
        )
    }

    /// Starts the owner thread with a config-resolved context no larger than the artifact.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_effective_context_runtime_chunking_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5StreamingPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        maximum_context_tokens: u32,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Qwen3_5StreamingEngine, InferenceEngineError> {
        Self::new_with_effective_context_runtime_chunking_performance_attribution_and_decode_astro_expert_cache_policy(
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            maximum_context_tokens,
            chunking,
            model_loading_performance_attribution,
            performance_attribution_log,
            WorkerDecodeAstroExpertCachePolicy::Baseline,
        )
    }

    /// Starts the owner thread with an internal decode `astro_expert_cache` benchmark policy.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_effective_context_runtime_chunking_performance_attribution_and_decode_astro_expert_cache_policy(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5StreamingPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        maximum_context_tokens: u32,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
        decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    ) -> Result<Qwen3_5StreamingEngine, InferenceEngineError> {
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
        let hard_maximum_position_count = artifact_maximum_position_count as usize;
        let vocabulary_size = validated_artifact.config().vocabulary_size();
        let context_memory_reservation_bytes_per_token = validated_artifact
            .config()
            .context_memory_reservation_bytes(1)
            .ok_or_else(|| fatal_engine_error("Qwen3.5 context memory reservation overflowed"))?;
        let memory_limits =
            MlxMemoryLimits::new(active_memory_limit_bytes, mlx_buffer_cache_limit_bytes)
                .map_err(qwen3_5_runtime_error)?;
        MlxInferenceEngine::new(move || Qwen3_5InferenceExecution {
            active_request: None,
            decode_astro_expert_cache_policy,
            astro_ssd_prompt_cache_disk_store_config,
            astro_ssd_prompt_cache_counters: AstroSsdPromptCacheCounters::default(),
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
            prompt_span_to_fail_once_for_tests: None,
            astro_ssd_prompt_cache_model_contract: None,
            persistent_visual_embedding_model_contract: None,
            astro_ssd_prompt_cache: None,
            prompt_processing_chunk_sizer,
            chunking,
            validated_artifact: Some(validated_artifact),
            startup_residency_facts: None,
            vocabulary_size,
        })
    }
}
