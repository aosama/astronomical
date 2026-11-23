//! Construction of the resident Qwen3.5 inference engine from a validated artifact.

use std::path::PathBuf;

use ipc_protocol::WorkerChunkingConfiguration;
use runtime_integration::MlxMemoryLimits;

use crate::astro_ssd_prompt_cache::counters::AstroSsdPromptCacheCounters;
use crate::{
    AstroSsdPromptCacheDiskStoreConfig, InferenceEngineError, MlxInferenceEngine,
    PerformanceAttribution, PerformanceAttributionLog,
};

use super::super::ValidatedQwen3_5Artifact;
use super::Qwen3_5InferenceExecution;
use super::Qwen3_5ResidentEngine;
use super::engine_error_helpers::fatal_engine_error;
use super::engine_error_helpers::qwen3_5_runtime_error;
use crate::qwen3_5_core::shared_execution::execution_core::{
    Qwen3_5ExecutionCore, Qwen3_5ExecutionCoreSettings,
};
use crate::qwen3_5_core::shared_execution::prompt_processing_chunk_sizer::Qwen3_5PromptProcessingChunkSizer;

impl MlxInferenceEngine<Qwen3_5InferenceExecution> {
    /// Starts the owner thread with an explicit `prefill_chunk_tokens` sizer.
    ///
    /// The model directory contains the complete resident expert tensors.
    // Construction dependencies remain explicit to avoid another configuration facade.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_prompt_processing_chunk_sizer(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
    ) -> Result<Qwen3_5ResidentEngine, InferenceEngineError> {
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
        prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Qwen3_5ResidentEngine, InferenceEngineError> {
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
            None,
        )
    }

    /// Starts the owner thread with a config-resolved context no larger than the artifact.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_effective_context_runtime_chunking_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        maximum_context_tokens: u32,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
        astro_neural_engine_program_cache_directory: Option<PathBuf>,
    ) -> Result<Qwen3_5ResidentEngine, InferenceEngineError> {
        let validated_core_settings = Qwen3_5ExecutionCoreSettings {
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
        }
        .validate()?;
        MlxInferenceEngine::new(move || Qwen3_5InferenceExecution {
            core: Qwen3_5ExecutionCore::new(validated_core_settings),
            neural_engine_prefill_status: ipc_protocol::NeuralEnginePrefillStatus::Off,
            neural_engine_prefill_preparation: None,
            astro_neural_engine_program_cache_directory,
            retained_live_prefix: None,
            speculative_draft_runtime: None,
            draft_model_status: ipc_protocol::DraftModelStatus::None,
        })
    }
}
