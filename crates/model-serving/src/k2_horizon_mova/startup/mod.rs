//! Validates a K2 Horizon MoVA directory and constructs processor plus engine.

use std::path::Path;

use crate::k2_horizon_mova::engine::{
    K2HorizonMoVAEngine, K2HorizonMoVAInferenceExecution, K2HorizonMoVAPendingStartup,
};
use crate::k2_horizon_mova::{
    K2HorizonMoVAArtifactValidator, K2HorizonMoVAGenerationProcessor, K2HorizonMoVAServingSettings,
};
use crate::{
    MlxInferenceEngine, PerformanceAttribution, PerformanceAttributionLog, PerformanceOperation,
    PersistentPromptCacheDiskStoreConfig,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum K2HorizonMoVAStartupError {
    #[error(transparent)]
    Artifact(#[from] crate::k2_horizon_mova::K2HorizonMoVAArtifactValidationError),
    #[error(transparent)]
    Tokenizer(#[from] crate::k2_horizon_mova::K2HorizonMoVATokenizerError),
    #[error("failed to open the K2 Horizon MoVA performance-attribution log")]
    PerformanceAttributionLog(#[source] std::io::Error),
    #[error("{reason}")]
    EngineOwner { reason: String },
}

pub fn initialize_k2_horizon_mova_model(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
) -> Result<(K2HorizonMoVAGenerationProcessor, K2HorizonMoVAEngine), K2HorizonMoVAStartupError> {
    initialize_k2_horizon_mova_model_with_serving_settings(
        model_directory,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        performance_attribution_enabled,
        K2HorizonMoVAServingSettings::default_fixed(),
    )
}

pub fn initialize_k2_horizon_mova_model_with_serving_settings(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
    serving_settings: K2HorizonMoVAServingSettings,
) -> Result<(K2HorizonMoVAGenerationProcessor, K2HorizonMoVAEngine), K2HorizonMoVAStartupError> {
    let (processor, pending) = prepare_startup(
        model_directory,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        performance_attribution_enabled,
        serving_settings,
    )?;
    let engine = MlxInferenceEngine::new(move || K2HorizonMoVAInferenceExecution::pending(pending))
        .map_err(|error| K2HorizonMoVAStartupError::EngineOwner {
            reason: error.to_string(),
        })?;
    Ok((processor, engine))
}

pub fn initialize_k2_horizon_mova_execution(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
) -> Result<
    (
        K2HorizonMoVAGenerationProcessor,
        K2HorizonMoVAInferenceExecution,
    ),
    K2HorizonMoVAStartupError,
> {
    initialize_k2_horizon_mova_execution_with_serving_settings(
        model_directory,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        performance_attribution_enabled,
        K2HorizonMoVAServingSettings::default_fixed(),
    )
}

/// Execution variant that accepts explicit serving settings, for direct-MLX
/// benchmarks and engines that carry their own experimental configuration.
pub fn initialize_k2_horizon_mova_execution_with_serving_settings(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
    serving_settings: K2HorizonMoVAServingSettings,
) -> Result<
    (
        K2HorizonMoVAGenerationProcessor,
        K2HorizonMoVAInferenceExecution,
    ),
    K2HorizonMoVAStartupError,
> {
    let (processor, pending) = prepare_startup(
        model_directory,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        performance_attribution_enabled,
        serving_settings,
    )?;
    Ok((processor, K2HorizonMoVAInferenceExecution::pending(pending)))
}

fn prepare_startup(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
    serving_settings: K2HorizonMoVAServingSettings,
) -> Result<
    (
        K2HorizonMoVAGenerationProcessor,
        K2HorizonMoVAPendingStartup,
    ),
    K2HorizonMoVAStartupError,
> {
    let mut performance_attribution = if performance_attribution_enabled {
        PerformanceAttribution::enabled()
    } else {
        PerformanceAttribution::disabled()
    };
    let validated_artifact = performance_attribution
        .measure_operation(PerformanceOperation::ArtifactValidation, |_| {
            K2HorizonMoVAArtifactValidator::new().validate(model_directory)
        })?;
    let processor = K2HorizonMoVAGenerationProcessor::from_validated_artifact(
        &validated_artifact,
        serving_settings.maximum_context_tokens,
        serving_settings.maximum_output_tokens,
    )?;
    let model_id = validated_artifact.model_id().to_owned();
    let model_revision = validated_artifact.revision().to_owned();
    let prompt_cache_disk_store_config = serving_settings
        .prompt_cache_config
        .as_ref()
        .filter(|_| serving_settings.persistent_prompt_cache_enabled)
        .map(|prompt_cache_config| {
            let per_model_prompt_cache_config =
                prompt_cache_config.for_model(&model_id, &model_revision);
            PersistentPromptCacheDiskStoreConfig::new(
                per_model_prompt_cache_config
                    .active_model_prompt_cache_directory()
                    .clone(),
                per_model_prompt_cache_config
                    .global_prompt_cache_root_directory()
                    .clone(),
                per_model_prompt_cache_config.global_prompt_cache_maximum_size_bytes(),
            )
        });
    let configured_prompt_cache_block_token_count =
        serving_settings.chunking.as_ref().and_then(|chunking| {
            chunking
                .prompt_cache_block_tokens
                .map(|block_token_count| block_token_count as usize)
        });
    let prompt_cache_common_prefix_stride_blocks = serving_settings
        .chunking
        .as_ref()
        .map(|chunking| chunking.prompt_cache_common_prefix_stride_blocks)
        .unwrap_or(1)
        .max(1);
    let full_attention_kv_state_growth_tokens = serving_settings
        .chunking
        .as_ref()
        .map(|chunking| chunking.full_attention_key_value_growth_tokens)
        .unwrap_or(serving_settings.full_attention_kv_state_growth_tokens)
        .max(1);
    let performance_attribution_log =
        match serving_settings.performance_attribution_log_path.as_deref() {
            Some(log_path) => {
                PerformanceAttributionLog::open(log_path, performance_attribution_enabled)
                    .map_err(K2HorizonMoVAStartupError::PerformanceAttributionLog)?
            }
            None => PerformanceAttributionLog::disabled(),
        };
    Ok((
        processor,
        K2HorizonMoVAPendingStartup {
            validated_artifact,
            effective_mlx_memory_ceiling_bytes,
            allocator_cache_memory_limit_bytes,
            prompt_processing_chunk_tokens: serving_settings.prompt_processing_chunk_tokens.max(1),
            performance_attribution,
            performance_attribution_enabled,
            performance_attribution_log,
            attribution_model_id: model_id,
            attribution_model_revision: model_revision,
            prompt_cache_disk_store_config,
            configured_prompt_cache_block_token_count,
            prompt_cache_common_prefix_stride_blocks,
            full_attention_kv_state_growth_tokens,
            decode_stage_attribution_enabled: serving_settings.decode_stage_attribution_enabled,
            quantized_kv_cache_enabled: serving_settings.quantized_kv_cache_enabled,
            fused_expert_decode_enabled: serving_settings.fused_expert_decode_enabled,
        },
    ))
}
