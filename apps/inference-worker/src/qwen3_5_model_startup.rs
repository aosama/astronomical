use std::path::PathBuf;

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    WorkerChunkingConfiguration, WorkerSpeculativePrefillConfiguration,
};
use astronomical_model_serving::{
    PerformanceAttribution, PerformanceAttributionLog, PerformanceAttributionOutcome,
    PerformanceOperation, PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator,
    Qwen3_5Engine, Qwen3_5GenerationProcessor, Qwen3_5PromptProcessingChunkSizer,
};

use crate::qwen3_5_model_startup_error::Qwen3_5ModelStartupError;
use astronomical_model_serving::ModelLoadingPerformanceAttributionMetadata;

#[allow(clippy::too_many_arguments)]
pub(crate) fn initialize_qwen3_5_model(
    model_directory_path: PathBuf,
    active_memory_limit_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    prompt_cache_config: PromptCacheConfig,
    max_output_tokens: u32,
    mtp_enabled: bool,
    mtp_draft_depth: Option<u8>,
    speculative_prefill: WorkerSpeculativePrefillConfiguration,
    persistent_prompt_cache_enabled: bool,
    performance_attribution_enabled: bool,
    performance_attribution_log_path: PathBuf,
    chunking: WorkerChunkingConfiguration,
) -> Result<(Qwen3_5GenerationProcessor, Qwen3_5Engine), Qwen3_5ModelStartupError> {
    let mut model_loading_performance_attribution = if performance_attribution_enabled {
        PerformanceAttribution::enabled()
    } else {
        PerformanceAttribution::disabled()
    };
    let mut performance_attribution_log = PerformanceAttributionLog::open(
        &performance_attribution_log_path,
        performance_attribution_enabled,
    )
    .map_err(
        |source| Qwen3_5ModelStartupError::OpenPerformanceAttributionLog {
            log_path: performance_attribution_log_path.clone(),
            source,
        },
    )?;
    let validated_artifact = match model_loading_performance_attribution.measure_operation(
        PerformanceOperation::ArtifactValidation,
        |_performance_attribution| {
            Qwen3_5ArtifactValidator::new().validate(&model_directory_path, max_output_tokens)
        },
    ) {
        Ok(validated_artifact) => validated_artifact,
        Err(source) => {
            tracing::warn!(
                error = %source,
                model_directory = ?model_directory_path,
                "Qwen3.5 artifact validation failed during model initialization"
            );
            record_failed_model_loading_performance_attribution(
                model_loading_performance_attribution,
                &mut performance_attribution_log,
                None,
                None,
                None,
                None,
                "artifact validation failed",
            );
            return Err(Qwen3_5ModelStartupError::ArtifactValidation {
                model_directory: model_directory_path.clone(),
                source,
            });
        }
    };
    let artifact_model_id = validated_artifact.model_id().to_owned();
    let loaded_model_speculative_prefill_configuration =
        speculative_prefill.for_loaded_model(&artifact_model_id);
    let artifact_model_revision = validated_artifact.revision().to_owned();
    let artifact_payload_bytes = validated_artifact.total_payload_bytes();
    let artifact_shard_count = validated_artifact.shard_count();
    let generation_processor = match model_loading_performance_attribution.measure_operation(
        PerformanceOperation::TokenizerInitialization,
        |_performance_attribution| {
            Qwen3_5GenerationProcessor::from_validated_artifact(
                &validated_artifact,
                true,
                performance_attribution_enabled,
            )
        },
    ) {
        Ok(generation_processor) => generation_processor,
        Err(source) => {
            record_failed_model_loading_performance_attribution(
                model_loading_performance_attribution,
                &mut performance_attribution_log,
                Some(artifact_model_id),
                Some(artifact_model_revision),
                Some(artifact_payload_bytes),
                Some(artifact_shard_count),
                "tokenizer initialization failed",
            );
            return Err(Qwen3_5ModelStartupError::ProcessorInitialization {
                model_directory: model_directory_path.clone(),
                source,
            });
        }
    };
    let think_end_token_id = generation_processor.think_end_token_id();
    let model_id = validated_artifact.model_id().to_owned();
    let model_revision = validated_artifact.revision().to_owned();
    // Derive the model-specific path only when the user enabled persistent
    // caching. This keeps disabled operation from touching model/revision cache
    // identity, opening directories, scanning files, or reserving publication
    // workspace later in model loading.
    let persistent_prompt_cache_disk_store_config = persistent_prompt_cache_enabled.then(|| {
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
    let prompt_processing_chunk_sizer_result =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
            chunking.fixed_prompt_processing_chunk_size_tokens,
            chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
        );
    let prompt_processing_chunk_sizer = match prompt_processing_chunk_sizer_result {
        Ok(prompt_processing_chunk_sizer) => prompt_processing_chunk_sizer,
        Err(prompt_processing_chunk_sizer_error) => {
            record_failed_model_loading_performance_attribution(
                model_loading_performance_attribution,
                &mut performance_attribution_log,
                Some(artifact_model_id),
                Some(artifact_model_revision),
                Some(artifact_payload_bytes),
                Some(artifact_shard_count),
                "prompt-processing chunk configuration failed",
            );
            return Err(Qwen3_5ModelStartupError::PromptProcessingChunkSizing(
                prompt_processing_chunk_sizer_error,
            ));
        }
    };
    let qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
        validated_artifact,
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        persistent_prompt_cache_disk_store_config,
        prompt_processing_chunk_sizer,
        think_end_token_id,
        model_directory_path.clone(),
        chunking,
        true,
        mtp_enabled,
        mtp_draft_depth,
        loaded_model_speculative_prefill_configuration,
        model_loading_performance_attribution,
        performance_attribution_log,
    )
    .map_err(|source| Qwen3_5ModelStartupError::EngineInitialization {
        model_directory: model_directory_path,
        source,
    })?;
    Ok((generation_processor, qwen3_5_engine))
}

#[allow(clippy::too_many_arguments)]
fn record_failed_model_loading_performance_attribution(
    model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: &mut PerformanceAttributionLog,
    model_id: Option<String>,
    model_revision: Option<String>,
    total_artifact_payload_bytes: Option<u64>,
    model_shard_count: Option<usize>,
    failure_description: &'static str,
) {
    let Some(performance_attribution_report) = model_loading_performance_attribution
        .finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
            outcome: PerformanceAttributionOutcome::Failed,
            model_id,
            model_revision,
            prefill_transient_observation_completed: false,
            prefill_observed_transient_high_water_bytes: 0,
            total_artifact_payload_bytes,
            resident_model_payload_bytes: None,
            model_shard_count,
            mlx_active_memory_bytes: None,
            mlx_allocator_cache_memory_bytes: None,
            mlx_peak_memory_bytes: None,
            failure_description: Some(failure_description.to_owned()),
        })
    else {
        return;
    };
    if let Err(performance_attribution_write_error) =
        performance_attribution_log.record(&performance_attribution_report)
    {
        tracing::warn!(
            error = %performance_attribution_write_error,
            "failed to append model-loading performance attribution"
        );
    }
}
