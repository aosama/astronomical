//! The Qwen3.5 autoregressive startup pipeline: validate the model directory's
//! artifacts, derive the RAM geometry and the complete-residency decision,
//! build the tokenizer and processor, and select and initialize the engine.

use crate::TextMlxEngineConstructors;
use config::AstroSsdPromptCacheConfig;
use std::path::PathBuf;

use crate::{
    AstroSsdPromptCacheDiskStoreConfig, EnginePlacement, ModelFamilyInferenceEngine,
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution, PerformanceAttributionLog,
    PerformanceAttributionOutcome, PerformanceOperation, PromptProcessingChunkSizer,
    Qwen3_5ArtifactValidator, Qwen3_5GenerationProcessor, Qwen3_5TextEngine, TextEngineSettings,
    TextExecutionCoreSettings, ValidatedQwen3_5Artifact, boxed_chat_engine,
};
use ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerDecodeAstroExpertCachePolicy,
};

use super::qwen3_5_engine_selection::{
    EngineStart, prompt_chunk_tokens_for, select_qwen3_5_engine_from_artifact,
};
use super::qwen3_5_model_startup_error::Qwen3_5ModelStartupError;
use super::qwen3_5_startup_storage::{
    astro_ssd_prompt_cache_disk_store_config, astro_ssd_prompt_cache_serves_artifact,
    model_astro_neural_engine_program_cache_directory, open_model_loading_attribution,
};
use super::{ChatFamilySettings, ChatRuntimeParts};

/// What the validated artifact says about itself, captured once so any later
/// startup phase that fails can record its failed loading attribution with it.
struct ReviewedArtifactMetadata {
    model_id: String,
    model_revision: String,
    total_payload_bytes: u64,
    shard_count: usize,
}

impl ReviewedArtifactMetadata {
    fn from_validated_artifact(validated_artifact: &ValidatedQwen3_5Artifact) -> Self {
        Self {
            model_id: validated_artifact.model_id().to_owned(),
            model_revision: validated_artifact.revision().to_owned(),
            total_payload_bytes: validated_artifact.total_payload_bytes(),
            shard_count: validated_artifact.shard_count(),
        }
    }
}

/// The folders the engine is built with: the model's `astro_ssd_prompt_cache`
/// disk store and its `astro_neural_engine_program_cache` folder.
struct StartupStorage {
    astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
    astro_neural_engine_program_cache_directory: Option<PathBuf>,
}

/// Validates the folder, selects and starts the engine, and builds the
/// processor, from the worker's chat settings and the model's policy.
pub(super) fn initialize_qwen3_5_model(
    model_directory_path: PathBuf,
    chat_family_settings: ChatFamilySettings,
    model_configuration: WorkerAutoregressiveModelConfiguration,
) -> Result<ChatRuntimeParts, Qwen3_5ModelStartupError> {
    // Every field is named, with no `..`, so a setting added to
    // `ChatFamilySettings` fails to compile here until startup decides what
    // it does with it.
    let ChatFamilySettings {
        effective_mlx_memory_ceiling_bytes: active_memory_limit_bytes,
        mlx_buffer_cache_limit_bytes,
        astro_ssd_prompt_cache_config,
        neural_engine_program_directory,
        performance_attribution_enabled,
        decode_astro_expert_cache_policy,
        performance_attribution_log_path,
        astro_ssd_prompt_cache_enabled,
    } = chat_family_settings;
    let WorkerAutoregressiveModelConfiguration {
        model_id: requested_model_id,
        maximum_context_tokens,
        maximum_output_tokens: max_output_tokens,
        chunking,
    } = model_configuration;
    let (model_loading_performance_attribution, mut performance_attribution_log) =
        open_model_loading_attribution(
            performance_attribution_enabled,
            &performance_attribution_log_path,
        )?;
    let (validated_artifact, model_loading_performance_attribution) = validate_startup_artifact(
        model_loading_performance_attribution,
        &mut performance_attribution_log,
        &model_directory_path,
        max_output_tokens,
    )?;
    let artifact_metadata = ReviewedArtifactMetadata::from_validated_artifact(&validated_artifact);
    let ((selected_engine, engine_placement), model_loading_performance_attribution) =
        select_startup_engine(
            model_loading_performance_attribution,
            &mut performance_attribution_log,
            &validated_artifact,
            &model_directory_path,
            active_memory_limit_bytes,
            &artifact_metadata,
        )?;
    let (generation_processor, model_loading_performance_attribution) =
        initialize_generation_processor(
            model_loading_performance_attribution,
            &mut performance_attribution_log,
            &validated_artifact,
            &model_directory_path,
            requested_model_id,
            maximum_context_tokens,
            performance_attribution_enabled,
            &artifact_metadata,
        )?;
    let think_end_token_id = generation_processor.think_end_token_id();
    let startup_storage = resolve_startup_storage(
        astro_ssd_prompt_cache_enabled,
        &astro_ssd_prompt_cache_config,
        neural_engine_program_directory.as_deref(),
        &validated_artifact,
        &model_directory_path,
        &artifact_metadata,
    );
    let qwen3_5_engine = construct_selected_qwen3_5_engine(
        selected_engine,
        validated_artifact,
        active_memory_limit_bytes,
        mlx_buffer_cache_limit_bytes,
        startup_storage,
        think_end_token_id,
        &model_directory_path,
        maximum_context_tokens,
        chunking,
        model_loading_performance_attribution,
        performance_attribution_log,
        decode_astro_expert_cache_policy,
        &artifact_metadata,
    )?;
    Ok(ChatRuntimeParts {
        generation_processor,
        inference_engine: qwen3_5_engine,
        engine_placement,
    })
}

/// Derives the model's `astro_ssd_prompt_cache` disk store and
/// `astro_neural_engine_program_cache` folder from the worker's settings. The
/// disk store follows the user's setting and the artifact's support for it;
/// the program folder stays `None` when the worker gave no program directory.
fn resolve_startup_storage(
    astro_ssd_prompt_cache_enabled: bool,
    astro_ssd_prompt_cache_config: &AstroSsdPromptCacheConfig,
    neural_engine_program_directory: Option<&std::path::Path>,
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_directory_path: &std::path::Path,
    artifact_metadata: &ReviewedArtifactMetadata,
) -> StartupStorage {
    let astro_ssd_prompt_cache_disk_store_config = astro_ssd_prompt_cache_disk_store_config(
        astro_ssd_prompt_cache_serves_artifact(astro_ssd_prompt_cache_enabled, validated_artifact),
        astro_ssd_prompt_cache_config,
        &artifact_metadata.model_id,
        &artifact_metadata.model_revision,
    );
    let astro_neural_engine_program_cache_directory =
        model_astro_neural_engine_program_cache_directory(
            neural_engine_program_directory,
            model_directory_path,
            &artifact_metadata.model_id,
            &artifact_metadata.model_revision,
        );
    StartupStorage {
        astro_ssd_prompt_cache_disk_store_config,
        astro_neural_engine_program_cache_directory,
    }
}

/// Constructs the engine as the startup selection said to start it, mapping
/// an engine-initialization failure onto the startup error.
#[allow(clippy::too_many_arguments)]
fn construct_selected_qwen3_5_engine(
    streaming_start: EngineStart,
    validated_artifact: ValidatedQwen3_5Artifact,
    active_memory_limit_bytes: usize,
    mlx_buffer_cache_limit_bytes: usize,
    startup_storage: StartupStorage,
    think_end_token_id: u32,
    model_directory_path: &std::path::Path,
    maximum_context_tokens: u32,
    chunking: WorkerChunkingConfiguration,
    model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: PerformanceAttributionLog,
    decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    artifact_metadata: &ReviewedArtifactMetadata,
) -> Result<ModelFamilyInferenceEngine, Qwen3_5ModelStartupError> {
    construct_text_engine(
        prompt_chunk_tokens_for(&chunking, streaming_start.holds_whole_model),
        validated_artifact,
        active_memory_limit_bytes,
        mlx_buffer_cache_limit_bytes,
        startup_storage.astro_ssd_prompt_cache_disk_store_config,
        think_end_token_id,
        model_directory_path,
        maximum_context_tokens,
        chunking,
        model_loading_performance_attribution,
        performance_attribution_log,
        TextEngineSettings {
            decode_astro_expert_cache_policy,
            astro_neural_engine_program_cache_directory: startup_storage
                .astro_neural_engine_program_cache_directory,
        },
        artifact_metadata,
    )
}

/// Validates the model directory's artifact under attribution; a validation
/// failure records the failed loading attribution (with no artifact metadata
/// available yet) and reports the startup error. The attribution returns to
/// the caller on success so the next startup phase owns it.
fn validate_startup_artifact(
    mut model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: &mut PerformanceAttributionLog,
    model_directory_path: &std::path::Path,
    max_output_tokens: u32,
) -> Result<(ValidatedQwen3_5Artifact, PerformanceAttribution), Qwen3_5ModelStartupError> {
    match model_loading_performance_attribution.measure_operation(
        PerformanceOperation::ArtifactValidation,
        |_performance_attribution| {
            Qwen3_5ArtifactValidator::new().validate(model_directory_path, max_output_tokens)
        },
    ) {
        Ok(validated_artifact) => Ok((validated_artifact, model_loading_performance_attribution)),
        Err(source) => {
            tracing::warn!(
                error = %source,
                model_directory = ?model_directory_path,
                "Qwen3.5 artifact validation failed during model initialization"
            );
            record_failed_model_loading_performance_attribution(
                model_loading_performance_attribution,
                performance_attribution_log,
                None,
                "artifact validation failed",
            );
            Err(Qwen3_5ModelStartupError::ArtifactValidation {
                model_directory: model_directory_path.to_path_buf(),
                source,
            })
        }
    }
}

/// Works out the startup placement from the artifact's memory geometry
/// under attribution; a failure records the failed loading attribution with
/// the validated artifact's metadata. The attribution returns to the caller
/// on success so the next startup phase owns it.
fn select_startup_engine(
    mut model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: &mut PerformanceAttributionLog,
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_directory_path: &std::path::Path,
    active_memory_limit_bytes: usize,
    artifact_metadata: &ReviewedArtifactMetadata,
) -> Result<((EngineStart, EnginePlacement), PerformanceAttribution), Qwen3_5ModelStartupError> {
    match model_loading_performance_attribution.measure_operation(
        PerformanceOperation::ArtifactMemoryGeometry,
        |_performance_attribution| {
            select_qwen3_5_engine_from_artifact(
                validated_artifact,
                model_directory_path,
                active_memory_limit_bytes,
            )
        },
    ) {
        Ok(startup_selection) => Ok((startup_selection, model_loading_performance_attribution)),
        Err(selection_error) => {
            tracing::warn!(
                error = %selection_error,
                model_directory = ?model_directory_path,
                "Qwen3.5 engine selection failed during model initialization"
            );
            record_failed_model_loading_performance_attribution(
                model_loading_performance_attribution,
                performance_attribution_log,
                Some(artifact_metadata),
                "Qwen3.5 engine selection failed",
            );
            Err(selection_error)
        }
    }
}

/// Builds the generation processor (tokenizer included) under attribution;
/// a failure records the failed loading attribution with the validated
/// artifact's metadata. The attribution returns to the caller on success.
#[allow(clippy::too_many_arguments)]
fn initialize_generation_processor(
    mut model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: &mut PerformanceAttributionLog,
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_directory_path: &std::path::Path,
    requested_model_id: String,
    maximum_context_tokens: u32,
    performance_attribution_enabled: bool,
    artifact_metadata: &ReviewedArtifactMetadata,
) -> Result<(Qwen3_5GenerationProcessor, PerformanceAttribution), Qwen3_5ModelStartupError> {
    match model_loading_performance_attribution.measure_operation(
        PerformanceOperation::TokenizerInitialization,
        |_performance_attribution| {
            Qwen3_5GenerationProcessor::from_validated_artifact_with_effective_policy(
                validated_artifact,
                requested_model_id,
                maximum_context_tokens,
                true,
                performance_attribution_enabled,
            )
        },
    ) {
        Ok(generation_processor) => {
            Ok((generation_processor, model_loading_performance_attribution))
        }
        Err(source) => {
            record_failed_model_loading_performance_attribution(
                model_loading_performance_attribution,
                performance_attribution_log,
                Some(artifact_metadata),
                "tokenizer initialization failed",
            );
            Err(Qwen3_5ModelStartupError::ProcessorInitialization {
                model_directory: model_directory_path.to_path_buf(),
                source,
            })
        }
    }
}

/// Constructs the engine with its SSD-streaming chunk sizer; a
/// sizing failure records the failed loading attribution first.
#[allow(clippy::too_many_arguments)]
fn construct_text_engine(
    prompt_chunk_tokens: u32,
    validated_artifact: ValidatedQwen3_5Artifact,
    active_memory_limit_bytes: usize,
    mlx_buffer_cache_limit_bytes: usize,
    astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
    think_end_token_id: u32,
    model_directory_path: &std::path::Path,
    maximum_context_tokens: u32,
    chunking: WorkerChunkingConfiguration,
    model_loading_performance_attribution: PerformanceAttribution,
    mut performance_attribution_log: PerformanceAttributionLog,
    text_engine_settings: TextEngineSettings,
    artifact_metadata: &ReviewedArtifactMetadata,
) -> Result<ModelFamilyInferenceEngine, Qwen3_5ModelStartupError> {
    let prompt_processing_chunk_sizer =
        match PromptProcessingChunkSizer::for_chunk_size_tokens(prompt_chunk_tokens) {
            Ok(prompt_processing_chunk_sizer) => prompt_processing_chunk_sizer,
            Err(prompt_processing_chunk_sizer_error) => {
                record_failed_model_loading_performance_attribution(
                    model_loading_performance_attribution,
                    &mut performance_attribution_log,
                    Some(artifact_metadata),
                    "prompt-processing chunk configuration failed",
                );
                return Err(Qwen3_5ModelStartupError::PromptProcessingChunkSizing(
                    prompt_processing_chunk_sizer_error,
                ));
            }
        };
    Qwen3_5TextEngine::new_with_execution_core_settings(
        TextExecutionCoreSettings {
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory: model_directory_path.to_path_buf(),
            maximum_context_tokens,
            chunking,
            model_loading_performance_attribution,
            performance_attribution_log,
        },
        text_engine_settings,
    )
    .map(boxed_chat_engine)
    .map_err(|source| Qwen3_5ModelStartupError::EngineInitialization {
        model_directory: model_directory_path.to_path_buf(),
        source,
    })
}

/// The model-loading attribution metadata of a failed startup. The artifact's
/// identity and size are absent when the failure came before the artifact
/// validated, because there is no validated artifact to take them from.
fn failed_model_loading_metadata(
    reviewed_artifact: Option<&ReviewedArtifactMetadata>,
    failure_description: &'static str,
) -> ModelLoadingPerformanceAttributionMetadata {
    let (model_id, model_revision, total_artifact_payload_bytes, model_shard_count) =
        match reviewed_artifact {
            Some(artifact_metadata) => (
                Some(artifact_metadata.model_id.clone()),
                Some(artifact_metadata.model_revision.clone()),
                Some(artifact_metadata.total_payload_bytes),
                Some(artifact_metadata.shard_count),
            ),
            None => (None, None, None, None),
        };
    ModelLoadingPerformanceAttributionMetadata {
        outcome: PerformanceAttributionOutcome::Failed,
        model_id,
        model_revision,
        prefill_transient_observation_completed: false,
        prefill_observed_transient_high_water_bytes: 0,
        total_artifact_payload_bytes,
        resident_model_payload_bytes: None,
        model_shard_count,
        mlx_active_memory_bytes: None,
        mlx_buffer_cache_bytes: None,
        mlx_peak_memory_bytes: None,
        failure_description: Some(failure_description.to_owned()),
    }
}

/// Records a failed loading attribution; `reviewed_artifact` is `None` while
/// the artifact itself has not validated.
fn record_failed_model_loading_performance_attribution(
    model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: &mut PerformanceAttributionLog,
    reviewed_artifact: Option<&ReviewedArtifactMetadata>,
    failure_description: &'static str,
) {
    let Some(performance_attribution_report) = model_loading_performance_attribution
        .finish_model_loading(failed_model_loading_metadata(
            reviewed_artifact,
            failure_description,
        ))
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

#[cfg(test)]
#[path = "qwen3_5_model_startup_tests.rs"]
mod tests;
