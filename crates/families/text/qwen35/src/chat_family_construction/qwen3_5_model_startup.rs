//! The Qwen3.5 autoregressive startup pipeline: validate the model directory's
//! artifacts, derive the RAM geometry and the complete-residency decision,
//! build the tokenizer and processor, and select and initialize the engine.

use crate::TextMlxEngineConstructors;
use std::path::PathBuf;

use crate::{
    AstroSsdPromptCacheDiskStoreConfig, EnginePlacement, ModelFamilyInferenceEngine,
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution, PerformanceAttributionLog,
    PerformanceAttributionOutcome, PerformanceOperation, PromptProcessingChunkSizer,
    Qwen3_5ArtifactValidator, Qwen3_5GenerationProcessor, Qwen3_5ResidentEngine,
    Qwen3_5StreamingEngine, ResidentEngineSettings, StreamingEngineSettings,
    TextExecutionCoreSettings, ValidatedQwen3_5Artifact, boxed_chat_engine,
};
use ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerDecodeAstroExpertCachePolicy, WorkerDiagnosticQwen3_5EngineOverride,
};

use super::qwen3_5_engine_selection::{
    Qwen3_5SelectedEngine, prompt_chunk_tokens_for, select_qwen3_5_engine_from_artifact,
};
use super::qwen3_5_model_startup_error::Qwen3_5ModelStartupError;
use super::qwen3_5_startup_storage::{
    astro_ssd_prompt_cache_disk_store_config, astro_ssd_prompt_cache_serves_artifact,
    model_astro_neural_engine_program_cache_directory, open_model_loading_attribution,
};
use super::{ChatEngineSelection, ChatFamilySettings, ChatRuntimeParts};

/// Validates the folder, selects and starts the engine, and builds the
/// processor, from the worker's chat settings and the model's policy.
pub(super) fn initialize_qwen3_5_model(
    model_directory_path: PathBuf,
    chat_family_settings: ChatFamilySettings,
    model_configuration: WorkerAutoregressiveModelConfiguration,
    engine_selection: ChatEngineSelection,
) -> Result<ChatRuntimeParts, Qwen3_5ModelStartupError> {
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
    let artifact_model_id = validated_artifact.model_id().to_owned();
    let artifact_model_revision = validated_artifact.revision().to_owned();
    let artifact_payload_bytes = validated_artifact.total_payload_bytes();
    let artifact_shard_count = validated_artifact.shard_count();
    let ((selected_engine, engine_placement), model_loading_performance_attribution) =
        select_startup_engine(
            model_loading_performance_attribution,
            &mut performance_attribution_log,
            &validated_artifact,
            &model_directory_path,
            active_memory_limit_bytes,
            engine_selection,
            &artifact_model_id,
            &artifact_model_revision,
            artifact_payload_bytes,
            artifact_shard_count,
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
            &artifact_model_id,
            &artifact_model_revision,
            artifact_payload_bytes,
            artifact_shard_count,
        )?;
    let think_end_token_id = generation_processor.think_end_token_id();
    let model_id = validated_artifact.model_id().to_owned();
    let model_revision = validated_artifact.revision().to_owned();
    let astro_ssd_prompt_cache_disk_store_config = astro_ssd_prompt_cache_disk_store_config(
        astro_ssd_prompt_cache_serves_artifact(astro_ssd_prompt_cache_enabled, &validated_artifact),
        &astro_ssd_prompt_cache_config,
        &model_id,
        &model_revision,
    );
    let astro_neural_engine_program_cache_directory =
        model_astro_neural_engine_program_cache_directory(
            neural_engine_program_directory.as_deref(),
            &model_directory_path,
            &model_id,
            &model_revision,
        );
    let qwen3_5_engine = construct_selected_qwen3_5_engine(
        selected_engine,
        validated_artifact,
        active_memory_limit_bytes,
        mlx_buffer_cache_limit_bytes,
        astro_ssd_prompt_cache_disk_store_config,
        astro_neural_engine_program_cache_directory,
        think_end_token_id,
        &model_directory_path,
        maximum_context_tokens,
        chunking,
        model_loading_performance_attribution,
        performance_attribution_log,
        decode_astro_expert_cache_policy,
        &artifact_model_id,
        &artifact_model_revision,
        artifact_payload_bytes,
        artifact_shard_count,
    )?;
    Ok(ChatRuntimeParts {
        generation_processor,
        inference_engine: qwen3_5_engine,
        engine_placement,
    })
}

/// Constructs whichever engine the startup selection chose, mapping an
/// engine-initialization failure onto the startup error.
#[allow(clippy::too_many_arguments)]
fn construct_selected_qwen3_5_engine(
    selected_engine: Qwen3_5SelectedEngine,
    validated_artifact: ValidatedQwen3_5Artifact,
    active_memory_limit_bytes: usize,
    mlx_buffer_cache_limit_bytes: usize,
    astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
    astro_neural_engine_program_cache_directory: Option<PathBuf>,
    think_end_token_id: u32,
    model_directory_path: &std::path::Path,
    maximum_context_tokens: u32,
    chunking: WorkerChunkingConfiguration,
    model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: PerformanceAttributionLog,
    decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    artifact_model_id: &str,
    artifact_model_revision: &str,
    artifact_payload_bytes: u64,
    artifact_shard_count: usize,
) -> Result<ModelFamilyInferenceEngine, Qwen3_5ModelStartupError> {
    match selected_engine {
        Qwen3_5SelectedEngine::Resident => construct_resident_engine(
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            astro_neural_engine_program_cache_directory,
            think_end_token_id,
            model_directory_path,
            maximum_context_tokens,
            chunking,
            model_loading_performance_attribution,
            performance_attribution_log,
            artifact_model_id,
            artifact_model_revision,
            artifact_payload_bytes,
            artifact_shard_count,
        ),
        Qwen3_5SelectedEngine::Streaming { holds_whole_model } => construct_streaming_engine(
            prompt_chunk_tokens_for(&chunking, holds_whole_model),
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            think_end_token_id,
            model_directory_path,
            maximum_context_tokens,
            chunking,
            model_loading_performance_attribution,
            performance_attribution_log,
            StreamingEngineSettings {
                decode_astro_expert_cache_policy,
                astro_neural_engine_program_cache_directory,
            },
            artifact_model_id,
            artifact_model_revision,
            artifact_payload_bytes,
            artifact_shard_count,
        ),
    }
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
                None,
                None,
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

/// Selects the startup engine: the computed complete-residency fit with any
/// diagnostic override, or a worker-directed forced selection. Forced
/// selections exist for one worker-directed swap; no startup selection runs,
/// so they carry no placement — the worker records the typed transition
/// (request demotion or resident promotion) itself. The attribution returns
/// to the caller on both outcomes.
#[allow(clippy::too_many_arguments)]
fn select_startup_engine(
    mut model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: &mut PerformanceAttributionLog,
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_directory_path: &std::path::Path,
    active_memory_limit_bytes: usize,
    engine_selection: ChatEngineSelection,
    artifact_model_id: &str,
    artifact_model_revision: &str,
    artifact_payload_bytes: u64,
    artifact_shard_count: usize,
) -> Result<
    (
        (Qwen3_5SelectedEngine, Option<EnginePlacement>),
        PerformanceAttribution,
    ),
    Qwen3_5ModelStartupError,
> {
    match engine_selection {
        ChatEngineSelection::Automatic(diagnostic_engine_override) => {
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
                Ok((automatically_selected_engine, engine_placement)) => {
                    let selected_engine = apply_diagnostic_engine_override(
                        automatically_selected_engine,
                        diagnostic_engine_override,
                    );
                    Ok((
                        (selected_engine, Some(engine_placement)),
                        model_loading_performance_attribution,
                    ))
                }
                Err(selection_error) => {
                    tracing::warn!(
                        error = %selection_error,
                        model_directory = ?model_directory_path,
                        "Qwen3.5 engine selection failed during model initialization"
                    );
                    record_reviewed_artifact_startup_failure(
                        model_loading_performance_attribution,
                        performance_attribution_log,
                        artifact_model_id,
                        artifact_model_revision,
                        artifact_payload_bytes,
                        artifact_shard_count,
                        "Qwen3.5 engine selection failed",
                    );
                    Err(selection_error)
                }
            }
        }
        ChatEngineSelection::StreamingRetry => Ok((
            (
                Qwen3_5SelectedEngine::Streaming {
                    holds_whole_model: false,
                },
                None,
            ),
            model_loading_performance_attribution,
        )),
        ChatEngineSelection::ResidentPromotion => Ok((
            (Qwen3_5SelectedEngine::Resident, None),
            model_loading_performance_attribution,
        )),
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
    artifact_model_id: &str,
    artifact_model_revision: &str,
    artifact_payload_bytes: u64,
    artifact_shard_count: usize,
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
            record_reviewed_artifact_startup_failure(
                model_loading_performance_attribution,
                performance_attribution_log,
                artifact_model_id,
                artifact_model_revision,
                artifact_payload_bytes,
                artifact_shard_count,
                "tokenizer initialization failed",
            );
            Err(Qwen3_5ModelStartupError::ProcessorInitialization {
                model_directory: model_directory_path.to_path_buf(),
                source,
            })
        }
    }
}

/// Constructs the resident engine with its fixed-prompt chunk sizer; a
/// sizing failure records the failed loading attribution first.
#[allow(clippy::too_many_arguments)]
fn construct_resident_engine(
    validated_artifact: ValidatedQwen3_5Artifact,
    active_memory_limit_bytes: usize,
    mlx_buffer_cache_limit_bytes: usize,
    astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
    astro_neural_engine_program_cache_directory: Option<PathBuf>,
    think_end_token_id: u32,
    model_directory_path: &std::path::Path,
    maximum_context_tokens: u32,
    chunking: WorkerChunkingConfiguration,
    model_loading_performance_attribution: PerformanceAttribution,
    mut performance_attribution_log: PerformanceAttributionLog,
    artifact_model_id: &str,
    artifact_model_revision: &str,
    artifact_payload_bytes: u64,
    artifact_shard_count: usize,
) -> Result<ModelFamilyInferenceEngine, Qwen3_5ModelStartupError> {
    let prompt_processing_chunk_sizer = match PromptProcessingChunkSizer::for_chunk_size_tokens(
        chunking.fixed_prompt_processing_chunk_size_tokens,
    ) {
        Ok(prompt_processing_chunk_sizer) => prompt_processing_chunk_sizer,
        Err(prompt_processing_chunk_sizer_error) => {
            record_reviewed_artifact_startup_failure(
                model_loading_performance_attribution,
                &mut performance_attribution_log,
                artifact_model_id,
                artifact_model_revision,
                artifact_payload_bytes,
                artifact_shard_count,
                "prompt-processing chunk configuration failed",
            );
            return Err(
                Qwen3_5ModelStartupError::ResidentPromptProcessingChunkSizing(
                    prompt_processing_chunk_sizer_error,
                ),
            );
        }
    };
    // DFlash speculative decoding is disabled by design: no draft
    // is ever constructed for the resident engine (see the
    // speculative-decode lifecycle module for the full rationale).
    Qwen3_5ResidentEngine::new_with_execution_core_settings(
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
        ResidentEngineSettings {
            astro_neural_engine_program_cache_directory,
        },
    )
    .map(boxed_chat_engine)
    .map_err(|source| Qwen3_5ModelStartupError::EngineInitialization {
        model_directory: model_directory_path.to_path_buf(),
        source,
    })
}

/// Constructs the streaming engine with its SSD-streaming chunk sizer; a
/// sizing failure records the failed loading attribution first.
#[allow(clippy::too_many_arguments)]
fn construct_streaming_engine(
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
    streaming_engine_settings: StreamingEngineSettings,
    artifact_model_id: &str,
    artifact_model_revision: &str,
    artifact_payload_bytes: u64,
    artifact_shard_count: usize,
) -> Result<ModelFamilyInferenceEngine, Qwen3_5ModelStartupError> {
    let prompt_processing_chunk_sizer =
        match PromptProcessingChunkSizer::for_chunk_size_tokens(prompt_chunk_tokens) {
            Ok(prompt_processing_chunk_sizer) => prompt_processing_chunk_sizer,
            Err(prompt_processing_chunk_sizer_error) => {
                record_reviewed_artifact_startup_failure(
                    model_loading_performance_attribution,
                    &mut performance_attribution_log,
                    artifact_model_id,
                    artifact_model_revision,
                    artifact_payload_bytes,
                    artifact_shard_count,
                    "prompt-processing chunk configuration failed",
                );
                return Err(
                    Qwen3_5ModelStartupError::StreamingPromptProcessingChunkSizing(
                        prompt_processing_chunk_sizer_error,
                    ),
                );
            }
        };
    Qwen3_5StreamingEngine::new_with_execution_core_settings(
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
        streaming_engine_settings,
    )
    .map(boxed_chat_engine)
    .map_err(|source| Qwen3_5ModelStartupError::EngineInitialization {
        model_directory: model_directory_path.to_path_buf(),
        source,
    })
}

fn apply_diagnostic_engine_override(
    automatically_selected_engine: Qwen3_5SelectedEngine,
    diagnostic_engine_override: WorkerDiagnosticQwen3_5EngineOverride,
) -> Qwen3_5SelectedEngine {
    let forced_engine = match diagnostic_engine_override {
        WorkerDiagnosticQwen3_5EngineOverride::Automatic => return automatically_selected_engine,
        WorkerDiagnosticQwen3_5EngineOverride::Resident => Qwen3_5SelectedEngine::Resident,
        WorkerDiagnosticQwen3_5EngineOverride::Streaming => Qwen3_5SelectedEngine::Streaming {
            holds_whole_model: false,
        },
    };
    tracing::warn!(
        automatically_selected_engine = ?automatically_selected_engine,
        forced_engine = ?forced_engine,
        "diagnostic Qwen3.5 engine override applied at startup"
    );
    forced_engine
}

/// Records a failed loading attribution whose artifact metadata was already
/// captured from the validated artifact.
#[allow(clippy::too_many_arguments)]
fn record_reviewed_artifact_startup_failure(
    model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: &mut PerformanceAttributionLog,
    artifact_model_id: &str,
    artifact_model_revision: &str,
    artifact_payload_bytes: u64,
    artifact_shard_count: usize,
    failure_description: &'static str,
) {
    record_failed_model_loading_performance_attribution(
        model_loading_performance_attribution,
        performance_attribution_log,
        Some(artifact_model_id.to_owned()),
        Some(artifact_model_revision.to_owned()),
        Some(artifact_payload_bytes),
        Some(artifact_shard_count),
        failure_description,
    );
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
            mlx_buffer_cache_bytes: None,
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
