//! The Qwen3.5 autoregressive startup pipeline: validate the model directory's
//! artifacts, derive the RAM geometry and the complete-residency decision,
//! build the tokenizer and processor, and select and initialize the engine.

use std::path::PathBuf;

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    WorkerChunkingConfiguration, WorkerDecodeExpertCachePolicy,
    WorkerDiagnosticQwen3_5EngineOverride,
};
use astronomical_model_serving::{
    CompleteResidencyRequirements, EnginePlacement, ModelFamilyInferenceEngine,
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution, PerformanceAttributionLog,
    PerformanceAttributionOutcome, PerformanceOperation, PersistentPromptCacheDiskStoreConfig,
    Qwen3_5ArtifactValidator, Qwen3_5GenerationProcessor, Qwen3_5RamBudgetGeometryError,
    Qwen3_5ResidentEngine, Qwen3_5ResidentPromptProcessingChunkSizer, Qwen3_5StreamingEngine,
    Qwen3_5StreamingPromptProcessingChunkSizer, ValidatedQwen3_5Artifact,
    mlx_ram_budget_model_geometry_from_validated_artifact,
};

use super::qwen3_5_model_startup_error::Qwen3_5ModelStartupError;

#[derive(Clone, Copy)]
pub(crate) enum Qwen3_5EngineSelection {
    /// Startup selection from the complete-residency fit. A diagnostic
    /// journey may force the engine family; the computed placement is kept
    /// either way so the worker lifecycle stays identical to production.
    Automatic(WorkerDiagnosticQwen3_5EngineOverride),
    StreamingRetry,
    ResidentPromotion,
}

#[derive(Clone, Copy, Debug)]
enum Qwen3_5SelectedEngine {
    Resident,
    Streaming,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn initialize_qwen3_5_model(
    model_directory_path: PathBuf,
    active_memory_limit_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    prompt_cache_config: PromptCacheConfig,
    requested_model_id: String,
    maximum_context_tokens: u32,
    max_output_tokens: u32,
    persistent_prompt_cache_enabled: bool,
    performance_attribution_enabled: bool,
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    performance_attribution_log_path: PathBuf,
    chunking: WorkerChunkingConfiguration,

    engine_selection: Qwen3_5EngineSelection,
    diagnostic_resident_prefill_activation_floor_bytes: Option<u64>,
) -> Result<
    (
        Qwen3_5GenerationProcessor,
        ModelFamilyInferenceEngine,
        Option<EnginePlacement>,
    ),
    Qwen3_5ModelStartupError,
> {
    let model_loading_performance_attribution = if performance_attribution_enabled {
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
    let qwen3_5_engine = construct_selected_qwen3_5_engine(
        selected_engine,
        validated_artifact,
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        persistent_prompt_cache_disk_store_config,
        think_end_token_id,
        &model_directory_path,
        maximum_context_tokens,
        chunking,
        model_loading_performance_attribution,
        performance_attribution_log,
        diagnostic_resident_prefill_activation_floor_bytes,
        decode_expert_cache_policy,
        &artifact_model_id,
        &artifact_model_revision,
        artifact_payload_bytes,
        artifact_shard_count,
    )?;
    Ok((generation_processor, qwen3_5_engine, engine_placement))
}

/// Constructs whichever engine the startup selection chose, mapping an
/// engine-initialization failure onto the startup error.
#[allow(clippy::too_many_arguments)]
fn construct_selected_qwen3_5_engine(
    selected_engine: Qwen3_5SelectedEngine,
    validated_artifact: ValidatedQwen3_5Artifact,
    active_memory_limit_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
    think_end_token_id: u32,
    model_directory_path: &std::path::Path,
    maximum_context_tokens: u32,
    chunking: WorkerChunkingConfiguration,
    model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: PerformanceAttributionLog,
    diagnostic_resident_prefill_activation_floor_bytes: Option<u64>,
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    artifact_model_id: &str,
    artifact_model_revision: &str,
    artifact_payload_bytes: u64,
    artifact_shard_count: usize,
) -> Result<ModelFamilyInferenceEngine, Qwen3_5ModelStartupError> {
    match selected_engine {
        Qwen3_5SelectedEngine::Resident => construct_resident_engine(
            validated_artifact,
            active_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
            persistent_prompt_cache_disk_store_config,
            think_end_token_id,
            model_directory_path,
            maximum_context_tokens,
            chunking,
            model_loading_performance_attribution,
            performance_attribution_log,
            diagnostic_resident_prefill_activation_floor_bytes,
            artifact_model_id,
            artifact_model_revision,
            artifact_payload_bytes,
            artifact_shard_count,
        ),
        Qwen3_5SelectedEngine::Streaming => construct_streaming_engine(
            validated_artifact,
            active_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
            persistent_prompt_cache_disk_store_config,
            think_end_token_id,
            model_directory_path,
            maximum_context_tokens,
            chunking,
            model_loading_performance_attribution,
            performance_attribution_log,
            decode_expert_cache_policy,
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
    engine_selection: Qwen3_5EngineSelection,
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
        Qwen3_5EngineSelection::Automatic(diagnostic_engine_override) => {
            match model_loading_performance_attribution.measure_operation(
                PerformanceOperation::ArtifactRamBudgetGeometry,
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
        Qwen3_5EngineSelection::StreamingRetry => Ok((
            (Qwen3_5SelectedEngine::Streaming, None),
            model_loading_performance_attribution,
        )),
        Qwen3_5EngineSelection::ResidentPromotion => Ok((
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
    allocator_cache_memory_limit_bytes: usize,
    persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
    think_end_token_id: u32,
    model_directory_path: &std::path::Path,
    maximum_context_tokens: u32,
    chunking: WorkerChunkingConfiguration,
    model_loading_performance_attribution: PerformanceAttribution,
    mut performance_attribution_log: PerformanceAttributionLog,
    diagnostic_resident_prefill_activation_floor_bytes: Option<u64>,
    artifact_model_id: &str,
    artifact_model_revision: &str,
    artifact_payload_bytes: u64,
    artifact_shard_count: usize,
) -> Result<ModelFamilyInferenceEngine, Qwen3_5ModelStartupError> {
    let prompt_processing_chunk_sizer =
        match Qwen3_5ResidentPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
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
                return Err(Qwen3_5ModelStartupError::ResidentPromptProcessingChunkSizing(
                    prompt_processing_chunk_sizer_error,
                ));
            }
        };
    // DFlash speculative decoding is disabled by design: no draft
    // is ever constructed for the resident engine (see the
    // speculative-decode lifecycle module for the full rationale).
    Qwen3_5ResidentEngine::new_with_effective_context_runtime_chunking_and_performance_attribution(
        validated_artifact,
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        persistent_prompt_cache_disk_store_config,
        prompt_processing_chunk_sizer,
        think_end_token_id,
        model_directory_path.to_path_buf(),
        maximum_context_tokens,
        chunking,
        true,
        model_loading_performance_attribution,
        performance_attribution_log,
        diagnostic_resident_prefill_activation_floor_bytes,
    )
    .map(ModelFamilyInferenceEngine::Qwen3_5)
    .map_err(|source| Qwen3_5ModelStartupError::EngineInitialization {
        model_directory: model_directory_path.to_path_buf(),
        source,
    })
}

/// Constructs the streaming engine with its SSD-streaming chunk sizer; a
/// sizing failure records the failed loading attribution first.
#[allow(clippy::too_many_arguments)]
fn construct_streaming_engine(
    validated_artifact: ValidatedQwen3_5Artifact,
    active_memory_limit_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
    think_end_token_id: u32,
    model_directory_path: &std::path::Path,
    maximum_context_tokens: u32,
    chunking: WorkerChunkingConfiguration,
    model_loading_performance_attribution: PerformanceAttribution,
    mut performance_attribution_log: PerformanceAttributionLog,
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    artifact_model_id: &str,
    artifact_model_revision: &str,
    artifact_payload_bytes: u64,
    artifact_shard_count: usize,
) -> Result<ModelFamilyInferenceEngine, Qwen3_5ModelStartupError> {
    let prompt_processing_chunk_sizer =
        match Qwen3_5StreamingPromptProcessingChunkSizer::for_ssd_streaming_chunk_size_tokens(
            chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
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
                    Qwen3_5ModelStartupError::StreamingPromptProcessingChunkSizing(
                        prompt_processing_chunk_sizer_error,
                    ),
                );
            }
        };
    Qwen3_5StreamingEngine::new_with_effective_context_runtime_chunking_performance_attribution_and_decode_cache_policy(
        validated_artifact,
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        persistent_prompt_cache_disk_store_config,
        prompt_processing_chunk_sizer,
        think_end_token_id,
        model_directory_path.to_path_buf(),
        maximum_context_tokens,
        chunking,
        true,
        model_loading_performance_attribution,
        performance_attribution_log,
        decode_expert_cache_policy,
    )
    .map(ModelFamilyInferenceEngine::Qwen3_5Streaming)
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
        WorkerDiagnosticQwen3_5EngineOverride::Streaming => Qwen3_5SelectedEngine::Streaming,
    };
    tracing::warn!(
        automatically_selected_engine = ?automatically_selected_engine,
        forced_engine = ?forced_engine,
        "diagnostic Qwen3.5 engine override applied at startup"
    );
    forced_engine
}

fn select_qwen3_5_engine_from_artifact(
    validated_artifact: &astronomical_model_serving::ValidatedQwen3_5Artifact,
    model_directory_path: &std::path::Path,
    active_memory_limit_bytes: usize,
) -> Result<(Qwen3_5SelectedEngine, EnginePlacement), Qwen3_5ModelStartupError> {
    let (model_geometry, required_headroom_bytes) =
        match mlx_ram_budget_model_geometry_from_validated_artifact(
            validated_artifact,
            model_directory_path,
        ) {
            Ok(geometry) => geometry,
            Err(Qwen3_5RamBudgetGeometryError::NotSparseMixtureOfExperts) => {
                return Ok((
                    Qwen3_5SelectedEngine::Resident,
                    EnginePlacement::CompleteResidencyFits,
                ));
            }
            Err(source) => {
                return Err(Qwen3_5ModelStartupError::ArtifactRamGeometry {
                    model_directory: model_directory_path.to_path_buf(),
                    source,
                });
            }
        };
    let complete_residency_decision = CompleteResidencyRequirements {
        current_active_memory_bytes: model_geometry.model_core_payload_bytes,
        retained_paged_expert_payload_bytes: 0,
        complete_expert_payload_bytes: model_geometry.complete_expert_payload_bytes,
        required_headroom_bytes,
        active_memory_ceiling_bytes: u64::try_from(active_memory_limit_bytes).unwrap_or(u64::MAX),
    }
    .decide();
    tracing::info!(
        model_core_payload_bytes = model_geometry.model_core_payload_bytes,
        complete_expert_payload_bytes = model_geometry.complete_expert_payload_bytes,
        largest_complete_expert_layer_bytes = model_geometry.largest_complete_expert_layer_bytes,
        required_headroom_bytes,
        active_memory_limit_bytes,
        complete_residency_decision = ?complete_residency_decision,
        "Qwen3.5 startup complete-residency decision"
    );
    let engine_placement =
        EnginePlacement::from_startup_complete_residency_decision(complete_residency_decision)
            .map_err(
                |source| Qwen3_5ModelStartupError::CompleteResidencyDecision {
                    model_directory: model_directory_path.to_path_buf(),
                    source,
                },
            )?;
    let selected_engine = if engine_placement.runs_resident_engine() {
        Qwen3_5SelectedEngine::Resident
    } else {
        Qwen3_5SelectedEngine::Streaming
    };
    Ok((selected_engine, engine_placement))
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
