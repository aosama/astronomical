//! Laguna-owned startup: validate once, then load weights on the MLX owner thread.

mod error;
pub(in crate::laguna) mod weight_loader;

use std::path::{Path, PathBuf};

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    ExpertMemoryMode, WorkerChunkingConfiguration, WorkerPromptProcessingChunkSizingPolicy,
};

use crate::PersistentPromptCacheDiskStoreConfig;
use crate::laguna::artifacts::{
    LagunaCanonicalTensorAssemblyKind, LagunaLayerTensorRole, LagunaTensorId,
};
use crate::laguna::engine::execution::LagunaPendingStartup;
use crate::laguna::engine::{LagunaEngine, LagunaInferenceExecution};
use crate::laguna::{
    LagunaArtifactValidator, LagunaExpertPagingPlan, LagunaGenerationProcessor,
    LagunaPromptProcessingChunkSizer, LagunaPromptProcessingExecutionProfile, LagunaTensorContract,
};
use crate::{
    MlxInferenceEngine, MlxRamBudget, MlxRamBudgetModelGeometry, PerformanceAttribution,
    PerformanceAttributionLog, PerformanceOperation,
};

pub use error::LagunaStartupError;

/// Optional serving policy supplied by the worker factory.
pub struct LagunaServingSettings {
    pub chunking: Option<WorkerChunkingConfiguration>,
    pub optimizer_state_directory: Option<PathBuf>,
    pub persistent_prompt_cache_enabled: bool,
    pub prompt_cache_config: Option<PromptCacheConfig>,
    pub performance_attribution_log_path: Option<PathBuf>,
}

impl LagunaServingSettings {
    /// Same-process tests keep one large fixed chunk so existing journeys stay one-shot.
    #[must_use]
    pub fn default_fixed() -> Self {
        Self {
            chunking: None,
            optimizer_state_directory: None,
            persistent_prompt_cache_enabled: false,
            prompt_cache_config: None,
            performance_attribution_log_path: None,
        }
    }
}

/// Validates a Laguna directory and constructs the family processor and engine.
pub fn initialize_laguna_model(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
) -> Result<(LagunaGenerationProcessor, LagunaEngine), LagunaStartupError> {
    initialize_laguna_model_with_serving_settings(
        model_directory,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        performance_attribution_enabled,
        LagunaServingSettings::default_fixed(),
    )
}

/// Validates a Laguna directory with worker-owned chunking and cache policy.
pub fn initialize_laguna_model_with_serving_settings(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
    serving_settings: LagunaServingSettings,
) -> Result<(LagunaGenerationProcessor, LagunaEngine), LagunaStartupError> {
    let (generation_processor, pending_startup) = prepare_laguna_startup(
        model_directory,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        performance_attribution_enabled,
        serving_settings,
    )?;
    let engine =
        MlxInferenceEngine::new(move || LagunaInferenceExecution::pending(pending_startup))
            .map_err(LagunaStartupError::EngineOwner)?;
    Ok((generation_processor, engine))
}

/// Constructs Laguna execution on the caller thread for same-process tests.
pub fn initialize_laguna_execution(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
) -> Result<(LagunaGenerationProcessor, LagunaInferenceExecution), LagunaStartupError> {
    initialize_laguna_execution_with_serving_settings(
        model_directory,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        performance_attribution_enabled,
        LagunaServingSettings::default_fixed(),
    )
}

/// Constructs same-thread Laguna execution with worker-owned serving policy.
pub fn initialize_laguna_execution_with_serving_settings(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
    serving_settings: LagunaServingSettings,
) -> Result<(LagunaGenerationProcessor, LagunaInferenceExecution), LagunaStartupError> {
    let (generation_processor, pending_startup) = prepare_laguna_startup(
        model_directory,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        performance_attribution_enabled,
        serving_settings,
    )?;
    Ok((
        generation_processor,
        LagunaInferenceExecution::pending(pending_startup),
    ))
}

fn prepare_laguna_startup(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
    serving_settings: LagunaServingSettings,
) -> Result<(LagunaGenerationProcessor, LagunaPendingStartup), LagunaStartupError> {
    // Every persisted identity must use provenance discovered from the immutable
    // artifact snapshot, never a process-local placeholder.
    let classified_model_revision =
        astronomical_config::discover_classified_model_artifacts(&[model_directory.to_path_buf()])
            .map_err(|_| LagunaStartupError::ImmutableRevisionRequired)?
            .into_iter()
            .find(|artifact| artifact.model_directory == model_directory)
            .and_then(|artifact| artifact.upstream_revision);
    let mut performance_attribution = if performance_attribution_enabled {
        PerformanceAttribution::enabled()
    } else {
        PerformanceAttribution::disabled()
    };
    let validated_artifact = performance_attribution
        .measure_operation(PerformanceOperation::ArtifactValidation, |_| {
            LagunaArtifactValidator::new().validate(model_directory)
        })
        .map_err(LagunaStartupError::ArtifactValidation)?;
    let model_revision = match classified_model_revision {
        Some(revision)
            if revision.len() == 40
                && revision
                    .bytes()
                    .all(|character| character.is_ascii_hexdigit()) =>
        {
            revision
        }
        Some(_) => return Err(LagunaStartupError::ImmutableRevisionRequired),
        None => validated_artifact
            .storage_fingerprint()
            .iter()
            .map(|fingerprint_byte| format!("{fingerprint_byte:02x}"))
            .collect(),
    };
    let model_id = astronomical_config::requestable_model_id(model_directory)
        .unwrap_or_else(|| "laguna".to_owned());
    let generation_processor = performance_attribution
        .measure_operation(PerformanceOperation::TokenizerInitialization, |_| {
            LagunaGenerationProcessor::new_with_performance_attribution(
                model_id.clone(),
                validated_artifact.text_artifact().clone(),
                performance_attribution_enabled,
            )
        })
        .map_err(LagunaStartupError::ProcessorInitialization)?;

    let paging_plan = LagunaExpertPagingPlan::from_validated_artifact(
        &validated_artifact,
        model_directory,
        &mut performance_attribution,
    )
    .map_err(|_| LagunaStartupError::PagingPlan)?;
    let mut complete_expert_payload_bytes = 0_u64;
    let mut largest_complete_expert_layer_bytes = 0_u64;
    let mut largest_routed_expert_page_bytes = 0_u64;
    for sparse_layer in paging_plan.sparse_layers() {
        let complete_layer_payload_bytes = sparse_layer
            .complete_layer_payload_byte_count()
            .map_err(|_| LagunaStartupError::PagingPlan)?;
        let routed_page_payload_bytes = sparse_layer
            .routed_page_payload_byte_count()
            .map_err(|_| LagunaStartupError::PagingPlan)?;
        complete_expert_payload_bytes = complete_expert_payload_bytes
            .checked_add(complete_layer_payload_bytes)
            .ok_or(LagunaStartupError::PagingPlan)?;
        largest_complete_expert_layer_bytes =
            largest_complete_expert_layer_bytes.max(complete_layer_payload_bytes);
        largest_routed_expert_page_bytes =
            largest_routed_expert_page_bytes.max(routed_page_payload_bytes);
    }
    let ceiling_bytes = u64::try_from(effective_mlx_memory_ceiling_bytes).unwrap_or(u64::MAX);
    // The validated artifact is the only model-specific source of truth. Expert
    // retention receives only the remainder after the packed non-expert core.
    let model_core_payload_bytes = validated_artifact
        .total_tensor_payload_bytes()
        .saturating_sub(complete_expert_payload_bytes);
    let mlx_ram_budget = MlxRamBudget::new(
        ceiling_bytes.max(1),
        MlxRamBudgetModelGeometry {
            model_core_payload_bytes,
            complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes,
            largest_routed_expert_page_bytes,
        },
    )
    .map_err(|_| LagunaStartupError::RuntimeInitialization)?;
    // Complete residency must charge the packed artifact plus the 10% activation
    // floor. Passing zero here previously treated a 35 GB model as free.
    let projected_resident_active_memory_bytes = validated_artifact
        .total_tensor_payload_bytes()
        .max(complete_expert_payload_bytes);
    let can_bind_routed_experts =
        routed_experts_use_direct_or_stacked_assembly(validated_artifact.tensor_contract());
    let activation_headroom_bytes = complete_expert_payload_bytes / 10;
    let mandatory_page_and_transient_reserve_bytes = largest_complete_expert_layer_bytes
        .saturating_mul(2)
        .saturating_add(largest_routed_expert_page_bytes)
        .saturating_add(activation_headroom_bytes);
    let minimum_mlx_memory_ceiling_bytes = model_core_payload_bytes
        .saturating_add(mandatory_page_and_transient_reserve_bytes)
        .max(1);
    let remaining_after_complete_residency_bytes = ceiling_bytes
        .saturating_sub(projected_resident_active_memory_bytes)
        .saturating_sub(activation_headroom_bytes);
    // Leave one eighth of the machine ceiling for the first prompt graph. A
    // packed-artifact projection undercounts dequantized embeddings and MLX
    // workspace, so a 10% expert floor alone still fills a 40 GB S load.
    let first_request_slack_bytes = ceiling_bytes / 8;
    let load_routed_experts = can_bind_routed_experts
        && (paging_plan.sparse_layers().is_empty()
            || (paging_plan.complete_residency_fits_ceiling(
                projected_resident_active_memory_bytes,
                ceiling_bytes,
                complete_expert_payload_bytes,
                0,
            ) && remaining_after_complete_residency_bytes >= first_request_slack_bytes));
    let expert_memory_mode = if load_routed_experts {
        ExpertMemoryMode::Resident
    } else {
        ExpertMemoryMode::Paged
    };
    let prompt_processing_execution_profile =
        LagunaPromptProcessingExecutionProfile::from_canonical_descriptors(
            validated_artifact.target_contract(),
            validated_artifact.storage_fingerprint(),
            expert_memory_mode,
            serving_settings.persistent_prompt_cache_enabled,
        );
    let prompt_processing_chunk_sizer = build_prompt_processing_chunk_sizer(
        &serving_settings,
        &model_id,
        &model_revision,
        validated_artifact
            .target_contract()
            .model()
            .maximum_position_count(),
    )?;
    let prompt_cache_model_revision = model_revision.clone();
    let prompt_cache_disk_store_config = serving_settings
        .prompt_cache_config
        .as_ref()
        .filter(|_| serving_settings.persistent_prompt_cache_enabled)
        .map(|prompt_cache_config| {
            let per_model_prompt_cache_config =
                prompt_cache_config.for_model(&model_id, &prompt_cache_model_revision);
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
    let performance_attribution_log =
        match serving_settings.performance_attribution_log_path.as_deref() {
            Some(log_path) => {
                PerformanceAttributionLog::open(log_path, performance_attribution_enabled)
                    .map_err(LagunaStartupError::PerformanceAttributionLog)?
            }
            None => PerformanceAttributionLog::disabled(),
        };
    let total_artifact_payload_bytes = validated_artifact.total_tensor_payload_bytes();
    let model_shard_count = validated_artifact.shard_index().shard_file_names().count();
    let pending_startup = LagunaPendingStartup {
        target_contract: validated_artifact.target_contract().clone(),
        tensor_contract: validated_artifact.tensor_contract().clone(),
        shard_files: validated_artifact
            .into_retained_files()
            .map_err(|_| LagunaStartupError::RuntimeInitialization)?
            .into_shard_files(),
        paging_plan,
        load_routed_experts,
        mlx_ram_budget,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        prompt_processing_chunk_sizer,
        prompt_processing_execution_profile,
        minimum_mlx_memory_ceiling_bytes,
        prompt_cache_disk_store_config,
        prompt_cache_model_id: model_id.clone(),
        prompt_cache_model_revision,
        configured_prompt_cache_block_token_count,
        prompt_cache_common_prefix_stride_blocks,
        model_loading_performance_attribution: performance_attribution,
        performance_attribution_log,
        attribution_model_id: model_id.clone(),
        attribution_model_revision: model_revision.clone(),
        total_artifact_payload_bytes,
        model_shard_count,
    };
    Ok((generation_processor, pending_startup))
}

fn build_prompt_processing_chunk_sizer(
    serving_settings: &LagunaServingSettings,
    model_id: &str,
    model_revision: &str,
    maximum_position_count: u32,
) -> Result<LagunaPromptProcessingChunkSizer, LagunaStartupError> {
    match serving_settings.chunking.as_ref().map(|chunking| {
        (
            &chunking.prompt_processing_chunk_sizing_policy,
            chunking
                .prompt_processing_chunk_size_optimizer_maximum_retained_measurements_per_candidate_and_context,
            chunking.prompt_processing_chunk_size_optimizer_position_range_size_tokens,
        )
    }) {
        Some((
            WorkerPromptProcessingChunkSizingPolicy::Optimized {
                prompt_processing_chunk_size_optimizer_candidate_token_counts,
            },
            maximum_retained_measurements,
            position_range_size_tokens,
        )) => {
            if let Some(optimizer_state_directory) =
                serving_settings.optimizer_state_directory.clone()
            {
                LagunaPromptProcessingChunkSizer::for_optimized_production_with_persisted_state(
                    maximum_position_count,
                    prompt_processing_chunk_size_optimizer_candidate_token_counts.clone(),
                    optimizer_state_directory,
                    model_id.to_owned(),
                    model_revision.to_owned(),
                    maximum_retained_measurements,
                    position_range_size_tokens,
                )
            } else {
                LagunaPromptProcessingChunkSizer::for_optimized_with_behavior(
                    maximum_position_count,
                    prompt_processing_chunk_size_optimizer_candidate_token_counts.clone(),
                    maximum_retained_measurements,
                    position_range_size_tokens,
                )
            }
        }
        Some((
            WorkerPromptProcessingChunkSizingPolicy::Fixed {
                fixed_prompt_processing_chunk_size_tokens,
                ..
            },
            _,
            _,
        )) => LagunaPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
            *fixed_prompt_processing_chunk_size_tokens,
        ),
        None => LagunaPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
            maximum_position_count.max(1),
        ),
    }
    .map_err(LagunaStartupError::ChunkSizer)
}

fn routed_experts_use_direct_or_stacked_assembly(tensor_contract: &LagunaTensorContract) -> bool {
    tensor_contract.descriptors().values().all(|descriptor| {
        !matches!(
            descriptor.tensor_id(),
            LagunaTensorId::Layer {
                role: LagunaLayerTensorRole::RoutedExpert(_),
                ..
            }
        ) || matches!(
            descriptor.assembly_kind(),
            LagunaCanonicalTensorAssemblyKind::DirectAlias
                | LagunaCanonicalTensorAssemblyKind::StackedSource
        )
    })
}
