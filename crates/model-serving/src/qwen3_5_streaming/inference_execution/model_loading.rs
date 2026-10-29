use astronomical_ipc_protocol::{WorkerChunkingConfiguration, WorkerDecodeExpertCachePolicy};
use astronomical_runtime_integration::MlxRuntime;
use std::sync::Arc;

use crate::inference_engine::load_result::EngineLoadResult;
use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceAttributionOutcome,
    PerformanceOperation, PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheModelContract, PersistentVisualEmbeddingModelContract,
    StreamingStartupResidencyFacts,
};

use super::persistent_prompt_cache_startup_logging;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::model::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
use crate::qwen3_5_core::vision::{Qwen3_5ImageProcessor, Qwen3_5VisionConfig};
use crate::qwen3_5_streaming::model::Qwen3_5Model;

impl Qwen3_5EngineState {
    pub(super) fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        if let Some(model) = self.model.as_ref() {
            return Ok(self.engine_load_result(model.minimum_mlx_memory_ceiling_bytes()?));
        }
        let mut model_loading_performance_attribution = self
            .model_loading_performance_attribution
            .take()
            .unwrap_or_else(PerformanceAttribution::disabled);
        let mut model_id = None;
        let mut model_revision = None;
        let mut total_artifact_payload_bytes = None;
        let mut model_shard_count = None;
        let model_loading_result: Result<_, InferenceEngineError> = (|| {
            let validated_artifact = self.validated_artifact.take().ok_or_else(|| {
                fatal_engine_error("validated Qwen3.5 artifact is unavailable during MLX load")
            })?;
            let qwen3_5_vision_config = validated_artifact.vision_config().cloned();
            model_id = Some(validated_artifact.model_id().to_owned());
            model_revision = Some(validated_artifact.revision().to_owned());
            total_artifact_payload_bytes = Some(validated_artifact.total_payload_bytes());
            model_shard_count = Some(validated_artifact.shard_count());
            // Capture the startup engine-selection facts before the artifact
            // moves into the model: a later live raise consults them to
            // decide whether the raised ceiling would select the resident
            // engine in a replacement worker. Derived from the same artifact
            // and directory the selection itself reads, so the two answers
            // cannot drift.
            if let Some(startup_residency_facts) = capture_streaming_startup_residency_facts(
                self,
                &validated_artifact,
                &mut model_loading_performance_attribution,
            )? {
                self.startup_residency_facts = Some(startup_residency_facts);
            }
            let runtime = initialize_mlx_runtime_with_cleanup(
                self.memory_limits,
                &mut model_loading_performance_attribution,
            )?;
            let model_chunking = resolve_model_chunking_configuration(&self.chunking)?;
            let model = load_and_materialize_streaming_model(
                runtime,
                validated_artifact,
                &self.model_directory,
                model_chunking,
                self.decode_expert_cache_policy,
                &mut model_loading_performance_attribution,
            )?;
            let (resolved_model_id, resolved_model_revision) =
                resolve_validated_model_identity(&model_id, &model_revision)?;
            let persistent_visual_embedding_model_contract =
                resolve_visual_embedding_model_contract(
                    qwen3_5_vision_config.as_ref(),
                    &resolved_model_id,
                    &resolved_model_revision,
                );
            let (persistent_prompt_cache_model_contract, persistent_prompt_cache) =
                open_streaming_persistent_prompt_cache(
                    self.persistent_prompt_cache_disk_store_config.clone(),
                    &model,
                    &self.chunking,
                    &resolved_model_id,
                    &resolved_model_revision,
                    persistent_visual_embedding_model_contract.as_ref(),
                    &mut model_loading_performance_attribution,
                )?;
            model
                .runtime()
                .synchronize_gpu_stream_and_clear_allocator_cache()
                .map_err(qwen3_5_runtime_error)?;
            Ok((
                model,
                persistent_prompt_cache_model_contract,
                persistent_visual_embedding_model_contract,
                persistent_prompt_cache,
            ))
        })();

        match model_loading_result {
            Ok((
                model,
                persistent_prompt_cache_model_contract,
                persistent_visual_embedding_model_contract,
                persistent_prompt_cache,
            )) => self.install_loaded_streaming_model(
                model,
                persistent_prompt_cache_model_contract,
                persistent_visual_embedding_model_contract,
                persistent_prompt_cache,
                model_loading_performance_attribution,
                model_id,
                model_revision,
                total_artifact_payload_bytes,
                model_shard_count,
            ),
            Err(model_loading_error) => {
                record_failed_streaming_model_loading(
                    self,
                    model_loading_performance_attribution,
                    model_id,
                    model_revision,
                    total_artifact_payload_bytes,
                    model_shard_count,
                    &model_loading_error,
                );
                Err(model_loading_error)
            }
        }
    }

    /// Moves the successfully loaded model and its cache ownership into the
    /// engine state, records the successful loading attribution, and reports
    /// the engine load result.
    #[allow(clippy::too_many_arguments)]
    fn install_loaded_streaming_model(
        &mut self,
        model: Qwen3_5Model,
        persistent_prompt_cache_model_contract: Option<PersistentPromptCacheModelContract>,
        persistent_visual_embedding_model_contract: Option<PersistentVisualEmbeddingModelContract>,
        persistent_prompt_cache: Option<Arc<PersistentPromptCacheDiskStore>>,
        model_loading_performance_attribution: PerformanceAttribution,
        model_id: Option<String>,
        model_revision: Option<String>,
        total_artifact_payload_bytes: Option<u64>,
        model_shard_count: Option<usize>,
    ) -> Result<EngineLoadResult, InferenceEngineError> {
        self.model_id = model_id.clone();
        self.model_revision = model_revision.clone();
        self.persistent_prompt_cache_model_contract = persistent_prompt_cache_model_contract;
        self.persistent_visual_embedding_model_contract =
            persistent_visual_embedding_model_contract;
        self.persistent_prompt_cache = persistent_prompt_cache;
        let mlx_memory_snapshot = model.runtime().memory_snapshot().ok();
        let resident_model_payload_bytes = Some(model.resident_model_payload_byte_count());
        let minimum_mlx_memory_ceiling_bytes = model.minimum_mlx_memory_ceiling_bytes()?;
        self.model = Some(model);
        if let Err(performance_attribution_error) = self
            .record_model_loading_performance_attribution(
                model_loading_performance_attribution,
                PerformanceAttributionOutcome::Success,
                model_id,
                model_revision,
                total_artifact_payload_bytes,
                resident_model_payload_bytes,
                model_shard_count,
                mlx_memory_snapshot,
                None,
            )
        {
            tracing::warn!(
                error = %performance_attribution_error,
                "failed to persist model-loading performance attribution after successful load"
            );
        }
        Ok(self.engine_load_result(minimum_mlx_memory_ceiling_bytes))
    }
}

/// Captures the startup residency facts from the validated artifact under
/// attribution. A zero complete-expert payload leaves the previous facts in
/// place: an artifact without experts has no residency story to tell.
fn capture_streaming_startup_residency_facts(
    engine_state: &Qwen3_5EngineState,
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<Option<StreamingStartupResidencyFacts>, InferenceEngineError> {
    let (startup_geometry, startup_complete_residency_headroom_bytes) =
        model_loading_performance_attribution
            .measure_operation(
                PerformanceOperation::ArtifactRamBudgetGeometry,
                |_performance_attribution| {
                    crate::mlx_ram_budget_model_geometry_from_validated_artifact(
                        validated_artifact,
                        &engine_state.model_directory,
                    )
                },
            )
            .map_err(|geometry_error| {
                fatal_engine_error(format!(
                    "startup residency selection facts are unreadable for this artifact: {geometry_error}"
                ))
            })?;
    if startup_geometry.complete_expert_payload_bytes == 0 {
        return Ok(None);
    }
    Ok(Some(crate::StreamingStartupResidencyFacts {
        model_core_payload_bytes: startup_geometry.model_core_payload_bytes,
        complete_expert_payload_bytes: startup_geometry.complete_expert_payload_bytes,
        startup_complete_residency_headroom_bytes,
        sequence_state_bytes_per_token: startup_geometry.sequence_state_bytes_per_token,
    }))
}

/// Initializes the MLX runtime under attribution and immediately clears the
/// allocator cache. Dropping the prior model moves its Metal buffers into
/// MLX's process-global allocator pool; releasing them before replacement
/// weights load keeps stale residency from competing during first-request
/// paging.
fn initialize_mlx_runtime_with_cleanup(
    memory_limits: astronomical_runtime_integration::MlxMemoryLimits,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxRuntime, InferenceEngineError> {
    let runtime = model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::MlxRuntimeInitialization,
            |_performance_attribution| MlxRuntime::initialize(memory_limits),
        )
        .map_err(qwen3_5_runtime_error)?;
    model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| runtime.clear_allocator_cache(),
        )
        .map_err(qwen3_5_runtime_error)?;
    Ok(runtime)
}

/// Pure: validates the worker chunking settings into the model chunking
/// configuration the streaming loader consumes.
fn resolve_model_chunking_configuration(
    chunking: &WorkerChunkingConfiguration,
) -> Result<Qwen3_5ModelChunkingConfiguration, InferenceEngineError> {
    Qwen3_5ModelChunkingConfiguration::new(
        chunking.full_attention_key_value_growth_tokens,
        chunking.prefill_graph_submission_layer_interval,
        chunking.experimental_ssd_paging_prefill_graph_submission_layer_interval,
        chunking.experimental_ssd_paging_generation_graph_submission_layer_interval,
    )
    .map_err(|configuration_error| {
        fatal_engine_error(format!(
            "failed to validate model chunking configuration: {configuration_error}"
        ))
    })
}

/// Loads the streaming model from the validated artifact under the engine's
/// decode-expert cache policy and materializes its target weights.
fn load_and_materialize_streaming_model(
    runtime: MlxRuntime,
    validated_artifact: ValidatedQwen3_5Artifact,
    model_directory: &std::path::Path,
    model_chunking: Qwen3_5ModelChunkingConfiguration,
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<Qwen3_5Model, InferenceEngineError> {
    let model = Qwen3_5Model::load_with_performance_attribution_and_decode_expert_cache_policy(
        runtime,
        validated_artifact,
        model_directory,
        true,
        model_chunking,
        decode_expert_cache_policy,
        model_loading_performance_attribution,
    )?;
    model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::ResidentWeightMaterializationSynchronizationWait,
            |_performance_attribution| model.materialize_target_weights(),
        )
        .map_err(qwen3_5_runtime_error)?;
    Ok(model)
}

/// Resolves the validated artifact identity the state fields report after a
/// successful load; losing either during loading is a fatal engine error.
fn resolve_validated_model_identity(
    model_id: &Option<String>,
    model_revision: &Option<String>,
) -> Result<(String, String), InferenceEngineError> {
    let resolved_model_id = model_id
        .clone()
        .ok_or_else(|| fatal_engine_error("model loading lost the validated model identifier"))?;
    let resolved_model_revision = model_revision
        .clone()
        .ok_or_else(|| fatal_engine_error("model loading lost the validated model revision"))?;
    Ok((resolved_model_id, resolved_model_revision))
}

/// Pure: builds the in-memory vision tensor geometry contract used to
/// validate both direct visual embeddings and optional disk entries. It
/// creates no storage owner; scanning and all other disk work remain inside
/// the cache-enabled branch.
fn resolve_visual_embedding_model_contract(
    qwen3_5_vision_config: Option<&Qwen3_5VisionConfig>,
    resolved_model_id: &str,
    resolved_model_revision: &str,
) -> Option<PersistentVisualEmbeddingModelContract> {
    qwen3_5_vision_config.map(|vision_config| {
        let qwen3_5_image_processor = Qwen3_5ImageProcessor::from_vision_config(vision_config);
        PersistentVisualEmbeddingModelContract::new(
            resolved_model_id.to_owned(),
            resolved_model_revision.to_owned(),
            vision_config.out_hidden_size() as usize,
            qwen3_5_image_processor.maximum_image_token_count_after_spatial_merge(),
        )
    })
}

/// Opens the persistent prompt cache when a disk store is configured; a
/// disabled cache yields no contract and no store, so it cannot fail model
/// loading because of storage alignment, quota, stale files, or filesystem
/// availability (contract derivation is intentionally inside the same branch
/// as store ownership).
#[allow(clippy::too_many_arguments)]
fn open_streaming_persistent_prompt_cache(
    persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
    model: &Qwen3_5Model,
    chunking: &WorkerChunkingConfiguration,
    resolved_model_id: &str,
    resolved_model_revision: &str,
    persistent_visual_embedding_model_contract: Option<&PersistentVisualEmbeddingModelContract>,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<
    (
        Option<PersistentPromptCacheModelContract>,
        Option<Arc<PersistentPromptCacheDiskStore>>,
    ),
    InferenceEngineError,
> {
    let Some(persistent_prompt_cache_disk_store_config) = persistent_prompt_cache_disk_store_config
    else {
        return Ok((None, None));
    };
    let global_prompt_cache_maximum_size_bytes =
        persistent_prompt_cache_disk_store_config.global_prompt_cache_maximum_size_bytes();
    let persistent_prompt_cache_model_contract = PersistentPromptCacheModelContract::resolve(
        resolved_model_id.to_owned(),
        resolved_model_revision.to_owned(),
        model.decoder_cache_layout().clone(),
        model.config().maximum_position_count() as usize,
        model.runtime().memory_limits().active_memory_limit_bytes() as u64,
        global_prompt_cache_maximum_size_bytes,
        chunking
            .prompt_cache_block_tokens
            .map(|block_token_count| block_token_count as usize),
        chunking.prompt_cache_common_prefix_stride_blocks,
    )
    .map_err(|model_contract_error| {
        fatal_engine_error(format!(
            "could not resolve persistent model-state storage contract: {model_contract_error}"
        ))
    })?;
    let persistent_prompt_cache = model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::PersistentPromptCacheOpenAndScan,
            |_performance_attribution| {
                PersistentPromptCacheDiskStore::open(
                    persistent_prompt_cache_disk_store_config,
                    persistent_prompt_cache_model_contract.clone(),
                )
            },
        )
        .map_err(|persistent_prompt_cache_error| {
            fatal_engine_error(format!(
                "required target prompt-state storage initialization failed: {persistent_prompt_cache_error}"
            ))
        })?;
    if let Some(persistent_visual_embedding_model_contract) =
        persistent_visual_embedding_model_contract
        && let Err(visual_embedding_scan_error) = persistent_prompt_cache
            .scan_visual_embeddings(persistent_visual_embedding_model_contract)
    {
        return Err(fatal_engine_error(format!(
            "required visual prompt-state storage scan failed: {visual_embedding_scan_error}"
        )));
    }
    persistent_prompt_cache_startup_logging::log_persistent_prompt_cache_startup_cleanup(
        "target",
        &persistent_prompt_cache,
    );
    tracing::info!(
        sequence_state_block_count = persistent_prompt_cache.sequence_state_block_count(),
        boundary_state_snapshot_count = persistent_prompt_cache.boundary_state_snapshot_count(),
        total_size_bytes = persistent_prompt_cache.total_size_bytes(),
        maximum_size_bytes = global_prompt_cache_maximum_size_bytes,
        "opened Qwen3.5 persistent prompt cache"
    );
    Ok((
        Some(persistent_prompt_cache_model_contract),
        Some(Arc::new(persistent_prompt_cache)),
    ))
}

/// Records the failed loading attribution for the artifact metadata
/// captured before the failure, warning without masking the original error.
fn record_failed_streaming_model_loading(
    engine_state: &mut Qwen3_5EngineState,
    model_loading_performance_attribution: PerformanceAttribution,
    model_id: Option<String>,
    model_revision: Option<String>,
    total_artifact_payload_bytes: Option<u64>,
    model_shard_count: Option<usize>,
    model_loading_error: &InferenceEngineError,
) {
    if let Err(performance_attribution_error) = engine_state
        .record_model_loading_performance_attribution(
            model_loading_performance_attribution,
            PerformanceAttributionOutcome::Failed,
            model_id,
            model_revision,
            total_artifact_payload_bytes,
            None,
            model_shard_count,
            None,
            Some(model_loading_error.to_string()),
        )
    {
        tracing::warn!(
            error = %performance_attribution_error,
            "failed to persist model-loading performance attribution after failure"
        );
    }
}
