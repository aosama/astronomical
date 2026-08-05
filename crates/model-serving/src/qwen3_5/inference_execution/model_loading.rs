use astronomical_runtime_integration::MlxRuntime;
use std::sync::Arc;

use crate::{
    EngineLoadResult, InferenceEngineError, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionOutcome, PerformanceOperation,
    PersistentPromptCacheDiskStore, PersistentPromptCacheModelContract,
    PersistentPromptCacheWriteQueue, PersistentVisualEmbeddingModelContract,
};

use super::{
    Qwen3_5EngineState, Qwen3_5MtpRuntimeState, fatal_engine_error, qwen3_5_runtime_error,
};
use crate::qwen3_5::{
    Qwen3_5FeedForwardArchitecture, Qwen3_5ImageProcessor, Qwen3_5Model,
    Qwen3_5MtpArtifactCapability, Qwen3_5MtpUnavailableReason,
};

impl Qwen3_5EngineState {
    pub(super) fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        if let Some(model) = self.model.as_ref() {
            return Ok(
                self.engine_load_result_for_mtp_state(model.minimum_mlx_memory_ceiling_bytes()?)
            );
        }
        let mut model_loading_performance_attribution = self
            .model_loading_performance_attribution
            .take()
            .unwrap_or_else(PerformanceAttribution::disabled);
        let mut model_id = None;
        let mut model_revision = None;
        let mut total_artifact_payload_bytes = None;
        let mut model_shard_count = None;
        let mut qwen3_5_mtp_artifact_capability = Qwen3_5MtpArtifactCapability::TargetOnly;
        let model_loading_result: Result<_, InferenceEngineError> = (|| {
            let validated_artifact = self.validated_artifact.take().ok_or_else(|| {
                fatal_engine_error("validated Qwen3.5 artifact is unavailable during MLX load")
            })?;
            let qwen3_5_vision_config = validated_artifact.vision_config().cloned();
            qwen3_5_mtp_artifact_capability = validated_artifact.mtp_artifact_capability().clone();
            model_id = Some(validated_artifact.model_id().to_owned());
            model_revision = Some(validated_artifact.revision().to_owned());
            total_artifact_payload_bytes = Some(validated_artifact.total_payload_bytes());
            model_shard_count = Some(validated_artifact.shard_count());
            let runtime = model_loading_performance_attribution
                .measure_operation(
                    PerformanceOperation::MlxRuntimeInitialization,
                    |_performance_attribution| MlxRuntime::initialize(self.memory_limits),
                )
                .map_err(qwen3_5_runtime_error)?;
            // Dropping the prior model moves its Metal buffers into MLX's
            // process-global allocator pool. Release them before replacement
            // weights compete with stale residency during first-request paging.
            model_loading_performance_attribution
                .measure_operation(
                    PerformanceOperation::MlxAllocatorCacheCleanup,
                    |_performance_attribution| runtime.clear_allocator_cache(),
                )
                .map_err(qwen3_5_runtime_error)?;
            let mut model = Qwen3_5Model::load_with_performance_attribution(
                runtime,
                validated_artifact,
                &self.model_directory,
                self.mtp_enabled,
                &mut model_loading_performance_attribution,
            )
            .map_err(qwen3_5_runtime_error)?;
            model_loading_performance_attribution
                .measure_operation(
                    PerformanceOperation::ResidentWeightMaterializationSynchronizationWait,
                    |_performance_attribution| model.materialize_target_weights(),
                )
                .map_err(qwen3_5_runtime_error)?;
            if self.mtp_enabled
                && model.mtp_weights()
                && let Err(mtp_materialization_error) = model_loading_performance_attribution
                    .measure_operation(
                        PerformanceOperation::ResidentWeightMaterializationSynchronizationWait,
                        |_performance_attribution| model.materialize_mtp_weights(),
                    )
            {
                tracing::warn!(
                    error = %mtp_materialization_error,
                    "optional MTP weight materialization failed; serving target-only"
                );
                if let Err(mlx_allocator_cleanup_error) = model.runtime().clear_allocator_cache() {
                    tracing::warn!(
                        error = %mlx_allocator_cleanup_error,
                        "failed to reclaim allocator memory after optional MTP initialization failure"
                    );
                }
            }
            match model.config().feed_forward_architecture() {
                Qwen3_5FeedForwardArchitecture::Dense => {}
                Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
                    let expert_residency_started_at = std::time::Instant::now();
                    tracing::info!("started filling idle memory with complete expert layers");
                    let expert_residency_outcome = model
                        .prewarm_complete_expert_layers_with_performance_attribution(
                            &mut model_loading_performance_attribution,
                        );
                    let expert_weight_memory_cache_statistics =
                        model.expert_weight_memory_cache_statistics();
                    tracing::info!(
                        residency_succeeded = expert_residency_outcome.is_ok(),
                        residency_elapsed_millis = expert_residency_started_at.elapsed().as_millis(),
                        expert_memory_mode = ?model.expert_memory_mode(),
                        retained_complete_layer_count =
                            expert_weight_memory_cache_statistics.complete_layer_count,
                        retained_expert_payload_bytes =
                            expert_weight_memory_cache_statistics.resident_payload_byte_count,
                        maximum_retained_expert_payload_bytes =
                            expert_weight_memory_cache_statistics.maximum_resident_payload_byte_count,
                        "finished filling idle memory with complete expert layers"
                    );
                    if let Err(expert_residency_error) = expert_residency_outcome {
                        tracing::warn!(
                            error = %expert_residency_error,
                            "could not finish automatic expert residency; serving with the admitted layers"
                        );
                    }
                }
            }
            let resolved_model_id = model_id.clone().ok_or_else(|| {
                fatal_engine_error("model loading lost the validated model identifier")
            })?;
            let resolved_model_revision = model_revision.clone().ok_or_else(|| {
                fatal_engine_error("model loading lost the validated model revision")
            })?;
            let decoder_cache_layout = model.decoder_cache_layout().clone();
            let model_contract = PersistentPromptCacheModelContract::new(
                resolved_model_id.clone(),
                resolved_model_revision.clone(),
                decoder_cache_layout,
            );
            let persistent_visual_embedding_model_contract =
                qwen3_5_vision_config.as_ref().map(|vision_config| {
                    let qwen3_5_image_processor =
                        Qwen3_5ImageProcessor::from_vision_config(vision_config);
                    PersistentVisualEmbeddingModelContract::new(
                        resolved_model_id.clone(),
                        resolved_model_revision.clone(),
                        vision_config.out_hidden_size() as usize,
                        qwen3_5_image_processor.maximum_image_token_count_after_spatial_merge(),
                    )
                });
            let (persistent_prompt_cache, persistent_prompt_cache_write_queue) = if let Some(
                persistent_prompt_cache_disk_store_config,
            ) =
                self.persistent_prompt_cache_disk_store_config.clone()
            {
                let global_prompt_cache_maximum_size_bytes =
                    persistent_prompt_cache_disk_store_config
                        .global_prompt_cache_maximum_size_bytes();
                let ssd_write_rate_megabytes_per_second =
                    persistent_prompt_cache_disk_store_config.ssd_write_rate_megabytes_per_second();
                match model_loading_performance_attribution.measure_operation(
                    PerformanceOperation::PersistentPromptCacheOpenAndScan,
                    |_performance_attribution| {
                        PersistentPromptCacheDiskStore::open(
                            persistent_prompt_cache_disk_store_config,
                            model_contract.clone(),
                        )
                    },
                ) {
                    Ok(persistent_prompt_cache) => {
                        if let Some(persistent_visual_embedding_model_contract) =
                            persistent_visual_embedding_model_contract.as_ref()
                            && let Err(visual_embedding_scan_error) = persistent_prompt_cache
                                .scan_visual_embeddings(persistent_visual_embedding_model_contract)
                        {
                            tracing::warn!(
                                "Qwen3.5 visual embedding cache could not be scanned; \
                                 continuing without persisted visual embeddings: {visual_embedding_scan_error}"
                            );
                        }
                        tracing::info!(
                            sequence_state_block_count =
                                persistent_prompt_cache.sequence_state_block_count(),
                            boundary_state_snapshot_count =
                                persistent_prompt_cache.boundary_state_snapshot_count(),
                            total_size_bytes = persistent_prompt_cache.total_size_bytes(),
                            maximum_size_bytes = global_prompt_cache_maximum_size_bytes,
                            "opened Qwen3.5 persistent prompt cache"
                        );
                        let persistent_prompt_cache = Arc::new(persistent_prompt_cache);
                        let persistent_prompt_cache_write_queue =
                            match PersistentPromptCacheWriteQueue::new(
                                Arc::clone(&persistent_prompt_cache),
                                ssd_write_rate_megabytes_per_second,
                            ) {
                                Ok(write_queue) => Some(write_queue),
                                Err(write_queue_error) => {
                                    tracing::warn!(
                                        error = %write_queue_error,
                                        "persistent prompt-cache writer could not start; serving cache reads without new writes"
                                    );
                                    None
                                }
                            };
                        (
                            Some(persistent_prompt_cache),
                            persistent_prompt_cache_write_queue,
                        )
                    }
                    Err(persistent_prompt_cache_error) => {
                        tracing::warn!(
                            "Qwen3.5 persistent prompt cache could not be opened; \
                             falling back to cold prefill: {persistent_prompt_cache_error}"
                        );
                        (None, None)
                    }
                }
            } else {
                (None, None)
            };
            Ok((
                model,
                model_contract,
                persistent_visual_embedding_model_contract,
                persistent_prompt_cache,
                persistent_prompt_cache_write_queue,
            ))
        })();

        match model_loading_result {
            Ok((
                model,
                model_contract,
                persistent_visual_embedding_model_contract,
                persistent_prompt_cache,
                persistent_prompt_cache_write_queue,
            )) => {
                self.model_id = model_id.clone();
                self.model_revision = model_revision.clone();
                self.persistent_prompt_cache_model_contract = Some(model_contract);
                self.persistent_visual_embedding_model_contract =
                    persistent_visual_embedding_model_contract;
                self.persistent_prompt_cache = persistent_prompt_cache;
                self.persistent_prompt_cache_write_queue = persistent_prompt_cache_write_queue;
                let mlx_memory_snapshot = model.runtime().memory_snapshot().ok();
                let resident_model_payload_bytes = Some(model.resident_model_payload_byte_count());
                let minimum_mlx_memory_ceiling_bytes = model.minimum_mlx_memory_ceiling_bytes()?;
                let model_has_mtp_weights = model.mtp_weights();
                self.model = Some(model);
                let (mtp_runtime_state, mtp_unavailable_reason) =
                    qwen3_5_mtp_runtime_state_after_load(
                        self.mtp_enabled,
                        &qwen3_5_mtp_artifact_capability,
                        model_has_mtp_weights,
                    );
                self.mtp_runtime_state = mtp_runtime_state;
                self.mtp_unavailable_reason = mtp_unavailable_reason;
                match self.mtp_runtime_state {
                    Qwen3_5MtpRuntimeState::Disabled => {}
                    Qwen3_5MtpRuntimeState::TargetOnly => tracing::info!(
                        model_id = self.model_id.as_deref().unwrap_or("unknown"),
                        "MTP is enabled but the selected model has no MTP inventory; serving target-only"
                    ),
                    Qwen3_5MtpRuntimeState::Active => tracing::info!(
                        model_id = self.model_id.as_deref().unwrap_or("unknown"),
                        "native MTP is active for this model"
                    ),
                    Qwen3_5MtpRuntimeState::Unavailable => {
                        let mtp_unavailable_reason = self
                            .mtp_unavailable_reason
                            .as_deref()
                            .unwrap_or("unknown MTP initialization failure");
                        tracing::warn!(
                            model_id = self.model_id.as_deref().unwrap_or("unknown"),
                            mtp_unavailable_reason,
                            "MTP is enabled but unavailable; serving target-only"
                        );
                    }
                }
                self.record_model_loading_performance_attribution(
                    model_loading_performance_attribution,
                    PerformanceAttributionOutcome::Success,
                    model_id,
                    model_revision,
                    total_artifact_payload_bytes,
                    resident_model_payload_bytes,
                    model_shard_count,
                    mlx_memory_snapshot,
                    None,
                );
                Ok(self.engine_load_result_for_mtp_state(minimum_mlx_memory_ceiling_bytes))
            }
            Err(model_loading_error) => {
                self.record_model_loading_performance_attribution(
                    model_loading_performance_attribution,
                    PerformanceAttributionOutcome::Failed,
                    model_id,
                    model_revision,
                    total_artifact_payload_bytes,
                    None,
                    model_shard_count,
                    None,
                    Some(model_loading_error.to_string()),
                );
                Err(model_loading_error)
            }
        }
    }

    fn engine_load_result_for_mtp_state(
        &self,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> EngineLoadResult {
        let mtp_runtime_state = match self.mtp_runtime_state {
            Qwen3_5MtpRuntimeState::Disabled => {
                astronomical_ipc_protocol::MtpRuntimeState::Disabled
            }
            Qwen3_5MtpRuntimeState::TargetOnly => {
                astronomical_ipc_protocol::MtpRuntimeState::TargetOnly
            }
            Qwen3_5MtpRuntimeState::Active => astronomical_ipc_protocol::MtpRuntimeState::Active,
            Qwen3_5MtpRuntimeState::Unavailable => {
                astronomical_ipc_protocol::MtpRuntimeState::Unavailable
            }
        };
        let mut engine_load_result =
            EngineLoadResult::new().with_mtp_runtime_state(mtp_runtime_state);
        if self.mtp_runtime_state == Qwen3_5MtpRuntimeState::Unavailable {
            if let Some(mtp_unavailable_reason) = self.mtp_unavailable_reason.as_ref() {
                engine_load_result =
                    engine_load_result.with_mtp_unavailable_reason(mtp_unavailable_reason.clone());
            }
        }
        engine_load_result.with_minimum_mlx_memory_ceiling_bytes(minimum_mlx_memory_ceiling_bytes)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_model_loading_performance_attribution(
        &mut self,
        model_loading_performance_attribution: PerformanceAttribution,
        outcome: PerformanceAttributionOutcome,
        model_id: Option<String>,
        model_revision: Option<String>,
        total_artifact_payload_bytes: Option<u64>,
        resident_model_payload_bytes: Option<u64>,
        model_shard_count: Option<usize>,
        mlx_memory_snapshot: Option<astronomical_runtime_integration::MlxMemorySnapshot>,
        failure_description: Option<String>,
    ) {
        let Some(performance_attribution_report) = model_loading_performance_attribution
            .finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
                outcome,
                model_id,
                model_revision,
                prefill_transient_observation_completed: self
                    .adaptive_ram_growth_guard
                    .has_completed_growth_observation(crate::AdaptiveRamGrowthPhase::Prefill),
                prefill_observed_transient_high_water_bytes: u64::try_from(
                    self.adaptive_ram_growth_guard
                        .observed_transient_high_water_bytes(
                            crate::AdaptiveRamGrowthPhase::Prefill,
                        ),
                )
                .unwrap_or(u64::MAX),
                retained_complete_expert_layer_count: self.model.as_ref().map_or(0, |model| {
                    u64::try_from(
                        model
                            .expert_weight_memory_cache_statistics()
                            .complete_layer_count,
                    )
                    .unwrap_or(u64::MAX)
                }),
                total_artifact_payload_bytes,
                resident_model_payload_bytes,
                model_shard_count,
                mlx_active_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_allocator_cache_memory_bytes: mlx_memory_snapshot.as_ref().and_then(
                    |snapshot| u64::try_from(snapshot.allocator_cache_memory_bytes()).ok(),
                ),
                mlx_peak_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
                failure_description,
            })
        else {
            return;
        };
        if let Err(performance_attribution_write_error) = self
            .performance_attribution_log
            .record(&performance_attribution_report)
        {
            tracing::warn!(
                error = %performance_attribution_write_error,
                "failed to append model-loading performance attribution"
            );
        }
    }
}

/// Derives the public MTP state after optional head initialization.
#[doc(hidden)]
#[must_use]
pub fn qwen3_5_mtp_runtime_state_after_load(
    mtp_enabled: bool,
    mtp_artifact_capability: &Qwen3_5MtpArtifactCapability,
    model_has_mtp_weights: bool,
) -> (Qwen3_5MtpRuntimeState, Option<String>) {
    if !mtp_enabled {
        return (Qwen3_5MtpRuntimeState::Disabled, None);
    }
    if model_has_mtp_weights {
        return (Qwen3_5MtpRuntimeState::Active, None);
    }
    match mtp_artifact_capability {
        Qwen3_5MtpArtifactCapability::TargetOnly => (Qwen3_5MtpRuntimeState::TargetOnly, None),
        Qwen3_5MtpArtifactCapability::MtpCapable { .. } => (
            Qwen3_5MtpRuntimeState::Unavailable,
            Some(Qwen3_5MtpUnavailableReason::NoCompatibleHead.to_string()),
        ),
        Qwen3_5MtpArtifactCapability::InvalidMtp { reason } => (
            Qwen3_5MtpRuntimeState::Unavailable,
            Some(format!("invalid MTP inventory: {reason}")),
        ),
    }
}
