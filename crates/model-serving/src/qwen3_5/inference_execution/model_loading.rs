use astronomical_runtime_integration::MlxRuntime;
use std::sync::Arc;

use crate::{
    EngineLoadResult, InferenceEngineError, PerformanceAttribution, PerformanceAttributionOutcome,
    PerformanceOperation, PersistentPromptCacheDiskStore, PersistentPromptCacheModelContract,
    PersistentPromptCacheWriteQueue, PersistentVisualEmbeddingModelContract,
};

use super::{
    Qwen3_5EngineState, Qwen3_5MtpRuntimeState, fatal_engine_error, qwen3_5_runtime_error,
};
use crate::qwen3_5::multi_token_prediction::{
    materialize_optional_weights, qwen3_5_mtp_runtime_state_after_load,
};
use crate::qwen3_5::{
    Qwen3_5FeedForwardArchitecture, Qwen3_5ImageProcessor, Qwen3_5Model,
    Qwen3_5MtpArtifactCapability,
};
use astronomical_ipc_protocol::SpeculativePrefillRuntimeState;

use super::speculative_prefill_model_loading::{
    load_speculative_prefill_draft_model, token_identifier_mapping_digest,
};

impl Qwen3_5EngineState {
    pub(super) fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        tracing::info!(
            speculative_prefill_enabled = self.speculative_prefill.enabled,
            speculative_prefill_draft_model_id = ?self.speculative_prefill.draft_model_id,
            "resolved Qwen3.5 speculative-prefill policy"
        );
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
        let mut target_token_identifier_mapping_digest = None;
        let mut qwen3_5_mtp_artifact_capability = Qwen3_5MtpArtifactCapability::TargetOnly;
        let model_loading_result: Result<_, InferenceEngineError> = (|| {
            let validated_artifact = self.validated_artifact.take().ok_or_else(|| {
                fatal_engine_error("validated Qwen3.5 artifact is unavailable during MLX load")
            })?;
            let resolved_target_token_identifier_mapping_digest =
                token_identifier_mapping_digest(&validated_artifact)?;
            target_token_identifier_mapping_digest =
                Some(resolved_target_token_identifier_mapping_digest);
            let target_max_output_tokens = validated_artifact.max_output_tokens();
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
                true,
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
                        |_performance_attribution| materialize_optional_weights(&mut model),
                    )
            {
                tracing::warn!(
                    error = %mtp_materialization_error,
                    "optional MTP weight materialization failed; serving target-only"
                );
                if let Err(mlx_allocator_cleanup_error) = model
                    .runtime()
                    .synchronize_gpu_stream_and_clear_allocator_cache()
                {
                    tracing::warn!(
                        error = %mlx_allocator_cleanup_error,
                        "failed to reclaim allocator memory after optional MTP initialization failure"
                    );
                }
            }
            let (speculative_prefill_draft_model, speculative_prefill_unavailable_reason) =
                load_speculative_prefill_draft_model(
                    &model,
                    &self.speculative_prefill,
                    resolved_target_token_identifier_mapping_digest,
                    target_max_output_tokens,
                    self.memory_limits,
                    &mut model_loading_performance_attribution,
                )?;
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
            let (
                speculative_prefill_draft_persistent_prompt_cache,
                speculative_prefill_draft_persistent_prompt_cache_write_queue,
            ) = if let Some((draft_model, draft_model_revision)) =
                speculative_prefill_draft_model.as_ref()
                && let Some(persistent_prompt_cache_disk_store_config) =
                    self.persistent_prompt_cache_disk_store_config.as_ref()
            {
                let draft_model_id =
                    self.speculative_prefill
                        .draft_model_id
                        .clone()
                        .ok_or_else(|| {
                            fatal_engine_error(
                                "loaded speculative-prefill draft model has no configured model ID",
                            )
                        })?;
                let draft_model_contract = PersistentPromptCacheModelContract::new(
                    draft_model_id.clone(),
                    draft_model_revision.clone(),
                    draft_model.decoder_cache_layout().clone(),
                );
                let draft_prompt_cache_config = persistent_prompt_cache_disk_store_config
                    .for_model(&draft_model_id, draft_model_revision);
                let draft_prompt_cache_maximum_size_bytes =
                    draft_prompt_cache_config.global_prompt_cache_maximum_size_bytes();
                let draft_ssd_write_rate_megabytes_per_second =
                    draft_prompt_cache_config.ssd_write_rate_megabytes_per_second();
                match model_loading_performance_attribution.measure_operation(
                    PerformanceOperation::PersistentPromptCacheOpenAndScan,
                    |_performance_attribution| {
                        PersistentPromptCacheDiskStore::open(
                            draft_prompt_cache_config,
                            draft_model_contract,
                        )
                    },
                ) {
                    Ok(draft_persistent_prompt_cache) => {
                        tracing::info!(
                            draft_model_id,
                            draft_model_revision,
                            sequence_state_block_count =
                                draft_persistent_prompt_cache.sequence_state_block_count(),
                            boundary_state_snapshot_count =
                                draft_persistent_prompt_cache.boundary_state_snapshot_count(),
                            total_size_bytes = draft_persistent_prompt_cache.total_size_bytes(),
                            maximum_size_bytes = draft_prompt_cache_maximum_size_bytes,
                            "opened speculative-prefill drafter persistent prompt cache"
                        );
                        let draft_persistent_prompt_cache = Arc::new(draft_persistent_prompt_cache);
                        let draft_persistent_prompt_cache_write_queue =
                            match PersistentPromptCacheWriteQueue::new(
                                Arc::clone(&draft_persistent_prompt_cache),
                                draft_ssd_write_rate_megabytes_per_second,
                            ) {
                                Ok(write_queue) => Some(write_queue),
                                Err(write_queue_error) => {
                                    tracing::warn!(
                                        error = %write_queue_error,
                                        "speculative-prefill drafter cache writer could not start; serving reads without new writes"
                                    );
                                    None
                                }
                            };
                        (
                            Some(draft_persistent_prompt_cache),
                            draft_persistent_prompt_cache_write_queue,
                        )
                    }
                    Err(draft_persistent_prompt_cache_error) => {
                        tracing::warn!(
                            error = %draft_persistent_prompt_cache_error,
                            "speculative-prefill drafter persistent cache could not be opened; serving without drafter cache persistence"
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
                speculative_prefill_draft_persistent_prompt_cache,
                speculative_prefill_draft_persistent_prompt_cache_write_queue,
                (
                    speculative_prefill_draft_model,
                    speculative_prefill_unavailable_reason,
                ),
            ))
        })();

        match model_loading_result {
            Ok((
                model,
                model_contract,
                persistent_visual_embedding_model_contract,
                persistent_prompt_cache,
                persistent_prompt_cache_write_queue,
                speculative_prefill_draft_persistent_prompt_cache,
                speculative_prefill_draft_persistent_prompt_cache_write_queue,
                (speculative_prefill_draft_model, speculative_prefill_unavailable_reason),
            )) => {
                self.model_id = model_id.clone();
                self.model_revision = model_revision.clone();
                self.persistent_prompt_cache_model_contract = Some(model_contract);
                self.persistent_visual_embedding_model_contract =
                    persistent_visual_embedding_model_contract;
                self.persistent_prompt_cache = persistent_prompt_cache;
                self.persistent_prompt_cache_write_queue = persistent_prompt_cache_write_queue;
                self.speculative_prefill_draft_persistent_prompt_cache =
                    speculative_prefill_draft_persistent_prompt_cache;
                self.speculative_prefill_draft_persistent_prompt_cache_write_queue =
                    speculative_prefill_draft_persistent_prompt_cache_write_queue;
                let speculative_prefill_draft_is_available =
                    speculative_prefill_draft_model.is_some();
                let speculative_prefill_draft_supports_processed_visual_images =
                    speculative_prefill_draft_model.as_ref().is_some_and(
                        |(draft_model, _draft_model_revision)| {
                            draft_model
                                .vision_model()
                                .is_some_and(|draft_vision_model| {
                                    model.vision_model().is_some_and(|target_vision_model| {
                                        draft_vision_model
                                            .accepts_processed_images_from(target_vision_model)
                                    })
                                })
                        },
                    );
                let (speculative_prefill_draft_model, loaded_draft_model_revision) =
                    speculative_prefill_draft_model.map_or(
                        (None, None),
                        |(draft_model, draft_model_revision)| {
                            drop(draft_model);
                            (None, Some(draft_model_revision))
                        },
                    );
                self.speculative_prefill_draft_model = speculative_prefill_draft_model;
                self.speculative_prefill_draft_model_revision = loaded_draft_model_revision;
                self.speculative_prefill_draft_is_available =
                    speculative_prefill_draft_is_available;
                self.speculative_prefill_draft_supports_processed_visual_images =
                    speculative_prefill_draft_supports_processed_visual_images;
                model
                    .runtime()
                    .synchronize_gpu_stream_and_clear_allocator_cache()
                    .map_err(qwen3_5_runtime_error)?;
                self.speculative_prefill_token_identifier_mapping_digest =
                    target_token_identifier_mapping_digest;
                self.speculative_prefill_runtime_state = if !self.speculative_prefill.enabled {
                    SpeculativePrefillRuntimeState::Disabled
                } else if self.speculative_prefill_draft_is_available {
                    SpeculativePrefillRuntimeState::Active
                } else {
                    SpeculativePrefillRuntimeState::Unavailable
                };
                self.speculative_prefill_unavailable_reason =
                    speculative_prefill_unavailable_reason;
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
                Ok(self.engine_load_result_for_mtp_state(minimum_mlx_memory_ceiling_bytes))
            }
            Err(model_loading_error) => {
                if let Err(performance_attribution_error) = self
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
                Err(model_loading_error)
            }
        }
    }
}
