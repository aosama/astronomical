use astronomical_runtime_integration::MlxRuntime;
use std::sync::Arc;

use crate::{
    EngineLoadResult, InferenceEngineError, PerformanceAttribution, PerformanceAttributionOutcome,
    PerformanceOperation, PersistentPromptCacheDiskStore, PersistentPromptCacheModelContract,
    PersistentVisualEmbeddingModelContract,
};

use super::{
    Qwen3_5EngineState, Qwen3_5MtpRuntimeState, fatal_engine_error, qwen3_5_runtime_error,
};
use crate::qwen3_5::model::Qwen3_5ModelChunkingConfiguration;
use crate::qwen3_5::multi_token_prediction::{
    materialize_optional_weights, qwen3_5_mtp_runtime_configuration_after_load,
};
use crate::qwen3_5::{Qwen3_5ImageProcessor, Qwen3_5Model, Qwen3_5MtpArtifactCapability};
use crate::qwen3_5_moe::Qwen3_5ExpertResidencyTransitionReason;
use astronomical_ipc_protocol::SpeculativePrefillRuntimeState;

use super::persistent_prompt_cache_startup_logging::log_persistent_prompt_cache_startup_cleanup;
use super::speculative_prefill::configured_speculative_prefill_activation_failure;
use super::speculative_prefill::{
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
        let mut qwen3_5_mtp_artifact_capability = Qwen3_5MtpArtifactCapability::target_only(
            crate::qwen3_5::multi_token_prediction::Qwen3_5MtpTargetOnlyReason::NoTensorInventory,
        );
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
            let should_bind_mtp_weights = self.mtp_enabled
                && qwen3_5_mtp_artifact_capability
                    .supports_configured_depth(self.configured_mtp_draft_depth);
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
            let model_chunking = Qwen3_5ModelChunkingConfiguration::new(
                self.chunking.full_attention_key_value_growth_tokens,
                self.chunking.prefill_graph_submission_layer_interval,
                self.chunking
                    .experimental_ssd_paging_generation_graph_submission_layer_interval,
                self.chunking.speculative_prefill_draft_forward_tokens,
            )
            .map_err(|configuration_error| {
                fatal_engine_error(format!(
                    "failed to validate model chunking configuration: {configuration_error}"
                ))
            })?;
            let mut model = Qwen3_5Model::load_with_performance_attribution(
                runtime,
                validated_artifact,
                &self.model_directory,
                should_bind_mtp_weights,
                true,
                model_chunking,
                &mut model_loading_performance_attribution,
            )
            .map_err(qwen3_5_runtime_error)?;
            model_loading_performance_attribution
                .measure_operation(
                    PerformanceOperation::ResidentWeightMaterializationSynchronizationWait,
                    |_performance_attribution| model.materialize_target_weights(),
                )
                .map_err(qwen3_5_runtime_error)?;
            if should_bind_mtp_weights
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
            // This startup drafter exists only to prove compatibility and derive
            // its revision/storage geometry. It is dropped below before target
            // expert residency is admitted; requests load their own temporary copy.
            let resolved_model_id = model_id.clone().ok_or_else(|| {
                fatal_engine_error("model loading lost the validated model identifier")
            })?;
            let resolved_model_revision = model_revision.clone().ok_or_else(|| {
                fatal_engine_error("model loading lost the validated model revision")
            })?;
            let persistent_visual_embedding_model_contract =
                qwen3_5_vision_config.as_ref().map(|vision_config| {
                    // This object is in-memory vision tensor geometry used to
                    // validate both direct visual embeddings and optional disk
                    // entries. It creates no storage owner. Scanning and all
                    // other disk work remain inside the cache-enabled branch.
                    let qwen3_5_image_processor =
                        Qwen3_5ImageProcessor::from_vision_config(vision_config);
                    PersistentVisualEmbeddingModelContract::new(
                        resolved_model_id.clone(),
                        resolved_model_revision.clone(),
                        vision_config.out_hidden_size() as usize,
                        qwen3_5_image_processor.maximum_image_token_count_after_spatial_merge(),
                    )
                });
            let (persistent_prompt_cache_model_contract, persistent_prompt_cache) = if let Some(
                persistent_prompt_cache_disk_store_config,
            ) =
                self.persistent_prompt_cache_disk_store_config.clone()
            {
                // Contract derivation is intentionally inside the same
                // branch as store ownership. A disabled cache therefore
                // cannot fail model loading because of storage alignment,
                // quota, stale files, or filesystem availability.
                let global_prompt_cache_maximum_size_bytes =
                    persistent_prompt_cache_disk_store_config
                        .global_prompt_cache_maximum_size_bytes();
                let model_contract = PersistentPromptCacheModelContract::resolve(
                        resolved_model_id.clone(),
                        resolved_model_revision.clone(),
                        model.decoder_cache_layout().clone(),
                        model.config().maximum_position_count() as usize,
                        model.runtime().memory_limits().active_memory_limit_bytes() as u64,
                        global_prompt_cache_maximum_size_bytes,
                        self.chunking
                            .prompt_cache_block_tokens
                            .map(|block_token_count| block_token_count as usize),
                        self.chunking.prompt_cache_common_prefix_stride_blocks,
                    )
                    .map_err(|model_contract_error| {
                        fatal_engine_error(format!(
                            "could not resolve persistent model-state storage contract: {model_contract_error}"
                        ))
                    })?;
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
                            return Err(fatal_engine_error(format!(
                                "required visual prompt-state storage scan failed: {visual_embedding_scan_error}"
                            )));
                        }
                        log_persistent_prompt_cache_startup_cleanup(
                            "target",
                            &persistent_prompt_cache,
                        );
                        tracing::info!(
                            sequence_state_block_count =
                                persistent_prompt_cache.sequence_state_block_count(),
                            boundary_state_snapshot_count =
                                persistent_prompt_cache.boundary_state_snapshot_count(),
                            total_size_bytes = persistent_prompt_cache.total_size_bytes(),
                            maximum_size_bytes = global_prompt_cache_maximum_size_bytes,
                            "opened Qwen3.5 persistent prompt cache"
                        );
                        (
                            Some(model_contract),
                            Some(Arc::new(persistent_prompt_cache)),
                        )
                    }
                    Err(persistent_prompt_cache_error) => {
                        return Err(fatal_engine_error(format!(
                            "required target prompt-state storage initialization failed: {persistent_prompt_cache_error}"
                        )));
                    }
                }
            } else {
                (None, None)
            };
            let speculative_prefill_draft_persistent_prompt_cache = if let Some((
                draft_model,
                draft_model_revision,
            )) =
                speculative_prefill_draft_model.as_ref()
                && let Some(persistent_prompt_cache_disk_store_config) =
                    self.persistent_prompt_cache_disk_store_config.as_ref()
            {
                // Drafter dense state has different tensor geometry from target
                // state and therefore owns a separate model/revision namespace.
                let draft_model_id =
                    self.speculative_prefill
                        .draft_model_id
                        .clone()
                        .ok_or_else(|| {
                            fatal_engine_error(
                                "loaded speculative-prefill draft model has no configured model ID",
                            )
                        })?;
                let draft_model_contract = PersistentPromptCacheModelContract::resolve(
                    draft_model_id.clone(),
                    draft_model_revision.clone(),
                    draft_model.decoder_cache_layout().clone(),
                    draft_model.config().maximum_position_count() as usize,
                    draft_model
                        .runtime()
                        .memory_limits()
                        .active_memory_limit_bytes() as u64,
                    persistent_prompt_cache_disk_store_config
                        .global_prompt_cache_maximum_size_bytes(),
                    self.chunking
                        .prompt_cache_block_tokens
                        .map(|block_token_count| block_token_count as usize),
                    self.chunking.prompt_cache_common_prefix_stride_blocks,
                )
                .map_err(|draft_model_contract_error| {
                    configured_speculative_prefill_activation_failure(
                        "drafter persistent model-state storage contract",
                        draft_model_contract_error,
                    )
                })?;
                let draft_prompt_cache_config = persistent_prompt_cache_disk_store_config
                    .for_model(&draft_model_id, draft_model_revision);
                let draft_prompt_cache_maximum_size_bytes =
                    draft_prompt_cache_config.global_prompt_cache_maximum_size_bytes();
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
                        log_persistent_prompt_cache_startup_cleanup(
                            "speculative_prefill_draft",
                            &draft_persistent_prompt_cache,
                        );
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
                        Some(Arc::new(draft_persistent_prompt_cache))
                    }
                    Err(draft_persistent_prompt_cache_error) => {
                        return Err(fatal_engine_error(format!(
                            "required drafter prompt-state storage initialization failed: {draft_persistent_prompt_cache_error}"
                        )));
                    }
                }
            } else {
                None
            };
            self.purge_obsolete_speculative_prefill_policy_state(
                &resolved_model_id,
                &resolved_model_revision,
                speculative_prefill_draft_model
                    .as_ref()
                    .map(|(_draft_model, draft_model_revision)| draft_model_revision.as_str()),
                persistent_prompt_cache.as_deref(),
                speculative_prefill_draft_persistent_prompt_cache.as_deref(),
            )?;
            let speculative_prefill_draft_is_available = speculative_prefill_draft_model.is_some();
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
            let loaded_draft_model_revision =
                speculative_prefill_draft_model.map(|(draft_model, draft_model_revision)| {
                    // Release all startup drafter ownership before allocator
                    // cleanup and exact target expert-residency admission.
                    drop(draft_model);
                    draft_model_revision
                });
            model
                .runtime()
                .synchronize_gpu_stream_and_clear_allocator_cache()
                .map_err(qwen3_5_runtime_error)?;
            // Core, vision, and optional draft loading are complete. Only now is
            // active memory a stable baseline for exact complete-expert admission.
            model
                .try_promote_experts_to_resident(
                    Qwen3_5ExpertResidencyTransitionReason::Startup,
                    &mut model_loading_performance_attribution,
                )
                .map_err(qwen3_5_runtime_error)?;
            Ok((
                model,
                persistent_prompt_cache_model_contract,
                persistent_visual_embedding_model_contract,
                persistent_prompt_cache,
                speculative_prefill_draft_persistent_prompt_cache,
                speculative_prefill_draft_is_available,
                speculative_prefill_draft_supports_processed_visual_images,
                loaded_draft_model_revision,
                speculative_prefill_unavailable_reason,
            ))
        })();

        match model_loading_result {
            Ok((
                model,
                persistent_prompt_cache_model_contract,
                persistent_visual_embedding_model_contract,
                persistent_prompt_cache,
                speculative_prefill_draft_persistent_prompt_cache,
                speculative_prefill_draft_is_available,
                speculative_prefill_draft_supports_processed_visual_images,
                loaded_draft_model_revision,
                speculative_prefill_unavailable_reason,
            )) => {
                self.model_id = model_id.clone();
                self.model_revision = model_revision.clone();
                self.persistent_prompt_cache_model_contract =
                    persistent_prompt_cache_model_contract;
                self.persistent_visual_embedding_model_contract =
                    persistent_visual_embedding_model_contract;
                self.persistent_prompt_cache = persistent_prompt_cache;
                self.speculative_prefill_draft_persistent_prompt_cache =
                    speculative_prefill_draft_persistent_prompt_cache;
                self.speculative_prefill_draft_model = None;
                self.speculative_prefill_draft_model_revision = loaded_draft_model_revision;
                self.speculative_prefill_draft_is_available =
                    speculative_prefill_draft_is_available;
                self.speculative_prefill_draft_supports_processed_visual_images =
                    speculative_prefill_draft_supports_processed_visual_images;
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
                let (mtp_runtime_state, mtp_unavailable_reason, mtp_depth_status) =
                    qwen3_5_mtp_runtime_configuration_after_load(
                        self.mtp_enabled,
                        self.configured_mtp_draft_depth,
                        &qwen3_5_mtp_artifact_capability,
                        model_has_mtp_weights,
                    );
                self.mtp_runtime_state = mtp_runtime_state;
                self.mtp_unavailable_reason = mtp_unavailable_reason;
                self.mtp_depth_status = mtp_depth_status;
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
