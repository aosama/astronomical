use astronomical_runtime_integration::MlxRuntime;
use std::sync::Arc;

use crate::{
    EngineLoadResult, InferenceEngineError, PerformanceAttribution, PerformanceAttributionOutcome,
    PerformanceOperation, PersistentPromptCacheDiskStore, PersistentPromptCacheModelContract,
    PersistentVisualEmbeddingModelContract,
};

use super::persistent_prompt_cache_startup_logging;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_core::model::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
use crate::qwen3_5_core::vision::Qwen3_5ImageProcessor;
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
                    .experimental_ssd_paging_prefill_graph_submission_layer_interval,
                self.chunking
                    .experimental_ssd_paging_generation_graph_submission_layer_interval,
            )
            .map_err(|configuration_error| {
                fatal_engine_error(format!(
                    "failed to validate model chunking configuration: {configuration_error}"
                ))
            })?;
            let model = Qwen3_5Model::load_with_performance_attribution(
                runtime,
                validated_artifact,
                &self.model_directory,
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
                        persistent_prompt_cache_startup_logging::log_persistent_prompt_cache_startup_cleanup(
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
            )) => {
                self.model_id = model_id.clone();
                self.model_revision = model_revision.clone();
                self.persistent_prompt_cache_model_contract =
                    persistent_prompt_cache_model_contract;
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
