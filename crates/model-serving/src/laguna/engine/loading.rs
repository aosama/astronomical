//! Laguna model loading performed on the single MLX owner thread.

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::laguna::startup::weight_loader::load_laguna_bindable_tensors;
use crate::laguna::{LagunaModel, LagunaNativeWeights};
use crate::{
    EngineLoadResult, InferenceEngineError, MlxRamBudgetPhase,
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution,
    PerformanceAttributionOutcome, PerformanceOperation,
};

use super::execution::LagunaInferenceExecution;
use super::memory::laguna_ram_budget_snapshot;

impl LagunaInferenceExecution {
    /// Loads deferred tensors, expert paging, cache storage, and memory policy.
    pub(super) fn load_model_on_owner_thread(
        &mut self,
    ) -> Result<EngineLoadResult, InferenceEngineError> {
        let Some(mut pending_startup) = self.pending_startup.take() else {
            return Ok(EngineLoadResult::new().with_expert_memory_mode(
                self.model.as_ref().map(LagunaModel::expert_memory_mode),
            ));
        };
        let memory_limits = MlxMemoryLimits::new(
            pending_startup.effective_mlx_memory_ceiling_bytes,
            pending_startup.allocator_cache_memory_limit_bytes,
        )
        .map_err(|_| InferenceEngineError::Fatal {
            reason: "Laguna runtime memory limits are invalid".to_owned(),
        })?;
        let runtime = pending_startup
            .model_loading_performance_attribution
            .measure_operation(PerformanceOperation::MlxRuntimeInitialization, |_| {
                MlxRuntime::initialize(memory_limits)
            })
            .map_err(|_| InferenceEngineError::Fatal {
                reason: "Laguna runtime initialization failed".to_owned(),
            })?;
        let shard_files = std::mem::take(&mut pending_startup.shard_files);
        let tensors = pending_startup
            .model_loading_performance_attribution
            .measure_operation(PerformanceOperation::ModelSafetensorsMapping, |_| {
                load_laguna_bindable_tensors(
                    &runtime,
                    &pending_startup.tensor_contract,
                    &pending_startup.target_contract,
                    shard_files,
                    pending_startup.load_routed_experts,
                )
            })
            .map_err(|_| InferenceEngineError::Fatal {
                reason: "Laguna weight loading failed".to_owned(),
            })?;
        let weights = pending_startup
            .model_loading_performance_attribution
            .measure_operation(PerformanceOperation::ModelTensorBinding, |_| {
                LagunaNativeWeights::bind(&runtime, tensors, &pending_startup.target_contract)
            })
            .map_err(|_| InferenceEngineError::Fatal {
                reason: "Laguna weight binding failed".to_owned(),
            })?;
        let mut model =
            LagunaModel::new(pending_startup.target_contract, weights).map_err(|_| {
                InferenceEngineError::Fatal {
                    reason: "Laguna model construction failed".to_owned(),
                }
            })?;
        if !pending_startup.paging_plan.sparse_layers().is_empty() {
            model = model
                .with_paging_plan(pending_startup.paging_plan)
                .map_err(|_| InferenceEngineError::Fatal {
                    reason: "Laguna paging-plan attachment failed".to_owned(),
                })?;
            if !pending_startup.load_routed_experts {
                let loaded_core_memory_snapshot =
                    runtime
                        .memory_snapshot()
                        .map_err(|_| InferenceEngineError::Fatal {
                            reason: "Laguna loaded-core MLX memory could not be measured"
                                .to_owned(),
                        })?;
                let measured_loaded_core_payload_bytes =
                    u64::try_from(loaded_core_memory_snapshot.active_memory_bytes())
                        .unwrap_or(u64::MAX);
                let mut measured_model_geometry = pending_startup.mlx_ram_budget.model_geometry();
                measured_model_geometry.model_core_payload_bytes = measured_model_geometry
                    .model_core_payload_bytes
                    .max(measured_loaded_core_payload_bytes);
                // Packed tensors can understate active MLX ownership after binding
                // expands or aligns arrays. Charge measured ownership before paging.
                pending_startup
                    .mlx_ram_budget
                    .update_model_geometry(measured_model_geometry);
                let initial_ram_budget = laguna_ram_budget_snapshot(
                    &pending_startup.mlx_ram_budget,
                    MlxRamBudgetPhase::Prefill,
                    0,
                );
                tracing::info!(
                    mlx_ceiling_bytes = initial_ram_budget.mlx_active_memory_ceiling_bytes,
                    model_core_payload_bytes = initial_ram_budget.model_core_payload_bytes,
                    context_window_reserve_bytes = initial_ram_budget.context_window_reserve_bytes,
                    activation_headroom_bytes = initial_ram_budget.activation_headroom_bytes,
                    complete_layer_stream_slot_bytes =
                        initial_ram_budget.complete_layer_stream_slot_bytes,
                    request_operational_reserve_bytes = initial_ram_budget.other_fixed_bytes,
                    retained_expert_budget_bytes = initial_ram_budget.retained_expert_budget_bytes,
                    "Laguna composed initial paged-model RAM budget"
                );
                model = model
                    .with_retained_expert_ceiling(initial_ram_budget.retained_expert_budget_bytes)
                    .map_err(|_| InferenceEngineError::Fatal {
                        reason: "Laguna retained-expert ceiling failed".to_owned(),
                    })?;
            }
        }
        let expert_memory_mode = model.expert_memory_mode();
        self.persistent_prompt_cache_disk_store_config =
            pending_startup.prompt_cache_disk_store_config.clone();
        self.persistent_prompt_cache = if let Some(prompt_cache_disk_store_config) =
            pending_startup.prompt_cache_disk_store_config
        {
            let mut cache_open_attribution = PerformanceAttribution::disabled();
            Some(
                super::prompt_cache::open_prompt_cache_store(
                    pending_startup.effective_mlx_memory_ceiling_bytes,
                    model.contract(),
                    &pending_startup.prompt_cache_model_id,
                    &pending_startup.prompt_cache_model_revision,
                    pending_startup.configured_prompt_cache_block_token_count,
                    pending_startup.prompt_cache_common_prefix_stride_blocks,
                    prompt_cache_disk_store_config,
                    &mut cache_open_attribution,
                )
                .map_err(|_| InferenceEngineError::Fatal {
                    reason: "required Laguna prompt cache initialization failed".to_owned(),
                })?,
            )
        } else {
            None
        };
        self.runtime = Some(runtime);
        self.model = Some(model);
        self.mlx_ram_budget = Some(pending_startup.mlx_ram_budget);
        self.prompt_processing_chunk_sizer = Some(pending_startup.prompt_processing_chunk_sizer);
        self.attribution_model_id = Some(pending_startup.attribution_model_id.clone());
        self.attribution_model_revision = Some(pending_startup.attribution_model_revision.clone());
        let loaded_memory_snapshot = self
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.memory_snapshot().ok());
        if let Some(model_loading_report) = pending_startup
            .model_loading_performance_attribution
            .finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
                outcome: PerformanceAttributionOutcome::Success,
                model_id: Some(pending_startup.attribution_model_id),
                model_revision: Some(pending_startup.attribution_model_revision),
                prefill_transient_observation_completed: false,
                prefill_observed_transient_high_water_bytes: 0,
                total_artifact_payload_bytes: Some(pending_startup.total_artifact_payload_bytes),
                resident_model_payload_bytes: loaded_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                model_shard_count: Some(pending_startup.model_shard_count),
                mlx_active_memory_bytes: loaded_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_allocator_cache_memory_bytes: loaded_memory_snapshot.as_ref().and_then(
                    |snapshot| u64::try_from(snapshot.allocator_cache_memory_bytes()).ok(),
                ),
                mlx_peak_memory_bytes: loaded_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
                failure_description: None,
            })
            && let Err(write_error) = pending_startup
                .performance_attribution_log
                .record(&model_loading_report)
        {
            tracing::warn!(error = %write_error, "Laguna model-load attribution could not be recorded");
        }
        self.performance_attribution_log = pending_startup.performance_attribution_log;
        Ok(EngineLoadResult::new()
            .with_expert_memory_mode(Some(expert_memory_mode))
            .with_minimum_mlx_memory_ceiling_bytes(self.minimum_mlx_memory_ceiling_bytes))
    }
}
