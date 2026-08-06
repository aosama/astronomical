//! Complete-layer payload sizing and live retention admission.

use astronomical_runtime_integration::MlxRuntime;

use crate::expert_paging::{
    ExpertWeightMemoryCache, MemoryBudgetSnapshot, QuantizedExpertLayerPlan,
    build_quantized_expert_page_manifest_from_plan,
};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::expert_pager::{ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5PagedExpertWeights};

impl Qwen3_5ExpertPager {
    pub(crate) fn minimum_decode_route_payload_byte_count_by_layer(
        &self,
        experts_per_token: u32,
    ) -> Result<Vec<u64>, ExpertPagingError> {
        let selected_expert_count =
            usize::try_from(experts_per_token).map_err(|_| ExpertPagingError::Runtime {
                description: "experts per token exceed the host integer range".to_owned(),
            })?;
        let selected_expert_ids = (0..selected_expert_count).collect::<Vec<_>>();
        self.layer_plans
            .iter()
            .map(|layer_plan| {
                Ok(build_quantized_expert_page_manifest_from_plan(
                    layer_plan,
                    &selected_expert_ids,
                )?
                .payload_byte_count)
            })
            .collect()
    }

    pub(crate) fn prewarm_complete_layers_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        expert_weight_memory_cache: &std::cell::RefCell<
            ExpertWeightMemoryCache<Qwen3_5PagedExpertWeights>,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), ExpertPagingError> {
        for layer_index in 0..self.layer_plans.len() {
            if expert_weight_memory_cache
                .borrow()
                .has_complete_expert_layer(layer_index)
            {
                tracing::debug!(
                    layer_index,
                    "skipped complete expert-layer retention because the layer is already resident"
                );
                continue;
            }
            let (complete_layer_expert_payload_byte_count, complete_layer_memory_budget_snapshot) =
                self.complete_layer_retention_memory_budget_snapshot(
                    runtime,
                    layer_index,
                    performance_attribution,
                )?;
            let (can_retain_complete_layer, expert_weight_memory_cache_statistics) = {
                let mut expert_weight_memory_cache = expert_weight_memory_cache.borrow_mut();
                expert_weight_memory_cache
                    .update_from_memory_budget_snapshot_while_protecting_selected_experts(
                        &complete_layer_memory_budget_snapshot,
                        layer_index,
                        &[],
                        complete_layer_expert_payload_byte_count,
                    );
                let can_retain_complete_layer = complete_layer_memory_budget_snapshot.within_cap()
                    && expert_weight_memory_cache.can_retain_complete_layer_expert_payload(
                        layer_index,
                        complete_layer_expert_payload_byte_count,
                    );
                (
                    can_retain_complete_layer,
                    expert_weight_memory_cache.statistics(),
                )
            };
            if !can_retain_complete_layer {
                tracing::info!(
                    layer_index,
                    complete_layer_expert_payload_bytes = complete_layer_expert_payload_byte_count,
                    within_configured_mlx_cap = complete_layer_memory_budget_snapshot.within_cap(),
                    active_memory_bytes = complete_layer_memory_budget_snapshot.active_bytes,
                    allocator_cache_memory_bytes =
                        complete_layer_memory_budget_snapshot.allocator_cache_bytes,
                    pending_allocation_bytes =
                        complete_layer_memory_budget_snapshot.pending_allocation_bytes,
                    projected_memory_bytes = complete_layer_memory_budget_snapshot.projected_bytes,
                    configured_mlx_memory_cap_bytes =
                        complete_layer_memory_budget_snapshot.configured_cap_bytes,
                    retained_expert_payload_bytes =
                        expert_weight_memory_cache_statistics.resident_payload_byte_count,
                    maximum_retained_expert_payload_bytes =
                        expert_weight_memory_cache_statistics.maximum_resident_payload_byte_count,
                    retained_complete_layer_count =
                        expert_weight_memory_cache_statistics.complete_layer_count,
                    "deferred complete expert-layer retention because the live memory budget rejected it"
                );
                continue;
            }
            tracing::info!(
                layer_index,
                complete_layer_expert_payload_bytes = complete_layer_expert_payload_byte_count,
                retained_complete_layer_count_before =
                    expert_weight_memory_cache_statistics.complete_layer_count,
                "started loading an admitted complete expert layer"
            );
            let sparse_expert_ids =
                (0..self.layer_plans[layer_index].expert_capacity).collect::<Vec<_>>();
            let (complete_layer_expert_weights, _page_manifest, _memory_budget_snapshot) =
                match self.load_selected_experts_with_performance_attribution(
                    runtime,
                    layer_index,
                    &sparse_expert_ids,
                    None,
                    performance_attribution,
                ) {
                    Ok(complete_layer_load) => complete_layer_load,
                    Err(ExpertPagingError::MemoryBudget(memory_budget_error)) => {
                        tracing::info!(
                            layer_index,
                            error = %memory_budget_error,
                            "deferred complete expert-layer retention after the load-time memory budget changed"
                        );
                        continue;
                    }
                    Err(expert_paging_error) => return Err(expert_paging_error),
                };
            let mut complete_layer_expert_arrays = Vec::new();
            complete_layer_expert_weights
                .append_array_references(&mut complete_layer_expert_arrays);
            runtime
                .evaluate_arrays(&complete_layer_expert_arrays)
                .map_err(|runtime_error| ExpertPagingError::Runtime {
                    description: runtime_error.to_string(),
                })?;
            expert_weight_memory_cache
                .borrow_mut()
                .remember_complete_layer_expert_weights(layer_index, complete_layer_expert_weights);
            let retained_expert_statistics = expert_weight_memory_cache.borrow().statistics();
            tracing::info!(
                completed_layer_count = retained_expert_statistics.complete_layer_count,
                total_layer_count = self.layer_plans.len(),
                retained_expert_payload_bytes =
                    retained_expert_statistics.resident_payload_byte_count,
                "auto expert-memory prewarm retained a complete layer"
            );
        }
        Ok(())
    }

    pub(crate) fn complete_layer_retention_memory_budget_snapshot(
        &self,
        runtime: &MlxRuntime,
        layer_index: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(u64, MemoryBudgetSnapshot), ExpertPagingError> {
        let layer_plan = self.layer_plans.get(layer_index).ok_or({
            ExpertPagingError::LayerIndexOutOfRange {
                layer_index,
                layer_count: self.layer_plans.len(),
            }
        })?;
        let complete_layer_expert_payload_byte_count =
            complete_layer_expert_payload_byte_count(layer_plan, layer_index)?;
        let memory_budget_snapshot = performance_attribution.measure_operation(
            PerformanceOperation::ExpertPageMemoryBudgetSnapshot,
            |_performance_attribution| {
                self.memory_budget.snapshot(
                    runtime,
                    &format!("complete_expert_layer_retention_{layer_index}"),
                    complete_layer_expert_payload_byte_count,
                )
            },
        )?;
        Ok((
            complete_layer_expert_payload_byte_count,
            memory_budget_snapshot,
        ))
    }
}

pub(super) fn complete_layer_expert_payload_byte_count(
    layer_plan: &QuantizedExpertLayerPlan,
    layer_index: usize,
) -> Result<u64, ExpertPagingError> {
    layer_plan.tensor_sources.iter().try_fold(
        0u64,
        |accumulated_payload_byte_count, tensor_source| {
            let bytes_per_expert = u64::try_from(tensor_source.bytes_per_expert).map_err(|_| {
                ExpertPagingError::Runtime {
                    description: format!(
                        "complete expert layer {layer_index} bytes per expert exceed the supported byte range"
                    ),
                }
            })?;
            let expert_capacity = u64::try_from(tensor_source.expert_capacity).map_err(|_| {
                ExpertPagingError::Runtime {
                    description: format!(
                        "complete expert layer {layer_index} capacity exceeds the supported byte range"
                    ),
                }
            })?;
            let tensor_payload_byte_count = bytes_per_expert
                .checked_mul(expert_capacity)
                .ok_or_else(|| payload_byte_count_overflow_error(layer_index))?;
            accumulated_payload_byte_count
                .checked_add(tensor_payload_byte_count)
                .ok_or_else(|| payload_byte_count_overflow_error(layer_index))
        },
    )
}

fn payload_byte_count_overflow_error(layer_index: usize) -> ExpertPagingError {
    ExpertPagingError::Runtime {
        description: format!("complete expert layer {layer_index} payload byte count overflowed"),
    }
}
