use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

use super::super::bounded_expert_reader::load_quantized_expert_page;
use super::super::expert_cache::ExpertWeightMemoryCache;
use super::super::memory_budget::MemoryBudgetSnapshot;
use super::super::paged_expert_weights::build_paged_expert_weights;
use super::super::quantized_expert_manifest::{
    QuantizedExpertPageManifest, build_quantized_expert_page_manifest_from_plan,
};
use super::{ExpertPager, ExpertPagingError, PagedExpertWeights};

impl ExpertPager {
    /// Loads a sorted, unique expert selection after fail-closed Metal admission.
    pub fn load_selected_experts(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        layer_index: usize,
        selected_expert_ids: &[usize],
    ) -> Result<
        (
            PagedExpertWeights,
            QuantizedExpertPageManifest,
            MemoryBudgetSnapshot,
        ),
        ExpertPagingError,
    > {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.load_selected_experts_with_performance_attribution(
            runtime,
            layer_index,
            selected_expert_ids,
            None,
            &mut disabled_performance_attribution,
        )
    }

    pub(crate) fn load_selected_experts_with_performance_attribution(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        layer_index: usize,
        selected_expert_ids: &[usize],
        mut expert_weight_memory_cache: Option<&mut ExpertWeightMemoryCache>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<
        (
            PagedExpertWeights,
            QuantizedExpertPageManifest,
            MemoryBudgetSnapshot,
        ),
        ExpertPagingError,
    > {
        let layer_plan = self.layer_plans.get(layer_index).ok_or({
            ExpertPagingError::LayerIndexOutOfRange {
                layer_index,
                layer_count: self.layer_plans.len(),
            }
        })?;
        let page_manifest = performance_attribution.measure_operation(
            PerformanceOperation::ExpertPageManifestConstruction,
            |_performance_attribution| {
                build_quantized_expert_page_manifest_from_plan(layer_plan, selected_expert_ids)
            },
        )?;
        let mut budget_snapshot = performance_attribution.measure_operation(
            PerformanceOperation::ExpertPageMemoryBudgetSnapshot,
            |_performance_attribution| {
                self.memory_budget.snapshot(
                    runtime,
                    &format!("expert_page_layer_{layer_index}"),
                    page_manifest.payload_byte_count,
                )
            },
        )?;
        if !budget_snapshot.within_cap() {
            if let Some(expert_weight_memory_cache) = expert_weight_memory_cache.as_deref_mut() {
                performance_attribution.measure_operation(
                    PerformanceOperation::ExpertWeightMemoryCacheEviction,
                    |_performance_attribution| {
                        expert_weight_memory_cache
                            .reconcile_retention_before_temporary_expert_page(
                                &budget_snapshot,
                                layer_index,
                                selected_expert_ids,
                            );
                    },
                );
            }
            budget_snapshot = performance_attribution.measure_operation(
                PerformanceOperation::ExpertPageMemoryBudgetSnapshot,
                |_performance_attribution| {
                    self.memory_budget.check(
                        runtime,
                        &format!("expert_page_after_retention_reconciliation_layer_{layer_index}"),
                        page_manifest.payload_byte_count,
                    )
                },
            )?;
        }
        let paged_expert_weights = performance_attribution
            .measure_operation(
                PerformanceOperation::ExpertBoundedSafetensorsLazyPageConstruction,
                |performance_attribution| {
                    let mut loaded_tensors = load_quantized_expert_page(
                        runtime,
                        &page_manifest,
                        performance_attribution.positional_file_read_metrics(),
                    )
                    .map_err(|load_error| ExpertPagingError::Runtime {
                        description: load_error.to_string(),
                    })?;
                    build_paged_expert_weights(&mut loaded_tensors, layer_plan)
                },
            )
            .map_err(|error| ExpertPagingError::Runtime {
                description: error.to_string(),
            })?;
        performance_attribution.record_counter(
            PerformanceCounter::ExpertPageLogicalPayloadBytes,
            page_manifest.payload_byte_count,
        );
        Ok((paged_expert_weights, page_manifest, budget_snapshot))
    }
}
