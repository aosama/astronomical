use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

use super::super::bounded_expert_reader::load_quantized_expert_page;
use super::super::expert_cache::ExpertWeightMemoryCache;
use super::super::expert_cache_statistics::ExpertWeightMemoryCacheRequestReport;
use super::super::paged_expert_weights::build_prefixed_paged_expert_weights;
use super::super::quantized_expert_manifest::{
    QuantizedExpertPageManifest, build_quantized_expert_cache_population_manifest_from_plan,
    build_quantized_expert_page_manifest_from_plan,
};
use super::{ExpertPager, ExpertPagingError, PagedExpertWeights};

impl ExpertPager {
    /// Loads and retains missing one-expert pages, then assembles the selection.
    pub fn load_selected_experts_through_memory_cache(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        layer_index: usize,
        selected_expert_ids: &[usize],
        expert_weight_memory_cache: &mut ExpertWeightMemoryCache,
    ) -> Result<
        (
            PagedExpertWeights,
            QuantizedExpertPageManifest,
            ExpertWeightMemoryCacheRequestReport,
        ),
        ExpertPagingError,
    > {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.load_selected_experts_through_memory_cache_with_performance_attribution(
            runtime,
            layer_index,
            selected_expert_ids,
            expert_weight_memory_cache,
            &mut disabled_performance_attribution,
        )
    }

    pub(crate) fn load_selected_experts_through_memory_cache_with_performance_attribution(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        layer_index: usize,
        selected_expert_ids: &[usize],
        expert_weight_memory_cache: &mut ExpertWeightMemoryCache,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<
        (
            PagedExpertWeights,
            QuantizedExpertPageManifest,
            ExpertWeightMemoryCacheRequestReport,
        ),
        ExpertPagingError,
    > {
        let layer_plan = self.layer_plans.get(layer_index).ok_or({
            ExpertPagingError::LayerIndexOutOfRange {
                layer_index,
                layer_count: self.layer_plans.len(),
            }
        })?;
        let selected_page_manifest = performance_attribution.measure_operation(
            PerformanceOperation::ExpertPageManifestConstruction,
            |_performance_attribution| {
                build_quantized_expert_page_manifest_from_plan(layer_plan, selected_expert_ids)
            },
        )?;
        let did_demote_complete_layer_before_selected_route_admission = performance_attribution
            .measure_operation(
                PerformanceOperation::ExpertWeightMemoryCacheEviction,
                |_performance_attribution| {
                    expert_weight_memory_cache.reconcile_complete_layers_for_decode_route_floors()
                },
            );
        if did_demote_complete_layer_before_selected_route_admission {
            performance_attribution
                .measure_operation(
                    PerformanceOperation::MlxAllocatorCacheCleanup,
                    |_performance_attribution| {
                        runtime.synchronize_gpu_stream_and_clear_allocator_cache()
                    },
                )
                .map_err(|runtime_error| ExpertPagingError::Runtime {
                    description: runtime_error.to_string(),
                })?;
            let memory_budget_snapshot = performance_attribution.measure_operation(
                PerformanceOperation::ExpertPageMemoryBudgetSnapshot,
                |_performance_attribution| {
                    self.memory_budget.snapshot(
                        runtime,
                        &format!(
                            "expert_cache_before_selected_route_admission_layer_{layer_index}"
                        ),
                        selected_page_manifest.payload_byte_count,
                    )
                },
            )?;
            performance_attribution.measure_operation(
                PerformanceOperation::ExpertWeightMemoryCacheEviction,
                |_performance_attribution| {
                    expert_weight_memory_cache
                        .update_from_memory_budget_snapshot_while_protecting_selected_experts(
                            &memory_budget_snapshot,
                            layer_index,
                            selected_expert_ids,
                            selected_page_manifest.payload_byte_count,
                        );
                },
            );
        }
        // A user-supplied ceiling can be smaller than one routed selection. The
        // request can still run by loading a temporary page and retaining nothing.
        if !expert_weight_memory_cache.can_retain_selected_expert_payload(
            layer_index,
            selected_page_manifest.payload_byte_count,
        ) {
            return self.load_selected_experts_while_bypassing_memory_cache(
                runtime,
                layer_index,
                selected_expert_ids,
                expert_weight_memory_cache,
                performance_attribution,
            );
        }

        // Full hits avoid a live memory sample. MLX counter reads can synchronize
        // work, so only the slower miss path pays for dynamic budget calculation.
        let missing_expert_ids = selected_expert_ids
            .iter()
            .copied()
            .filter(|selected_expert_id| {
                expert_weight_memory_cache
                    .cached_expert(layer_index, *selected_expert_id)
                    .is_none()
            })
            .collect::<Vec<_>>();
        let cache_population_manifest = if missing_expert_ids.is_empty() {
            None
        } else {
            Some(performance_attribution.measure_operation(
                PerformanceOperation::ExpertPageManifestConstruction,
                |_performance_attribution| {
                    build_quantized_expert_cache_population_manifest_from_plan(
                        layer_plan,
                        &missing_expert_ids,
                    )
                },
            )?)
        };

        if let Some(cache_population_manifest) = cache_population_manifest.as_ref() {
            // This snapshot includes the pending disk page. It lets the cache
            // shrink existing retained experts before any new MLX arrays exist.
            let mut memory_budget_snapshot = performance_attribution.measure_operation(
                PerformanceOperation::ExpertPageMemoryBudgetSnapshot,
                |_performance_attribution| {
                    self.memory_budget.snapshot(
                        runtime,
                        &format!("expert_cache_batch_layer_{layer_index}"),
                        cache_population_manifest.payload_byte_count,
                    )
                },
            )?;
            performance_attribution.measure_operation(
                PerformanceOperation::ExpertWeightMemoryCacheEviction,
                |_performance_attribution| {
                    expert_weight_memory_cache
                        .update_from_memory_budget_snapshot_while_protecting_selected_experts(
                            &memory_budget_snapshot,
                            layer_index,
                            selected_expert_ids,
                            cache_population_manifest.payload_byte_count,
                        )
                },
            );
            let did_demote_complete_layer_for_decode_routes = performance_attribution
                .measure_operation(
                    PerformanceOperation::ExpertWeightMemoryCacheEviction,
                    |_performance_attribution| {
                        expert_weight_memory_cache
                            .reconcile_complete_layers_for_decode_route_floors()
                    },
                );
            if did_demote_complete_layer_for_decode_routes {
                performance_attribution
                    .measure_operation(
                        PerformanceOperation::MlxAllocatorCacheCleanup,
                        |_performance_attribution| {
                            runtime.synchronize_gpu_stream_and_clear_allocator_cache()
                        },
                    )
                    .map_err(|runtime_error| ExpertPagingError::Runtime {
                        description: runtime_error.to_string(),
                    })?;
                memory_budget_snapshot = performance_attribution.measure_operation(
                    PerformanceOperation::ExpertPageMemoryBudgetSnapshot,
                    |_performance_attribution| {
                        self.memory_budget.snapshot(
                            runtime,
                            &format!(
                                "expert_cache_batch_after_hybrid_demotion_layer_{layer_index}"
                            ),
                            cache_population_manifest.payload_byte_count,
                        )
                    },
                )?;
                performance_attribution.measure_operation(
                    PerformanceOperation::ExpertWeightMemoryCacheEviction,
                    |_performance_attribution| {
                        expert_weight_memory_cache
                            .update_from_memory_budget_snapshot_while_protecting_selected_experts(
                                &memory_budget_snapshot,
                                layer_index,
                                selected_expert_ids,
                                cache_population_manifest.payload_byte_count,
                            );
                    },
                );
            }

            if !expert_weight_memory_cache.can_retain_selected_expert_payload(
                layer_index,
                selected_page_manifest.payload_byte_count,
            ) {
                // The newly derived live ceiling cannot retain this routed set.
                // Remove old protected entries too, then serve this request through
                // a temporary direct page rather than failing generation.
                let live_maximum_resident_payload_byte_count = expert_weight_memory_cache
                    .statistics()
                    .maximum_resident_payload_byte_count;
                performance_attribution.measure_operation(
                    PerformanceOperation::ExpertWeightMemoryCacheEviction,
                    |_performance_attribution| {
                        expert_weight_memory_cache.update_maximum_resident_payload_byte_count(
                            live_maximum_resident_payload_byte_count,
                        );
                    },
                );
                return self.load_selected_experts_while_bypassing_memory_cache(
                    runtime,
                    layer_index,
                    selected_expert_ids,
                    expert_weight_memory_cache,
                    performance_attribution,
                );
            }

            if !memory_budget_snapshot.within_cap() {
                // Re-sample after eviction. This second check proves the retained
                // state is safe before the missing page is read from disk.
                performance_attribution.measure_operation(
                    PerformanceOperation::ExpertPageMemoryBudgetSnapshot,
                    |_performance_attribution| {
                        self.memory_budget.check(
                            runtime,
                            &format!("expert_cache_batch_after_eviction_layer_{layer_index}"),
                            cache_population_manifest.payload_byte_count,
                        )
                    },
                )?;
            }
        }

        let mut request_report = ExpertWeightMemoryCacheRequestReport::default();
        // Record hits before loading misses so every selected expert contributes
        // exactly once to request and cumulative counters.
        performance_attribution.measure_operation(
            PerformanceOperation::ExpertWeightMemoryCacheLookup,
            |_performance_attribution| {
                for &selected_expert_id in selected_expert_ids {
                    if expert_weight_memory_cache
                        .record_expert_access(layer_index, selected_expert_id)
                    {
                        request_report.cache_hit_count += 1;
                    } else {
                        request_report.cache_miss_count += 1;
                    }
                }
            },
        );
        if let Some(cache_population_manifest) = cache_population_manifest {
            // At this point the selected set fits. Evict only unselected entries
            // to make room for the exact missing payload that is about to load.
            let page_fits_after_eviction = performance_attribution.measure_operation(
                PerformanceOperation::ExpertWeightMemoryCacheEviction,
                |_performance_attribution| {
                    expert_weight_memory_cache.evict_oldest_unselected_experts_to_fit(
                        layer_index,
                        selected_expert_ids,
                        cache_population_manifest.payload_byte_count,
                    )
                },
            );
            if !page_fits_after_eviction {
                return Err(ExpertPagingError::Runtime {
                    description: format!(
                        "expert weight memory-cache layer {layer_index} cannot retain the selected expert payload within its byte budget"
                    ),
                });
            }
            let disk_batch_load_count = cache_population_manifest.source_manifests.len();
            let mut loaded_tensors = performance_attribution
                .measure_operation(
                    PerformanceOperation::ExpertBoundedSafetensorsLazyPageConstruction,
                    |performance_attribution| {
                        load_quantized_expert_page(
                            runtime,
                            &cache_population_manifest,
                            performance_attribution.positional_file_read_metrics(),
                        )
                    },
                )
                .map_err(|error| ExpertPagingError::Runtime {
                    description: error.to_string(),
                })?;
            expert_weight_memory_cache.record_disk_batch_loads(disk_batch_load_count);
            request_report.disk_batch_load_count += disk_batch_load_count;
            performance_attribution.record_counter(
                PerformanceCounter::ExpertPageLogicalPayloadBytes,
                cache_population_manifest.payload_byte_count,
            );

            for missing_expert_id in missing_expert_ids {
                // The batched safetensors read returns names prefixed by expert ID.
                // Split that map into complete one-expert owners before retention;
                // retaining slices of a larger page could keep the full page alive.
                let one_expert_weights = build_prefixed_paged_expert_weights(
                    &mut loaded_tensors,
                    layer_plan,
                    &format!("expert_{missing_expert_id}"),
                )?;
                expert_weight_memory_cache.record_disk_page_load();
                request_report.disk_page_load_count += 1;
                expert_weight_memory_cache.remember_expert(
                    layer_index,
                    missing_expert_id,
                    one_expert_weights,
                );
            }
        }
        let paged_expert_weights = performance_attribution.measure_operation(
            PerformanceOperation::ExpertWeightMemoryCachePageAssemblyGraphConstruction,
            |_performance_attribution| {
                expert_weight_memory_cache.assemble_selected_experts(
                    runtime,
                    layer_index,
                    selected_expert_ids,
                )
            },
        )?;
        Ok((paged_expert_weights, selected_page_manifest, request_report))
    }
}
