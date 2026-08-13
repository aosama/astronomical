//! Barrier-safe fill of complete expert layers under the retained-expert budget.
//!
//! Multi-token prefill stays operation-local. After prefill, activations shrink and
//! leftover `retained_expert_budget_bytes` may pin a deterministic complete-layer
//! prefix for decode and later prompt processing. Live pressure may shrink it.
//!
//! # Safe call boundary
//!
//! Call this only after a synchronization/cleanup barrier where operation-local
//! arrays are no longer competing with refill. Loading a complete layer creates
//! lazy MLX arrays and evaluating them materializes payload, so running this in the
//! middle of a forward could invalidate the memory proof that admitted that
//! forward.
//!
//! # Why a complete-layer prefix
//!
//! The current deterministic policy scans layer indices from zero upward and
//! stops at the first layer that does not fit. The cache evicts in reverse order,
//! so growth and shrink preserve one contiguous prefix. This gives predictable
//! ownership and zero route-time lookup policy. It is not evidence that low-index
//! layers are universally the most valuable; changing selection belongs to a
//! separately measured policy.
//!
//! # Best-effort does not mean unaccounted
//!
//! Refill failure must not fail a completed user request, but every candidate is
//! still checked against exact startup-validated payload bytes before I/O. The
//! cache repeats the check when ownership is committed, protecting against a
//! budget change between planning and publication.

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{MlxRamBudgetPhase, PerformanceAttribution, PerformanceOperation};

use super::super::expert_paging::expert_pager::{ExpertPagingError, Qwen3_5RetainedExpertLayer};

impl Qwen3_5Model {
    /// Fills complete expert layers for prefill, decode, or idle reuse.
    ///
    /// Best-effort: returns the number of newly retained layers. Callers should not
    /// fail the user request if warm fill is skipped or partially admitted.
    pub(crate) fn fill_retained_complete_expert_layers(
        &self,
        context_token_count: u64,
        requested_retained_expert_payload_bytes: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<usize, Qwen3_5ExecutionError> {
        if self.resident_expert_weights.is_some() {
            // Complete-model residency already owns every expert. Creating a
            // second retained-layer representation would double-count payload.
            return Ok(0);
        }
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(0);
        };
        let Some(retained_expert_layers) = self.retained_expert_layers.as_ref() else {
            return Ok(0);
        };

        let planned_budget = self.mlx_ram_budget.borrow().plan(
            MlxRamBudgetPhase::Decode,
            context_token_count,
            0,
            false,
        );
        if !planned_budget.may_grow_retained_expert_layers
            || planned_budget.retained_expert_budget_bytes == 0
        {
            return Ok(0);
        }

        // The composed budget has already subtracted model core, learned
        // context reserve, activation headroom, and one complete-layer loading
        // slot. A second fixed cap would strand usable RAM on larger machines
        // and violate the single-source adaptive policy.
        let retained_expert_fill_budget_bytes = planned_budget
            .retained_expert_budget_bytes
            .min(requested_retained_expert_payload_bytes);
        retained_expert_layers
            .borrow_mut()
            .update_maximum_resident_payload_bytes(retained_expert_fill_budget_bytes);
        // Detailed composition logs are gated by performance attribution. Normal
        // serving avoids the extra memory snapshot and verbose per-layer records.
        if performance_attribution.is_enabled() {
            let memory_snapshot = self.runtime.memory_snapshot()?;
            let retained_statistics = retained_expert_layers.borrow().statistics();
            tracing::info!(
                phase = ?MlxRamBudgetPhase::Decode,
                context_token_count,
                requested_retained_expert_payload_bytes,
                mlx_active_memory_ceiling_bytes = planned_budget.mlx_active_memory_ceiling_bytes,
                current_active_memory_bytes = memory_snapshot.active_memory_bytes(),
                allocator_cache_memory_bytes = memory_snapshot.allocator_cache_memory_bytes(),
                model_core_payload_bytes = planned_budget.model_core_payload_bytes,
                context_window_reserve_bytes = planned_budget.context_window_reserve_bytes,
                activation_headroom_bytes = planned_budget.activation_headroom_bytes,
                complete_layer_stream_slot_bytes = planned_budget.complete_layer_stream_slot_bytes,
                other_fixed_bytes = planned_budget.other_fixed_bytes,
                retained_expert_budget_bytes = planned_budget.retained_expert_budget_bytes,
                retained_expert_fill_budget_bytes,
                current_retained_expert_payload_bytes = retained_statistics.resident_payload_byte_count,
                maximum_retained_expert_payload_bytes = retained_statistics.maximum_resident_payload_byte_count,
                complete_expert_payload_bytes = self.mlx_ram_budget.borrow().model_geometry().complete_expert_payload_bytes,
                complete_residency_fits = planned_budget.complete_residency_fits,
                "retained expert fill budget decision"
            );
        }

        // Let the cache admit the exact heterogeneous payload of each layer.
        // Dividing by the largest layer size would reject smaller later layers
        // even though their real payload still fits the byte budget.
        let decoder_layer_count = expert_pager.layer_count();
        let mut newly_retained_layer_count = 0usize;

        for layer_index in 0..decoder_layer_count {
            if retained_expert_layers
                .borrow()
                .retained_layer(layer_index)
                .is_some()
            {
                // Preserve existing warm ownership and continue scanning. Request
                // pressure can evict only a suffix, but this also remains correct
                // if a future validated policy leaves an interior layer resident.
                continue;
            }
            // Obtain exact payload from immutable startup geometry. This query
            // performs no SafeTensors payload read and is therefore safe before
            // the admission decision.
            let layer_plan = expert_pager.layer_plan(layer_index)?;
            let candidate_layer_payload_bytes = layer_plan
                .complete_expert_payload_byte_count()
                .map_err(ExpertPagingError::from)?;
            let retained_statistics_before_candidate = retained_expert_layers.borrow().statistics();
            if !retained_expert_layers
                .borrow()
                .can_retain_additional_payload_bytes(candidate_layer_payload_bytes)
            {
                if performance_attribution.is_enabled() {
                    tracing::info!(
                        layer_index,
                        candidate_layer_payload_bytes,
                        current_retained_expert_payload_bytes =
                            retained_statistics_before_candidate.resident_payload_byte_count,
                        maximum_retained_expert_payload_bytes =
                            retained_statistics_before_candidate
                                .maximum_resident_payload_byte_count,
                        remaining_retained_expert_budget_bytes =
                            retained_statistics_before_candidate
                                .maximum_resident_payload_byte_count
                                .saturating_sub(
                                    retained_statistics_before_candidate
                                        .resident_payload_byte_count
                                ),
                        rejection_reason = "exact_layer_payload_exceeds_remaining_budget",
                        "retained expert layer candidate decision"
                    );
                }
                break;
            }
            if performance_attribution.is_enabled() {
                tracing::info!(
                    layer_index,
                    candidate_layer_payload_bytes,
                    current_retained_expert_payload_bytes =
                        retained_statistics_before_candidate.resident_payload_byte_count,
                    maximum_retained_expert_payload_bytes =
                        retained_statistics_before_candidate.maximum_resident_payload_byte_count,
                    remaining_retained_expert_budget_bytes = retained_statistics_before_candidate
                        .maximum_resident_payload_byte_count
                        .saturating_sub(
                            retained_statistics_before_candidate.resident_payload_byte_count
                        ),
                    decision = "load_and_retain",
                    "retained expert layer candidate decision"
                );
            }
            let complete_layer_expert_ids = (0..layer_plan.expert_capacity).collect::<Vec<_>>();
            // Passing every identifier turns the same bounded routed-page loader
            // used by decode into an exact complete-layer load. This avoids a
            // second storage implementation and preserves source dtype/quantization.
            let (complete_layer_weights, page_manifest) = performance_attribution
                .measure_operation(
                    PerformanceOperation::RustExpertStreamingLayerPreparation,
                    |performance_attribution| {
                        expert_pager.load_rust_streamed_expert_layer(
                            &self.runtime,
                            layer_index,
                            &complete_layer_expert_ids,
                            performance_attribution,
                        )
                    },
                )?;

            let mut complete_layer_arrays = Vec::new();
            complete_layer_weights.append_array_references(&mut complete_layer_arrays);
            // SafeTensors arrays are lazy. Evaluate before publishing to guarantee
            // that `retained_layer == Some` means all payload is materialized and
            // any read/allocation error occurs while ownership is still local.
            self.runtime.evaluate_arrays(&complete_layer_arrays)?;

            retained_expert_layers.borrow_mut().record_disk_load(
                complete_layer_expert_ids.len(),
                page_manifest.source_manifests.len(),
            );
            let did_retain = retained_expert_layers.borrow_mut().retain_complete_layer(
                layer_index,
                Qwen3_5RetainedExpertLayer {
                    weights: complete_layer_weights,
                    manifest: page_manifest,
                },
            );
            if !did_retain {
                // A live ceiling change or request-pressure cap may race between
                // the pre-I/O check and commit. Dropping the local page is safe;
                // the cache never observed partial ownership.
                if performance_attribution.is_enabled() {
                    tracing::info!(
                        layer_index,
                        candidate_layer_payload_bytes,
                        rejection_reason = "cache_rejected_after_load",
                        "retained expert layer candidate decision"
                    );
                }
                break;
            }
            newly_retained_layer_count = newly_retained_layer_count.saturating_add(1);
            if performance_attribution.is_enabled() {
                let memory_snapshot_after_retention = self.runtime.memory_snapshot()?;
                let retained_statistics_after_candidate =
                    retained_expert_layers.borrow().statistics();
                tracing::info!(
                    layer_index,
                    candidate_layer_payload_bytes,
                    active_memory_bytes_after_retention =
                        memory_snapshot_after_retention.active_memory_bytes(),
                    allocator_cache_memory_bytes_after_retention =
                        memory_snapshot_after_retention.allocator_cache_memory_bytes(),
                    retained_expert_payload_bytes_after_retention =
                        retained_statistics_after_candidate.resident_payload_byte_count,
                    remaining_retained_expert_budget_bytes = retained_statistics_after_candidate
                        .maximum_resident_payload_byte_count
                        .saturating_sub(
                            retained_statistics_after_candidate.resident_payload_byte_count
                        ),
                    "retained expert layer admitted"
                );
            }
        }

        if newly_retained_layer_count > 0 {
            tracing::info!(
                newly_retained_layer_count,
                retained_expert_budget_bytes = planned_budget.retained_expert_budget_bytes,
                retained_expert_fill_budget_bytes,
                context_token_count,
                "filled complete expert layers under retained-expert budget"
            );
        }
        Ok(newly_retained_layer_count)
    }
}
