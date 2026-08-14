//! Barrier-safe fill of globally ranked demand-selected expert pages.
//!
//! # What this file does not do
//!
//! It does not restore complete residency. That is
//! `try_promote_experts_to_resident`. If the complete owner already exists,
//! this fill returns zero new pages so payload is not double-counted.
//!
//! # What this file does
//!
//! Multi-token prefill streams one complete layer, uses it, and drops it.
//! After prefill, activations shrink. The leftover composed decode budget
//! (`retained_expert_budget_bytes`) may then pin the experts this prompt
//! actually routed to, ranked by how often they appeared.
//!
//! After request-pressure demotion that leftover is still the decode fill
//! budget. The temporary pressure cap must already have been released by
//! decode handoff; otherwise this fill sees about one gigabyte and rejects
//! every useful page.
//!
//! # Safe call boundary
//!
//! Call this only after a synchronization/cleanup barrier where operation-local
//! arrays are no longer competing with refill. Loading a page creates lazy
//! MLX arrays and evaluating them materializes payload, so running this in the
//! middle of a forward could invalidate the memory proof that admitted that
//! forward.
//!
//! # Why route-frequency pages
//!
//! Every decode token visits every decoder layer, and a retained page is useful
//! only when it covers that layer's routed set. Measured 23 GB journeys were
//! closest to the source-read gate when experts were ranked by route frequency.
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
    /// Fills demand-selected expert pages for decode reuse.
    ///
    /// `context_token_count` is the prompt length. The decode plan uses it to
    /// reserve key/value growth so pages do not steal the leftover ceiling
    /// from later tokens.
    ///
    /// `requested_retained_expert_payload_bytes` is an optional smaller caller
    /// cap. Decode handoff passes `u64::MAX` so the composed leftover budget
    /// wins. Do not pass "one routed page times layer count" after pressure;
    /// that is the old 1 GB working-set trap.
    ///
    /// Best-effort: returns the number of newly retained layers. Callers must
    /// not fail the user request if warm fill is skipped or partially admitted.
    pub(crate) fn fill_retained_expert_pages(
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
        // slot. After prefill that leftover is the decode fill budget, including
        // after request-pressure demotion. Promotion of the complete owner is a
        // separate admit and must not be impersonated by this page fill.
        let decoder_layer_count = expert_pager.layer_count();
        let composed_retained_expert_fill_budget_bytes =
            planned_budget.retained_expert_budget_bytes;
        let retained_expert_fill_budget_bytes = crate::retained_expert_fill_budget_bytes(
            composed_retained_expert_fill_budget_bytes,
            requested_retained_expert_payload_bytes,
        );
        if retained_expert_fill_budget_bytes == composed_retained_expert_fill_budget_bytes {
            tracing::info!(
                composed_retained_expert_fill_budget_bytes,
                retained_expert_fill_budget_bytes,
                complete_residency_fits = planned_budget.complete_residency_fits,
                "decode-warm fill using composed retained-expert budget after prefill"
            );
        }
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

        // Decode visits every decoder layer. Rank observed route frequency so
        // the retained budget keeps the experts that appeared most often.
        let mut newly_retained_page_count = 0usize;
        let preferred_expert_ids_by_layer = performance_attribution.measure_operation(
            PerformanceOperation::RetainedExpertPagePlanning,
            |_performance_attribution| {
                let layer_payloads = expert_pager
                    .layer_plans()
                    .iter()
                    .map(|layer_plan| {
                        layer_plan
                            .complete_expert_payload_byte_count()
                            .map_err(ExpertPagingError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let expert_capacities = expert_pager
                    .layer_plans()
                    .iter()
                    .map(|layer_plan| layer_plan.expert_capacity)
                    .collect::<Vec<_>>();
                retained_expert_layers
                    .borrow()
                    .preferred_expert_ids_for_global_budget(&layer_payloads, &expert_capacities)
                    .map_err(|planner_error| ExpertPagingError::InvalidPagingPlan {
                        description: planner_error.to_string(),
                    })
                    .map_err(Qwen3_5ExecutionError::from)
            },
        )?;

        // Retire every page whose desired selection changed before loading any
        // replacement. Otherwise an earlier growing layer can be rejected while
        // bytes assigned away from a later stale layer are still resident.
        {
            let mut retained_expert_layers = retained_expert_layers.borrow_mut();
            for (layer_index, preferred_expert_ids) in
                preferred_expert_ids_by_layer.iter().enumerate()
            {
                let should_remove_stale_layer = retained_expert_layers
                    .retained_layer(layer_index)
                    .is_some_and(|retained_layer| {
                        !retained_layer.has_exact_expert_ids(preferred_expert_ids)
                    });
                if should_remove_stale_layer {
                    retained_expert_layers.remove_layer(layer_index);
                }
            }
        }

        for layer_index in 0..decoder_layer_count {
            let preferred_expert_ids = preferred_expert_ids_by_layer[layer_index].clone();
            if preferred_expert_ids.is_empty() {
                continue;
            }
            if retained_expert_layers
                .borrow()
                .retained_layer(layer_index)
                .is_some_and(|retained_layer| {
                    retained_layer.has_exact_expert_ids(&preferred_expert_ids)
                })
            {
                continue;
            }
            let (retained_weights, page_manifest) = performance_attribution.measure_operation(
                PerformanceOperation::RustExpertStreamingLayerPreparation,
                |performance_attribution| {
                    expert_pager.load_rust_streamed_expert_layer(
                        &self.runtime,
                        layer_index,
                        &preferred_expert_ids,
                        performance_attribution,
                    )
                },
            )?;

            let mut retained_arrays = Vec::new();
            retained_weights.append_array_references(&mut retained_arrays);
            // SafeTensors arrays are lazy. Evaluate before publishing to guarantee
            // that `retained_layer == Some` means all payload is materialized and
            // any read/allocation error occurs while ownership is still local.
            self.runtime.evaluate_arrays(&retained_arrays)?;

            retained_expert_layers.borrow_mut().record_disk_load(
                preferred_expert_ids.len(),
                page_manifest.source_manifests.len(),
            );
            let candidate_layer_payload_bytes = page_manifest.payload_byte_count;
            let did_retain = retained_expert_layers.borrow_mut().replace_layer(
                layer_index,
                Qwen3_5RetainedExpertLayer {
                    weights: retained_weights,
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
                continue;
            }
            newly_retained_page_count = newly_retained_page_count.saturating_add(1);
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

        if newly_retained_page_count > 0 {
            tracing::info!(
                newly_retained_page_count,
                retained_expert_budget_bytes = planned_budget.retained_expert_budget_bytes,
                retained_expert_fill_budget_bytes,
                context_token_count,
                "filled demand-selected expert pages under retained-expert budget"
            );
        }
        // The current topology consumed this evidence. Decode and the next
        // request now build a fresh window so an old prompt cannot dominate
        // retention forever in a long-lived server.
        retained_expert_layers.borrow_mut().clear_expert_demand();
        Ok(newly_retained_page_count)
    }

    /// Applies the last-chunk or ordinary demand multiplier for later recordings.
    pub(crate) fn set_expert_demand_assignment_weight(&self, assignment_weight: u64) {
        if let Some(retained_expert_layers) = self.retained_expert_layers.as_ref() {
            retained_expert_layers
                .borrow_mut()
                .set_demand_assignment_weight(assignment_weight);
        }
    }
}
