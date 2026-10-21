//! Prepares pager-backed expert retention for decode after prompt prefill.
//!
//! Prefill can temporarily cap retained pages to protect prompt-processing
//! headroom. Leaving that request-scoped cap in place would prevent decode from
//! using its available memory to retain routed experts. Reconciliation is
//! best-effort because missing routes remain correct through operation-local
//! streaming.

use astronomical_ipc_protocol::RequestId;

use crate::{AdaptiveRamGrowthGuard, InferenceEngineError, MemoryPhase, PerformanceOperation};

use super::Qwen3_5EngineState;
use crate::qwen3_5_streaming::model::Qwen3_5Model;

impl Qwen3_5EngineState {
    /// Reconciles retained ownership with the leftover ceiling after prefill.
    ///
    /// Call this exactly once, after the last prefill chunk has synchronized and
    /// cleaned allocator storage, and before the first decode forward. The
    /// request flag `generation_residency_preparation_attempted` is the one-shot guard.
    pub(super) fn prepare_decode_expert_residency_after_prefill(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<(), InferenceEngineError> {
        let Some(model) = self.model.as_mut() else {
            return Err(super::fatal_engine_error(
                "Qwen3.5 engine lost its loaded model",
            ));
        };
        // Prefill pressure protects the remaining prompt by installing a
        // temporary retained-page ceiling. That cap must die here. Decode uses
        // a smaller activation footprint, so the leftover composed budget is
        // the user's real grant again. Leaving the cap in place was the bug
        // that kept generation at about one gigabyte of experts.
        let resumed_after_prefill_memory_pressure =
            model.resume_expert_retention_after_request_memory_pressure();
        if resumed_after_prefill_memory_pressure {
            tracing::info!(
                request_id = request_id.value(),
                "released prefill request-pressure expert retention ceiling before decode"
            );
        }
        let context_token_count =
            u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX);
        let residency_preparation_result =
            active_request.performance_attribution.measure_operation(
                PerformanceOperation::GenerationPreparation,
                |performance_attribution| {
                    model.refresh_phase_aware_expert_residency_plan(
                        MemoryPhase::GenerationPreparation,
                        context_token_count,
                        1,
                        performance_attribution,
                    )
                },
            );
        if let Err(residency_preparation_error) = residency_preparation_result {
            // Residency planning is an accelerator, not a correctness gate. A
            // missing plan makes every uncovered route operation-local, which is
            // slower but preserves exact model execution and the user's request.
            model.clear_phase_aware_expert_residency_plan();
            tracing::warn!(
                request_id = request_id.value(),
                error = %residency_preparation_error,
                "continued decode with operation-local expert streaming after optional residency planning failed"
            );
        }
        let planned_complete_layer_indexes_to_seat =
            model.planned_complete_layer_indexes_to_seat_before_decode();
        // The warm floor must survive seating, so seating is capped to the
        // budget the floor leaves free BEFORE any layer streams: otherwise
        // the handoff streams complete layers it immediately evicts below.
        let complete_layer_indexes_to_seat = if self.adaptive_ram_growth_guard_enabled {
            cap_complete_layer_seating_to_decode_warm_floor(
                &self.adaptive_ram_growth_guard,
                model,
                &planned_complete_layer_indexes_to_seat,
            )
        } else {
            planned_complete_layer_indexes_to_seat.clone()
        };
        if !complete_layer_indexes_to_seat.is_empty() {
            match model.seat_complete_layers_before_decode(
                &complete_layer_indexes_to_seat,
                &mut active_request.performance_attribution,
            ) {
                Ok(seated_payload_bytes) => {
                    tracing::info!(
                        request_id = request_id.value(),
                        planned_complete_layer_count = planned_complete_layer_indexes_to_seat.len(),
                        warm_floor_capped_complete_layer_count =
                            complete_layer_indexes_to_seat.len(),
                        seated_payload_bytes,
                        "seated planned complete layers before decode"
                    );
                }
                Err(seating_error) => {
                    tracing::warn!(
                        request_id = request_id.value(),
                        error = %seating_error,
                        "continued decode after planned complete-layer seating failed"
                    );
                }
            }
        }
        // Seating adopts lazy pages. Evaluate them now so the first decode
        // admission snapshot includes the seated weights. Leaving them lazy made
        // the first generate token look like a huge activation spike and leftover
        // publication evicted the same pages.
        if let Err(materialize_error) = model.materialize_seated_complete_layers() {
            tracing::warn!(
                request_id = request_id.value(),
                error = %materialize_error,
                "continued decode after seated complete-layer materialization failed"
            );
        }
        // Cache bookkeeping after seating and materialization. Decode admission
        // takes its own MLX snapshot next.
        let expert_statistics = model.expert_weight_memory_cache_statistics();
        let total_layer_count = model
            .expert_pager
            .as_ref()
            .map_or(0, |expert_pager| expert_pager.layer_count());
        tracing::info!(
            request_id = request_id.value(),
            context_token_count,
            total_layer_count,
            resident_expert_count = expert_statistics.entry_count,
            resident_expert_payload_bytes = expert_statistics.resident_payload_byte_count,
            "generation preparation seated leftover complete layers before decode"
        );
        let hot_expert_warm_slot_count = if self.adaptive_ram_growth_guard_enabled {
            // Complete-layer seating may have consumed the retained budget
            // before warm tables were sized; enforce the distributed floor
            // first so the slot count below sees a budget that can actually
            // warm every unseated layer (issue #373's mixed serving depends
            // on nonzero tables to matter at all).
            enforce_decode_warm_slot_floor(
                request_id,
                &self.adaptive_ram_growth_guard,
                model,
                usize::try_from(model.config.experts_per_token()).unwrap_or(1),
            );
            let expert_statistics = model.expert_weight_memory_cache_statistics();
            decode_warm_slot_count(&self.adaptive_ram_growth_guard, model, expert_statistics)
        } else {
            0
        };
        model.set_hot_expert_warm_slot_count(hot_expert_warm_slot_count);
        tracing::info!(
            request_id = request_id.value(),
            hot_expert_warm_slot_count,
            "decode hot-expert warm capacity decided at handoff"
        );
        Ok(())
    }
}

/// Sizes the hot-expert warm tables from the adaptive growth guard's
/// current headroom (issue #372, resized per decode step for issue #512).
///
/// The memory left after current active ownership, the phase's learned
/// transients, and one routed page, divided across the sparse layers that
/// warming may cover. Rich machines warm at the full policy capacity;
/// memory-tight machines warm narrower or not at all, because on them the
/// warm tables would compete with the paging machinery for the same bytes
/// and the eviction/re-stream cycle is net-negative.
///
/// Issue #512: this runs at the decode handoff AND on every decode step.
/// Sizing only at the handoff froze the warm tables at the first request's
/// conservative fallback reserve, because no decode-phase transient
/// evidence exists before the first decode token; decode forwards then
/// proved their transients are small, but the frozen slot count kept the
/// warm tables churning at a few experts per layer while entitlement sat
/// unclaimed. Resizing as evidence accumulates lets warming grow into the
/// entitlement the guard's decode evidence supports.
pub(in crate::qwen3_5_streaming) fn decode_warm_slot_count(
    adaptive_ram_growth_guard: &AdaptiveRamGrowthGuard,
    model: &Qwen3_5Model,
    expert_statistics: crate::ExpertWeightMemoryCacheStatistics,
) -> usize {
    model
        .runtime()
        .memory_snapshot()
        .map(|memory_snapshot| {
            let warm_budget_bytes = decode_warm_budget_bytes(
                adaptive_ram_growth_guard,
                model,
                memory_snapshot.active_memory_bytes(),
                expert_statistics.resident_payload_byte_count,
            );
            let (expert_capacity, per_expert_payload_bytes, sparse_layer_count) = model
                .expert_pager
                .as_ref()
                .and_then(|expert_pager| expert_pager.layer_plans().first())
                .and_then(|layer_plan| {
                    let expert_capacity = layer_plan.expert_capacity;
                    let per_expert_payload_bytes =
                        crate::qwen3_5_streaming::expert_paging::retained_expert_layer_page_payload_bytes(
                            layer_plan,
                            1,
                        )
                        .ok()?;
                    Some((expert_capacity, per_expert_payload_bytes))
                })
                .map_or((0, 0, 0), |(expert_capacity, per_expert_payload_bytes)| {
                    (
                        expert_capacity,
                        per_expert_payload_bytes,
                        model
                            .expert_pager
                            .as_ref()
                            .map_or(0, |expert_pager| expert_pager.layer_count()),
                    )
                });
            if expert_capacity == 0 || per_expert_payload_bytes == 0 {
                return 0;
            }
            let warm_capacity = crate::hot_expert_warm_slot_count(expert_capacity);
            // The real expert capacity decides the elastic/complete
            let complete_layer_count =
                model
                    .retained_experts
                    .as_ref()
                    .map_or(0, |retained_experts| {
                        retained_experts
                            .borrow()
                            .topology_snapshot(expert_capacity)
                            .into_iter()
                            .filter(|residency| {
                                residency.class
                                    == crate::RetainedExpertPageClass::StableCompleteLayer
                            })
                            .count()
                    });
            let unseated_sparse_layer_count =
                sparse_layer_count.saturating_sub(complete_layer_count);
            // The warm capacity scales down to what the guard's headroom
            // affords across the unseated layers; zero disables warming.
            let budget_affordable_capacity =
                if unseated_sparse_layer_count == 0 || per_expert_payload_bytes == 0 {
                    0
                } else {
                    let per_layer_warm_bytes = u128::try_from(per_expert_payload_bytes)
                        .unwrap_or(u128::MAX)
                        .saturating_mul(
                            u128::try_from(unseated_sparse_layer_count).unwrap_or(u128::MAX),
                        );
                    let affordable_slots = u128::try_from(warm_budget_bytes).unwrap_or(u128::MAX)
                        .checked_div(per_layer_warm_bytes)
                        .unwrap_or(0);
                    usize::try_from(affordable_slots).unwrap_or(usize::MAX)
                };
            warm_capacity.min(budget_affordable_capacity)
        })
        .unwrap_or(0)
}

/// The warm budget under the tighter of the guard's and the plan's ceilings.
///
/// Shared by the warm-slot floor enforcement and the slot-count sizing so
/// both decisions see byte-identical headroom for one memory snapshot.
fn decode_warm_budget_bytes(
    adaptive_ram_growth_guard: &AdaptiveRamGrowthGuard,
    model: &Qwen3_5Model,
    active_memory_bytes: usize,
    resident_payload_byte_count: u64,
) -> u64 {
    let routed_expert_page_reservation_bytes =
        model.expert_page_reservation_bytes_for_forward(1)
            .unwrap_or(u64::MAX);
    let warm_retention_ceiling_bytes = adaptive_ram_growth_guard
        .hot_expert_retention_ceiling_bytes(
            // GenerationPreparation budgets like decode (see
            // memory/budget/phase.rs), and the forwards warming reserves
            // for are decode forwards, so the decode phase's own learned
            // workspace is the right reserve basis here too.
            crate::MemoryPhase::Decode,
            active_memory_bytes,
            resident_payload_byte_count,
            usize::try_from(routed_expert_page_reservation_bytes).unwrap_or(usize::MAX),
        );
    // The plan-composed ceiling already subtracted the composer's
    // activation workspace and context reserve; the warm budget is what
    // remains under the tighter of the two ceilings.
    let plan_retained_ceiling_bytes = model.retained_expert_normal_ceiling_bytes();
    warm_retention_ceiling_bytes
        .min(plan_retained_ceiling_bytes)
        .saturating_sub(resident_payload_byte_count)
}

/// Evicts the coldest complete layers until the warm budget can seat the
/// minimum decode warm floor across every unseated sparse layer.
///
/// Issue #373's mixed serving only pays off when the warm tables hold
/// something: concentrated complete-layer seating at the handoff can
/// otherwise consume the whole retained budget and leave every unseated
/// layer re-streaming its routed pages per token. The floor is one token's
/// routing (`minimum_decode_warm_slot_count`) per sparse layer — exactly
/// the previous token's routes plus prefetch landing room. Complete layers
/// are the eviction source because a spread of partial tables covers
/// strictly more of the next token's routes per byte than full coverage of
/// a few layers; the coldest-first selector keeps the layers the request
/// actually routes to seated the longest.
fn enforce_decode_warm_slot_floor(
    request_id: RequestId,
    adaptive_ram_growth_guard: &AdaptiveRamGrowthGuard,
    model: &Qwen3_5Model,
    experts_per_token: usize,
) {
    let Some((expert_capacity, per_expert_payload_bytes, sparse_layer_count)) = model
        .expert_pager
        .as_ref()
        .and_then(|expert_pager| {
            let first_layer_plan = expert_pager.layer_plans().first()?;
            let per_expert_payload_bytes =
                crate::qwen3_5_streaming::expert_paging::retained_expert_layer_page_payload_bytes(
                    first_layer_plan,
                    1,
                )
                .ok()?;
            Some((
                first_layer_plan.expert_capacity,
                per_expert_payload_bytes,
                expert_pager.layer_count(),
            ))
        })
    else {
        return;
    };
    if expert_capacity == 0 || per_expert_payload_bytes == 0 || sparse_layer_count == 0 {
        return;
    }
    let expert_statistics = model.expert_weight_memory_cache_statistics();
    let warm_budget_bytes = model
        .runtime()
        .memory_snapshot()
        .map_or(0_u64, |memory_snapshot| {
            decode_warm_budget_bytes(
                adaptive_ram_growth_guard,
                model,
                memory_snapshot.active_memory_bytes(),
                expert_statistics.resident_payload_byte_count,
            )
        });
    let eviction_indexes = crate::decode_warm_floor_eviction_indexes(
        &model
            .retained_experts
            .as_ref()
            .map_or(Vec::new(), |retained_experts| {
                retained_experts
                    .borrow()
                    .topology_snapshot(expert_capacity)
            }),
        warm_budget_bytes,
        per_expert_payload_bytes,
        crate::minimum_decode_warm_slot_count(experts_per_token),
        sparse_layer_count,
    );
    let evicted_complete_layer_count = eviction_indexes.len();
    for coldest_complete_layer_index in eviction_indexes {
        if !model.remove_retained_expert_layer(coldest_complete_layer_index) {
            break;
        }
    }
    if evicted_complete_layer_count > 0 {
        tracing::info!(
            request_id = request_id.value(),
            evicted_complete_layer_count,
            "evicted cold complete layers to fund the decode warm-slot floor"
        );
    }
}

/// Caps the complete-layer seating list to the budget the warm floor leaves.
///
/// Runs before any seating streams a byte, so the handoff never pays the
/// solid-state-drive read and decode cost of a complete layer the floor
/// would evict moments later. Layers beyond the cap stay unseated; the
/// residency plan's later refreshes can still seat them if decode evidence
/// ever frees the budget.
fn cap_complete_layer_seating_to_decode_warm_floor(
    adaptive_ram_growth_guard: &AdaptiveRamGrowthGuard,
    model: &Qwen3_5Model,
    planned_complete_layer_indexes_to_seat: &[usize],
) -> Vec<usize> {
    let Some(expert_pager) = model.expert_pager.as_ref() else {
        return planned_complete_layer_indexes_to_seat.to_vec();
    };
    let Some(first_layer_plan) = expert_pager.layer_plans().first() else {
        return planned_complete_layer_indexes_to_seat.to_vec();
    };
    let expert_capacity = first_layer_plan.expert_capacity;
    let Some(per_expert_payload_bytes) =
        crate::qwen3_5_streaming::expert_paging::retained_expert_layer_page_payload_bytes(
            first_layer_plan,
            1,
        )
        .ok()
    else {
        return planned_complete_layer_indexes_to_seat.to_vec();
    };
    let sparse_layer_count = expert_pager.layer_count();
    let experts_per_token = usize::try_from(model.config.experts_per_token()).unwrap_or(1);
    let complete_layer_count = model
        .retained_experts
        .as_ref()
        .map_or(0, |retained_experts| {
            retained_experts
                .borrow()
                .topology_snapshot(expert_capacity)
                .into_iter()
                .filter(|layer_residency| {
                    layer_residency.class == crate::RetainedExpertPageClass::StableCompleteLayer
                })
                .count()
        });
    let unseated_sparse_layer_count = sparse_layer_count.saturating_sub(complete_layer_count);
    let minimum_warm_budget_bytes = u128::from(per_expert_payload_bytes)
        .saturating_mul(u128::try_from(unseated_sparse_layer_count).unwrap_or(u128::MAX))
        .saturating_mul(u128::try_from(crate::minimum_decode_warm_slot_count(experts_per_token))
            .unwrap_or(u128::MAX));
    let expert_statistics = model.expert_weight_memory_cache_statistics();
    let warm_budget_bytes = model
        .runtime()
        .memory_snapshot()
        .map_or(0_u64, |memory_snapshot| {
            decode_warm_budget_bytes(
                adaptive_ram_growth_guard,
                model,
                memory_snapshot.active_memory_bytes(),
                expert_statistics.resident_payload_byte_count,
            )
        });
    // Seating may consume the free warm budget beyond what the floor needs;
    // once the free budget is at or below the floor, no complete layer may
    // stream at all and the post-seating floor enforcement evicts instead.
    let seatable_budget_bytes = warm_budget_bytes.saturating_sub(u64::try_from(
        minimum_warm_budget_bytes.min(u128::from(u64::MAX)),
    )
    .unwrap_or(u64::MAX));
    if seatable_budget_bytes == 0 {
        return Vec::new();
    }
    let mut seatable_budget_bytes = seatable_budget_bytes;
    let mut capped_indexes = Vec::with_capacity(planned_complete_layer_indexes_to_seat.len());
    for &layer_index in planned_complete_layer_indexes_to_seat {
        let Some(layer_plan) = expert_pager.layer_plans().get(layer_index) else {
            continue;
        };
        let Some(complete_layer_payload_bytes) =
            crate::qwen3_5_streaming::expert_paging::retained_expert_layer_page_payload_bytes(
                layer_plan,
                layer_plan.expert_capacity,
            )
            .ok()
        else {
            continue;
        };
        if complete_layer_payload_bytes > seatable_budget_bytes {
            break;
        }
        seatable_budget_bytes -= complete_layer_payload_bytes;
        capped_indexes.push(layer_index);
    }
    capped_indexes
}
