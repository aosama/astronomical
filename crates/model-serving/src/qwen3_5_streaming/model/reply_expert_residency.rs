//! Prepares the model's expert retention for a reply, after the prompt is read.
//!
//! Entering the reply phase is one event for the memory governor: a reply
//! step needs far fewer MLX working buffers than a prompt step, so the
//! astro expert cache's capacity rises (issue #2221). This file then fills that
//! capacity: whole layers the plan chose, no more than the reply's
//! whole-layer keep count allows (`decode_whole_layer_keep`, issue #2338),
//! and hot-expert slots on the layers left, sized from the room the whole
//! layers leave. Filling is best-effort because missing routes remain correct through
//! operation-local streaming.
//!
//! It lives with the model, not the engine, because every step reads or
//! changes the model's memory and nothing about a request: a measurement can
//! put a directly loaded model in the state the engine serves a reply in by
//! calling the same function.

use super::Qwen3_5Model;
use crate::PerformanceAttribution;

/// What preparing a reply's expert retention did, for the caller to log.
#[derive(Debug)]
pub(crate) struct ReplyExpertResidencyPreparation {
    /// Why no plan was made; the reply then streams what it lacks.
    pub(crate) planning_error: Option<String>,
    pub(crate) planned_complete_layer_count: usize,
    /// Planned layers left after the warm floor's cap: the ones seating tried.
    pub(crate) seated_complete_layer_count: usize,
    /// Bytes seated, or why seating stopped.
    pub(crate) seating: Result<u64, String>,
    pub(crate) total_layer_count: usize,
    pub(crate) resident_expert_count: usize,
    pub(crate) resident_expert_payload_bytes: u64,
    /// Whole layers given up beyond the reply's whole-layer keep count.
    pub(crate) released_beyond_keep_count: usize,
    pub(crate) evicted_complete_layer_count: usize,
    pub(crate) hot_expert_warm_slot_count: usize,
}

impl Qwen3_5Model {
    /// Tells the memory governor the reply begins, then fills the capacity
    /// it answers with: plans the layers, seats the whole ones, and sizes the
    /// hot-expert tables. Nothing here can fail the request.
    pub(crate) fn prepare_expert_residency_for_reply(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> ReplyExpertResidencyPreparation {
        self.govern_reply_start();
        let planning_error = self.plan_reply_layers_or_stream(performance_attribution);
        let planned_complete_layer_indexes_to_seat = self.expert_layers_to_seat_before_reply();
        // The warm floor must survive seating, so seating is capped to the
        // budget the floor leaves free BEFORE any layer streams: otherwise
        // the handoff streams complete layers it immediately evicts below.
        let complete_layer_indexes_to_seat =
            super::decode_whole_layer_keep::cap_seating_to_whole_layer_keep_count(
                self,
                &cap_complete_layer_seating_to_decode_warm_floor(
                    self,
                    &planned_complete_layer_indexes_to_seat,
                ),
            );
        let seating = self
            .seat_complete_layers_before_decode(
                &complete_layer_indexes_to_seat,
                performance_attribution,
            )
            .map_err(|seating_error| seating_error.to_string());
        let expert_statistics = self.astro_expert_cache_statistics();
        // Complete-layer seating may have used the capacity before the hot
        // tables were sized; enforce the distributed floor first so the slot
        // count below sees room that can warm every unseated layer (issue
        // #373's mixed serving depends on nonzero tables to matter at all).
        let released_beyond_keep_count =
            super::decode_whole_layer_keep::release_complete_layers_beyond_keep_count(self);
        let evicted_complete_layer_count = enforce_decode_warm_slot_floor(
            self,
            usize::try_from(self.config.experts_per_token()).unwrap_or(1),
        );
        let hot_expert_warm_slot_count = decode_warm_slot_count(self);
        self.set_hot_expert_warm_slot_count(hot_expert_warm_slot_count);
        ReplyExpertResidencyPreparation {
            planning_error,
            planned_complete_layer_count: planned_complete_layer_indexes_to_seat.len(),
            seated_complete_layer_count: complete_layer_indexes_to_seat.len(),
            seating,
            total_layer_count: self
                .expert_pager
                .as_ref()
                .map_or(0, |expert_pager| expert_pager.layer_count()),
            resident_expert_count: expert_statistics.entry_count,
            resident_expert_payload_bytes: expert_statistics.resident_payload_byte_count,
            released_beyond_keep_count,
            evicted_complete_layer_count,
            hot_expert_warm_slot_count,
        }
    }

    /// Plans where each layer ends up for the reply. The plan is an
    /// accelerator, not a correctness gate: without one no extra layer is
    /// read in at the reply's start, which is slower but preserves exact
    /// model execution, so a failure is returned for the log and the reply
    /// goes on.
    fn plan_reply_layers_or_stream(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Option<String> {
        // Only the planning is timed as generation preparation, as before
        // this lived with the model; seating records its own operations.
        let planning_result = performance_attribution.measure_operation(
            crate::PerformanceOperation::GenerationPreparation,
            |planning_attribution| self.plan_expert_layer_states_for_reply(planning_attribution),
        );
        match planning_result {
            Ok(()) => None,
            Err(planning_error) => {
                self.forget_expert_layer_states();
                Some(planning_error.to_string())
            }
        }
    }

    /// Puts a model loaded directly into the state the engine serves a reply
    /// in, for measurements: the reply's layers planned and seated and the
    /// hot-expert tables sized, exactly as the engine does after a prompt.
    #[doc(hidden)]
    pub fn prepare_expert_residency_for_reply_for_tests(&self) {
        let _ = self.prepare_expert_residency_for_reply(&mut PerformanceAttribution::disabled());
    }
}

/// Sizes the hot-expert tables from the room the astro expert cache has left (issue
/// #372).
///
/// The astro expert cache's capacity for the reply phase, less what it already keeps,
/// divided across the sparse layers the tables may cover. With room to
/// spare the tables run at the full policy size; with little they run
/// narrower or not at all, because tables that do not fit would only be
/// evicted and read again.
///
/// This runs at the reply's start and after every reply step, so the tables
/// follow the capacity as the governor's reply-step measurement settles
/// (issue #512).
pub(in crate::qwen3_5_streaming) fn decode_warm_slot_count(model: &Qwen3_5Model) -> usize {
    let warm_budget_bytes = decode_warm_budget_bytes(model);
    let Some(layer_geometry) = decode_warm_layer_geometry(model) else {
        return 0;
    };
    if layer_geometry.expert_capacity == 0 || layer_geometry.per_expert_payload_bytes == 0 {
        return 0;
    }
    let warm_capacity = crate::hot_expert_warm_slot_count(layer_geometry.expert_capacity);
    let complete_layer_count = stable_complete_layer_count(model, layer_geometry.expert_capacity);
    let unseated_sparse_layer_count = layer_geometry
        .sparse_layer_count
        .saturating_sub(complete_layer_count);
    // The warm capacity scales down to what the room left in the astro expert cache
    // affords across the unseated layers; zero disables warming.
    let budget_affordable_capacity = budget_affordable_slot_count(
        warm_budget_bytes,
        layer_geometry.per_expert_payload_bytes,
        unseated_sparse_layer_count,
    );
    warm_capacity.min(budget_affordable_capacity)
}

/// The first sparse layer plan's paging geometry, shared by the decode
/// warm-sizing decisions so they all price bytes from the same layer.
struct DecodeWarmLayerGeometry {
    expert_capacity: usize,
    per_expert_payload_bytes: u64,
    sparse_layer_count: usize,
}

/// Reads the first sparse layer plan's expert capacity, one expert's payload
/// bytes averaged over every sparse layer, and the pager's sparse layer
/// count.
///
/// `None` means no warm decision applies — no pager, no layer plans, or no
/// layer that can price its experts — and every caller
/// treats that as "size to zero" rather than inventing a fallback geometry.
fn decode_warm_layer_geometry(model: &Qwen3_5Model) -> Option<DecodeWarmLayerGeometry> {
    let expert_pager = model.expert_pager.as_ref()?;
    let first_layer_plan = expert_pager.layer_plans().first()?;
    // Priced at the mean over every layer: a mixed-precision model's first
    // layer can hold experts twice the size of the rest, and pricing by it
    // left gigabytes of the capacity unused (issue #2338).
    let per_expert_payload_bytes =
        super::decode_whole_layer_keep::mean_expert_payload_bytes(expert_pager);
    if per_expert_payload_bytes == 0 {
        return None;
    }
    Some(DecodeWarmLayerGeometry {
        expert_capacity: first_layer_plan.expert_capacity,
        per_expert_payload_bytes,
        sparse_layer_count: expert_pager.layer_count(),
    })
}

/// Counts the sparse layers seated as stable complete layers: the layers
/// whose experts are already fully resident, so warm tables would duplicate
/// them. Layers without retained-expert seating report zero.
fn stable_complete_layer_count(model: &Qwen3_5Model, expert_capacity: usize) -> usize {
    let Some(astro_expert_cache) = model.astro_expert_cache.as_ref() else {
        return 0;
    };
    let topology_snapshot = astro_expert_cache
        .borrow()
        .topology_snapshot(expert_capacity);
    let mut complete_layer_count = 0;
    for residency in topology_snapshot {
        if residency.class == crate::RetainedExpertPageClass::StableCompleteLayer {
            complete_layer_count += 1;
        }
    }
    complete_layer_count
}

/// The warm-table slot count the warm budget affords across every unseated
/// sparse layer: the budget divided by one layer's warm payload.
///
/// Zero unseated layers (or a zero per-expert payload) disables warming,
/// because there is nothing left to warm; the division never rounds up into
/// bytes the guard has not released.
fn budget_affordable_slot_count(
    warm_budget_bytes: u64,
    per_expert_payload_bytes: u64,
    unseated_sparse_layer_count: usize,
) -> usize {
    if unseated_sparse_layer_count == 0 || per_expert_payload_bytes == 0 {
        return 0;
    }
    let per_layer_warm_bytes = u128::from(per_expert_payload_bytes)
        .saturating_mul(u128::try_from(unseated_sparse_layer_count).unwrap_or(u128::MAX));
    let affordable_slots = u128::from(warm_budget_bytes)
        .checked_div(per_layer_warm_bytes)
        .unwrap_or(0);
    usize::try_from(affordable_slots).unwrap_or(usize::MAX)
}

/// The room the warm tables share: the capacity the memory governor set for
/// the reply phase, less the layers held whole.
///
/// The warm tables' own experts are not subtracted. This is re-read after
/// every reply step to size the tables, and subtracting what the tables
/// already held made each re-size shrink them toward the floor as they
/// filled: replies at half the model's size kept every table at 16 experts
/// and left about 0.6 GB of the capacity unused (issue #2338). Every sizing
/// decision in this file reads the room through this one function, so they
/// agree.
fn decode_warm_budget_bytes(model: &Qwen3_5Model) -> u64 {
    model
        .kept_expert_capacity_bytes()
        .saturating_sub(held_whole_layer_payload_bytes(model))
}

/// Payload bytes of the layers the `astro_expert_cache` holds whole.
fn held_whole_layer_payload_bytes(model: &Qwen3_5Model) -> u64 {
    let mut held_whole_bytes = 0_u64;
    for layer_residency in model.current_expert_layer_residencies() {
        if layer_residency.class == crate::RetainedExpertPageClass::StableCompleteLayer {
            held_whole_bytes = held_whole_bytes.saturating_add(layer_residency.payload_bytes);
        }
    }
    held_whole_bytes
}

/// Evicts the coldest complete layers until the warm budget can seat the
/// minimum decode warm floor across every unseated sparse layer, and returns
/// how many it evicted.
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
fn enforce_decode_warm_slot_floor(model: &Qwen3_5Model, experts_per_token: usize) -> usize {
    let Some(layer_geometry) = decode_warm_layer_geometry(model) else {
        return 0;
    };
    if layer_geometry.expert_capacity == 0
        || layer_geometry.per_expert_payload_bytes == 0
        || layer_geometry.sparse_layer_count == 0
    {
        return 0;
    }
    let warm_budget_bytes = decode_warm_budget_bytes(model);
    let retained_topology_snapshot = match model.astro_expert_cache.as_ref() {
        Some(astro_expert_cache) => astro_expert_cache
            .borrow()
            .topology_snapshot(layer_geometry.expert_capacity),
        None => Vec::new(),
    };
    let eviction_indexes = crate::decode_warm_floor_eviction_indexes(
        &retained_topology_snapshot,
        warm_budget_bytes,
        layer_geometry.per_expert_payload_bytes,
        crate::minimum_decode_warm_slot_count(experts_per_token),
        layer_geometry.sparse_layer_count,
    );
    let mut evicted_complete_layer_count = 0;
    for coldest_complete_layer_index in eviction_indexes {
        if !model.remove_retained_expert_layer(coldest_complete_layer_index) {
            break;
        }
        evicted_complete_layer_count += 1;
    }
    evicted_complete_layer_count
}

/// Caps the complete-layer seating list to the budget the warm floor leaves.
///
/// Runs before any seating streams a byte, so the handoff never pays the
/// solid-state-drive read and decode cost of a complete layer the floor
/// would evict moments later. Layers beyond the cap stay unseated; the
/// residency plan's later refreshes can still seat them if decode evidence
/// ever frees the budget.
fn cap_complete_layer_seating_to_decode_warm_floor(
    model: &Qwen3_5Model,
    planned_complete_layer_indexes_to_seat: &[usize],
) -> Vec<usize> {
    let Some(expert_pager) = model.expert_pager.as_ref() else {
        return planned_complete_layer_indexes_to_seat.to_vec();
    };
    let experts_per_token = usize::try_from(model.config.experts_per_token()).unwrap_or(1);
    let geometries = match crate::qwen3_5_streaming::model::phase_aware_expert_residency::retained_expert_layer_geometries(
        expert_pager.layer_plans(), experts_per_token
    ) {
        Ok(geometries) => geometries,
        Err(geometry_error) => {
            tracing::warn!(error = %geometry_error, "skipped decode seating after invalid expert geometry");
            return Vec::new();
        }
    };
    let current_residencies =
        model
            .astro_expert_cache
            .as_ref()
            .map_or_else(Vec::new, |astro_expert_cache| {
                astro_expert_cache.borrow().topology_snapshot(
                    geometries
                        .first()
                        .map_or(0, |geometry| geometry.expert_capacity),
                )
            });
    let warm_budget_bytes = decode_warm_budget_bytes(model);
    crate::memory::decode_seating::complete_layers_within_decode_warm_budget(
        &geometries,
        &current_residencies,
        planned_complete_layer_indexes_to_seat,
        warm_budget_bytes,
    )
}

#[cfg(test)]
mod budget_affordable_slot_count_tests {
    use super::budget_affordable_slot_count;

    #[test]
    fn should_size_slots_to_the_budget_across_every_unseated_layer() {
        // 1000 warm bytes across 10 unseated layers, one expert priced at
        // 10 bytes: 100 bytes per layer affords 10 warm slots.
        assert_eq!(budget_affordable_slot_count(1_000, 10, 10), 10);
    }

    #[test]
    fn should_seat_exactly_one_layer_when_the_budget_divides_evenly() {
        assert_eq!(budget_affordable_slot_count(100, 10, 10), 1);
    }

    #[test]
    fn should_floor_to_zero_when_the_budget_cannot_seat_one_layer() {
        assert_eq!(budget_affordable_slot_count(99, 10, 10), 0);
    }

    #[test]
    fn should_disable_warming_when_no_layers_remain_unseated() {
        assert_eq!(budget_affordable_slot_count(1_000, 10, 0), 0);
    }

    #[test]
    fn should_disable_warming_when_the_payload_price_is_zero() {
        assert_eq!(budget_affordable_slot_count(1_000, 0, 10), 0);
    }

    #[test]
    fn should_report_zero_for_a_budget_dwarfed_by_extreme_geometry() {
        // u64::MAX warm bytes cannot seat even one expert per layer once
        // every layer prices at u64::MAX bytes: the per-layer product
        // approaches the u128 ceiling and the division floors to zero.
        assert_eq!(
            budget_affordable_slot_count(u64::MAX, u64::MAX, usize::MAX),
            0
        );
    }
}
