//! Route-demand recording into the phase-split evidence streams.
//!
//! All three production record paths — the routed dispatch in `forward.rs`,
//! the read-through residency path, and the deferred routes of seated
//! complete layers — funnel through this single dispatch, so the phase field
//! alone decides which stream receives a sighting. No call site needs to
//! know the split exists.

use super::WarmExpertSlotTableCache;
use super::demand_phase::ExpertDemandPhase;

impl WarmExpertSlotTableCache {
    /// Records one forward's routed selection into the current phase's
    /// stream.
    ///
    /// Prefill evidence carries the chunk weight (the final-chunk token
    /// density boost); decode evidence always records at plain weight so
    /// lifetime counts stay comparable across tokens, requests, and the
    /// sessions they persist into.
    pub fn record_expert_demand(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_expert_ids: &[usize],
    ) {
        match self.demand_phase {
            ExpertDemandPhase::Prefill => record_selection_into_layer(
                &mut self.prefill_demand_counts_by_layer,
                layer_index,
                expert_capacity,
                selected_expert_ids,
                self.demand_assignment_weight,
            ),
            ExpertDemandPhase::Decode => record_selection_into_layer(
                &mut self.decode_demand_counts_by_layer,
                layer_index,
                expert_capacity,
                selected_expert_ids,
                1,
            ),
        }
    }
}

/// Pure: adds one weighted sighting of every selected expert into its
/// layer's counter vector, sizing the vector to the layer's expert capacity
/// on first contact. A phase weight of zero floors at one so a configured
/// zero can never silently discard real routing evidence.
fn record_selection_into_layer(
    demand_counts_by_layer: &mut [Vec<u64>],
    layer_index: usize,
    expert_capacity: usize,
    selected_expert_ids: &[usize],
    assignment_weight: u64,
) {
    let Some(layer_demand_counts) = demand_counts_by_layer.get_mut(layer_index) else {
        return;
    };
    if layer_demand_counts.len() < expert_capacity {
        layer_demand_counts.resize(expert_capacity, 0);
    }
    let effective_assignment_weight = assignment_weight.max(1);
    for expert_id in selected_expert_ids {
        if let Some(demand_count) = layer_demand_counts.get_mut(*expert_id) {
            *demand_count = demand_count.saturating_add(effective_assignment_weight);
        }
    }
}
