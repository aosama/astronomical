//! Previous-token prefetch admission (issue #537).
//!
//! After a decode token streams its routed experts, leftover expert RAM may
//! keep that exact set for the next token. This module only answers which
//! experts may occupy leftover capacity. It never names an eviction victim:
//! if leftover cannot hold an expert, that expert is dropped.

/// One expert the previous token routed, with the facts admission needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreviousTokenPrefetchCandidate {
    pub layer_index: usize,
    pub expert_id: usize,
    pub payload_bytes: u64,
    pub is_already_resident: bool,
}

/// How many leftover slots one sparse layer may fill without eviction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreviousTokenPrefetchLayerCapacity {
    pub layer_index: usize,
    pub free_slot_count: usize,
}

/// Admission outcome for one previous-token prefetch attempt.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PreviousTokenPrefetchPlan {
    pub experts_to_retain: Vec<(usize, usize)>,
    pub skipped_already_resident_count: u64,
    pub dropped_for_capacity_count: u64,
}

/// Chooses previous-token experts that fit leftover slots without eviction.
///
/// Candidates are considered in caller order. Already-resident experts are
/// skipped. A layer with no free slot drops every remaining miss for that
/// layer rather than displacing a demanded page.
#[must_use]
pub fn plan_previous_token_prefetch(
    candidates: &[PreviousTokenPrefetchCandidate],
    layer_capacity: &[PreviousTokenPrefetchLayerCapacity],
) -> PreviousTokenPrefetchPlan {
    let mut remaining_free_slots_by_layer: Vec<(usize, usize)> = layer_capacity
        .iter()
        .map(|layer| (layer.layer_index, layer.free_slot_count))
        .collect();
    let mut plan = PreviousTokenPrefetchPlan::default();
    for candidate in candidates {
        if candidate.is_already_resident {
            plan.skipped_already_resident_count =
                plan.skipped_already_resident_count.saturating_add(1);
            continue;
        }
        if candidate.payload_bytes == 0 {
            plan.dropped_for_capacity_count = plan.dropped_for_capacity_count.saturating_add(1);
            continue;
        }
        let Some(remaining_free_slots) = remaining_free_slots_by_layer
            .iter_mut()
            .find(|(layer_index, _)| *layer_index == candidate.layer_index)
            .map(|(_, free_slot_count)| free_slot_count)
        else {
            plan.dropped_for_capacity_count = plan.dropped_for_capacity_count.saturating_add(1);
            continue;
        };
        if *remaining_free_slots == 0 {
            plan.dropped_for_capacity_count = plan.dropped_for_capacity_count.saturating_add(1);
            continue;
        }
        *remaining_free_slots = remaining_free_slots.saturating_sub(1);
        plan.experts_to_retain
            .push((candidate.layer_index, candidate.expert_id));
    }
    plan
}
