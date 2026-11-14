//! Which whole layers to read in before the first reply token.
//!
//! The plan may name layers to keep whole that the room left can hold. A
//! reply never reads a whole layer on its own, so those layers are read in
//! at the reply's start. This decision is the only answer to "which layers
//! must be seated before the first reply token." Family code enacts it; it
//! must not invent a second policy of skipping the load.

use super::{
    CurrentExpertLayerResidency, ExpertLayerState, ExpertResidencyPlan, RetainedExpertPageClass,
};

/// Layers the plan keeps whole that the cache does not hold whole yet.
///
/// Empty means the reply may proceed without a seating pass.
#[must_use]
pub fn kept_layer_indexes_to_seat_before_reply(
    plan: &ExpertResidencyPlan,
    current_residencies: &[CurrentExpertLayerResidency],
) -> Vec<usize> {
    let mut layer_indexes_to_seat = Vec::new();
    for layer_index in plan.complete_layer_targets.iter().copied() {
        if plan.layer_states.get(layer_index) != Some(&ExpertLayerState::Kept) {
            continue;
        }
        let is_held_whole = current_residencies.iter().any(|current_residency| {
            current_residency.layer_index == layer_index
                && current_residency.class == RetainedExpertPageClass::StableCompleteLayer
        });
        if !is_held_whole {
            layer_indexes_to_seat.push(layer_index);
        }
    }
    layer_indexes_to_seat
}
