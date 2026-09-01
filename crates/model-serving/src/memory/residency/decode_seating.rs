//! Decode-handoff seating after an atomic complete-owner demote.
//!
//! The planner may name complete layers that leftover RAM can hold. Decode never
//! takes the complete-layer stream path, so those indexes are not loaded by
//! later generate tokens. This decision is the only answer to "which complete
//! layers must be seated before the first decode token." Family code enacts it;
//! it must not invent a second policy of skipping the load.

use super::{ExpertLayerResidencyTarget, ExpertResidencyPlan};

/// Complete-layer indexes the plan named for promotion that are not already retained.
///
/// Empty means decode may proceed without a seating pass.
#[must_use]
pub fn complete_layer_indexes_required_before_decode(plan: &ExpertResidencyPlan) -> Vec<usize> {
    plan.complete_layer_targets
        .iter()
        .copied()
        .filter(|&layer_index| {
            matches!(
                plan.layer_targets.get(layer_index),
                Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead)
            )
        })
        .collect()
}
