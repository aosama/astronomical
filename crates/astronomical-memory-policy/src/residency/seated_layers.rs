//! What is resident right now, as opposed to what a plan intends.

use super::{CurrentExpertLayerResidency, RetainedExpertPageClass};

/// Counts the layers that are actually resident as complete layers right now.
/// This is occupancy, not what a plan intends: publishing an intended count
/// under an occupancy name once made a log claim 34 layers while nine were
/// resident.
#[must_use]
pub fn seated_complete_layer_count(current_residencies: &[CurrentExpertLayerResidency]) -> usize {
    current_residencies
        .iter()
        .filter(|current_residency| {
            current_residency.class == RetainedExpertPageClass::StableCompleteLayer
        })
        .count()
}

/// Floors the retained-page ceiling at every resident expert page, warm
/// tables included.
///
/// A learned budget that tightens between requests (context-reserve growth)
/// must not proactively discard routing knowledge any more than it discards
/// seated complete layers: the hot-expert tables cost disk re-reads to rebuild
/// and are reclaimed exactly when a real forward needs the bytes.
#[must_use]
pub const fn retained_resident_ceiling_after_budget_refresh(
    leftover_expert_budget_bytes: u64,
    current_resident_payload_bytes: u64,
) -> u64 {
    if leftover_expert_budget_bytes > current_resident_payload_bytes {
        leftover_expert_budget_bytes
    } else {
        current_resident_payload_bytes
    }
}

#[cfg(test)]
#[path = "seated_layers_tests.rs"]
mod tests;
