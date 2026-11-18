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

/// Whether every one of a model's `layer_count` expert layers is resident as
/// a complete layer right now. A step that routes several rows a layer (a
/// draft round) then finds every expert in memory, and needs none of the
/// working room a read from storage takes.
#[must_use]
pub fn holds_every_layer_whole(
    current_residencies: &[CurrentExpertLayerResidency],
    layer_count: usize,
) -> bool {
    layer_count > 0 && seated_complete_layer_count(current_residencies) == layer_count
}

#[cfg(test)]
#[path = "seated_layers_tests.rs"]
mod tests;
