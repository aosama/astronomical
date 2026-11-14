//! The foundation-and-overlay expert residency strategy.
//!
//! When the ceiling cannot hold every complete layer, this planner keeps the
//! routed floor for all layers and promotes the cheapest complete layers on
//! top. This strategy stays separate from `planner.rs`'s dispatch so each file
//! stays inside the source-size budget.

use super::planner::{compare_partial_coverage, release_order};
use super::validation;
use super::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerState, ExpertResidencyPlan,
    ExpertResidencyPlanError, RetainedExpertPageClass,
};
use crate::phase::MemoryPhase;

/// The per-layer planning facts every foundation-and-overlay stage reads.
struct FoundationOverlayLayerFacts<'a> {
    geometries: &'a [ExpertLayerGeometry],
    current_by_layer: &'a [Option<&'a CurrentExpertLayerResidency>],
    routed_floor_bytes: &'a [u64],
}

/// The complete-layer foundation chosen inside the retained-bytes ceiling.
#[derive(Debug)]
struct CompleteLayerSelection {
    complete_targets: Vec<bool>,
    foundation_and_floor_bytes: u64,
}

/// The preserved and newly retained byte charges of the planned layer states.
#[derive(Debug)]
struct LayerResidencyAccounting {
    preserved_bytes: u64,
    new_retained_bytes: u64,
    layer_states: Vec<ExpertLayerState>,
}

pub(super) fn foundation_and_overlay_plan(
    phase: MemoryPhase,
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
    routed_floor_bytes: &[u64],
    all_layer_routed_floor_bytes: u64,
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let layer_facts = FoundationOverlayLayerFacts {
        geometries,
        current_by_layer,
        routed_floor_bytes,
    };
    let complete_selection = select_complete_layers_within_ceiling(
        &layer_facts,
        ceiling_bytes,
        all_layer_routed_floor_bytes,
    )?;
    let preserve_partial = select_preserved_partial_layers(
        &layer_facts,
        ceiling_bytes,
        complete_selection.foundation_and_floor_bytes,
        &complete_selection.complete_targets,
    );
    let residency_accounting = account_planned_layer_charges(
        &layer_facts,
        &complete_selection.complete_targets,
        &preserve_partial,
    )?;
    let reserved_routed_overlay_bytes =
        validation::checked_sum(routed_floor_bytes.iter().enumerate().filter_map(
            |(layer_index, floor_bytes)| {
                (!complete_selection.complete_targets[layer_index]).then_some(*floor_bytes)
            },
        ))?;
    let target_capacity_bytes = validation::checked_sum(geometries.iter().map(|geometry| {
        if complete_selection.complete_targets[geometry.layer_index] {
            geometry.complete_layer_payload_bytes
        } else {
            routed_floor_bytes[geometry.layer_index]
        }
    }))?;
    if target_capacity_bytes > ceiling_bytes {
        return Err(ExpertResidencyPlanError::PlannedResidencyExceedsCeiling);
    }
    Ok(ExpertResidencyPlan {
        phase,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets: complete_selection
            .complete_targets
            .iter()
            .enumerate()
            .filter_map(|(layer_index, is_complete)| is_complete.then_some(layer_index))
            .collect(),
        layer_states: residency_accounting.layer_states,
        reserved_routed_overlay_bytes,
        expected_preserved_bytes: residency_accounting.preserved_bytes,
        maximum_new_retained_bytes: residency_accounting.new_retained_bytes,
        deterministic_release_order: release_order(current_by_layer),
        is_low_budget_partial_mode: false,
    })
}

/// Promotes the cheapest complete layers on top of the routed-floor foundation
/// until the retained-bytes ceiling is exhausted.
fn select_complete_layers_within_ceiling(
    layer_facts: &FoundationOverlayLayerFacts<'_>,
    ceiling_bytes: u64,
    all_layer_routed_floor_bytes: u64,
) -> Result<CompleteLayerSelection, ExpertResidencyPlanError> {
    let mut complete_targets = vec![false; layer_facts.geometries.len()];
    let mut foundation_and_floor_bytes = all_layer_routed_floor_bytes;
    let mut complete_candidates = layer_facts
        .geometries
        .iter()
        .map(|geometry| {
            let layer_index = geometry.layer_index;
            (
                layer_index,
                geometry.complete_layer_payload_bytes - layer_facts.routed_floor_bytes[layer_index],
                layer_facts.current_by_layer[layer_index].is_some_and(|residency| {
                    residency.class == RetainedExpertPageClass::StableCompleteLayer
                }),
            )
        })
        .collect::<Vec<_>>();
    complete_candidates.sort_unstable_by_key(|(layer_index, incremental_bytes, is_complete)| {
        (!*is_complete, *incremental_bytes, *layer_index)
    });
    for (layer_index, incremental_bytes, _) in complete_candidates {
        if incremental_bytes <= ceiling_bytes.saturating_sub(foundation_and_floor_bytes) {
            complete_targets[layer_index] = true;
            foundation_and_floor_bytes = foundation_and_floor_bytes
                .checked_add(incremental_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
    }
    Ok(CompleteLayerSelection {
        complete_targets,
        foundation_and_floor_bytes,
    })
}

/// Preserves seated elastic pages that fit inside the overlay budget left
/// above the routed-floor foundation, skipping layers with complete targets.
fn select_preserved_partial_layers(
    layer_facts: &FoundationOverlayLayerFacts<'_>,
    ceiling_bytes: u64,
    foundation_and_floor_bytes: u64,
    complete_targets: &[bool],
) -> Vec<bool> {
    let mut overlay_extra_bytes = ceiling_bytes.saturating_sub(foundation_and_floor_bytes);
    let mut preserve_partial = vec![false; layer_facts.geometries.len()];
    let mut partial_candidates = layer_facts
        .current_by_layer
        .iter()
        .enumerate()
        .filter_map(|(layer_index, residency)| {
            let residency = residency.as_ref().copied()?;
            (!complete_targets[layer_index]
                && residency.class == RetainedExpertPageClass::ElasticRoutedExperts)
                .then_some((layer_index, residency))
        })
        .collect::<Vec<_>>();
    partial_candidates.sort_unstable_by(|left, right| compare_partial_coverage(*right, *left));
    for (layer_index, residency) in partial_candidates {
        let incremental_bytes = residency
            .payload_bytes
            .saturating_sub(layer_facts.routed_floor_bytes[layer_index]);
        if incremental_bytes <= overlay_extra_bytes {
            preserve_partial[layer_index] = true;
            overlay_extra_bytes = overlay_extra_bytes.saturating_sub(incremental_bytes);
        }
    }
    preserve_partial
}

/// Charges every planned layer state against the preserved and new-retention byte
/// budgets.
fn account_planned_layer_charges(
    layer_facts: &FoundationOverlayLayerFacts<'_>,
    complete_targets: &[bool],
    preserve_partial: &[bool],
) -> Result<LayerResidencyAccounting, ExpertResidencyPlanError> {
    let mut preserved_bytes = 0_u64;
    let mut new_retained_bytes = 0_u64;
    let mut layer_states = Vec::with_capacity(layer_facts.geometries.len());
    for layer_index in 0..layer_facts.geometries.len() {
        let current_residency = layer_facts.current_by_layer[layer_index];
        let is_held_whole = current_residency.is_some_and(|residency| {
            residency.class == RetainedExpertPageClass::StableCompleteLayer
        });
        let layer_state = if complete_targets[layer_index] {
            ExpertLayerState::Kept
        } else if current_residency.is_none() || (!is_held_whole && preserve_partial[layer_index]) {
            // A layer holding nothing keeps the experts tokens route to; a
            // partly held one stays as it is while the overlay has room.
            ExpertLayerState::Hot
        } else {
            ExpertLayerState::Absent
        };
        // The layer's charged capacity: its complete payload when seated,
        // otherwise the routed floor the plan reserves for it.
        let charged_capacity_bytes = if complete_targets[layer_index] {
            layer_facts.geometries[layer_index].complete_layer_payload_bytes
        } else {
            layer_facts.routed_floor_bytes[layer_index]
        };
        match (layer_state, current_residency) {
            // What is held stays. Bytes held above the charged capacity (a
            // partly held layer larger than its routed floor) are overlay
            // that is resident already, not budget taken from other layers'
            // floors, so only the overlap with the charged capacity reduces
            // the new-retention budget.
            (ExpertLayerState::Kept, Some(residency))
            | (ExpertLayerState::Hot, Some(residency)) => {
                let kept_charged_bytes = residency.payload_bytes.min(charged_capacity_bytes);
                preserved_bytes = preserved_bytes
                    .checked_add(residency.payload_bytes)
                    .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
                new_retained_bytes = new_retained_bytes
                    .checked_add(charged_capacity_bytes - kept_charged_bytes)
                    .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
            }
            // A layer to keep whole that holds nothing is read in when the
            // reply begins; it adds nothing here.
            (ExpertLayerState::Kept, None) => {}
            // A whole layer that no longer has a complete claim: its routed
            // floor is resident inside the seated table already and must not
            // be counted as a fresh storage read.
            (ExpertLayerState::Absent, Some(residency)) if is_held_whole => {
                let retained_floor_bytes = residency
                    .payload_bytes
                    .min(layer_facts.routed_floor_bytes[layer_index]);
                preserved_bytes = preserved_bytes
                    .checked_add(retained_floor_bytes)
                    .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
                new_retained_bytes = new_retained_bytes
                    .checked_add(charged_capacity_bytes - retained_floor_bytes)
                    .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
            }
            // Every other layer's reserved floor becomes resident through
            // fresh reads: a partly held layer that is dropped, and a layer
            // that holds nothing and keeps what tokens route to.
            (ExpertLayerState::Absent, _) | (ExpertLayerState::Hot, None) => {
                new_retained_bytes = new_retained_bytes
                    .checked_add(charged_capacity_bytes)
                    .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
            }
        }
        layer_states.push(layer_state);
    }
    Ok(LayerResidencyAccounting {
        preserved_bytes,
        new_retained_bytes,
        layer_states,
    })
}

#[cfg(test)]
#[path = "foundation_overlay_tests.rs"]
mod foundation_overlay_tests;
