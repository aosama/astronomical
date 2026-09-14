//! The foundation-and-overlay expert residency strategy.
//!
//! When the ceiling cannot hold every complete layer, this planner keeps the
//! routed floor for all layers and promotes the cheapest complete layers on
//! top; split from the strategy dispatch in the parent so each file stays
//! inside the source-size budget. As a child module it reaches the parent's
//! private helpers and plan vocabulary directly.

use super::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPlan, ExpertResidencyPlanError, RetainedExpertPageClass, checked_sum,
    compare_partial_coverage, release_order,
};
use crate::memory::MemoryPhase;

pub(super) fn foundation_and_overlay_plan(
    phase: MemoryPhase,
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
    routed_floor_bytes: &[u64],
    all_layer_routed_floor_bytes: u64,
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let mut complete_targets = vec![false; geometries.len()];
    let mut foundation_and_floor_bytes = all_layer_routed_floor_bytes;
    let mut complete_candidates = geometries
        .iter()
        .map(|geometry| {
            let layer_index = geometry.layer_index;
            (
                layer_index,
                geometry.complete_layer_payload_bytes - routed_floor_bytes[layer_index],
                current_by_layer[layer_index].is_some_and(|residency| {
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

    let mut overlay_extra_bytes = ceiling_bytes.saturating_sub(foundation_and_floor_bytes);
    let mut preserve_partial = vec![false; geometries.len()];
    let mut partial_candidates = current_by_layer
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
            .saturating_sub(routed_floor_bytes[layer_index]);
        if incremental_bytes <= overlay_extra_bytes {
            preserve_partial[layer_index] = true;
            overlay_extra_bytes = overlay_extra_bytes.saturating_sub(incremental_bytes);
        }
    }

    let mut preserved_bytes = 0_u64;
    let mut layer_targets = Vec::with_capacity(geometries.len());
    for layer_index in 0..geometries.len() {
        let current_residency = current_by_layer[layer_index];
        let target = if complete_targets[layer_index] {
            if current_residency.is_some_and(|residency| {
                residency.class == RetainedExpertPageClass::StableCompleteLayer
            }) {
                ExpertLayerResidencyTarget::PreserveComplete
            } else {
                ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
            }
        } else {
            match current_residency {
                Some(residency)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
                }
                Some(_) if preserve_partial[layer_index] => {
                    ExpertLayerResidencyTarget::PreservePartial
                }
                Some(_) => ExpertLayerResidencyTarget::ReleasePartial,
                None => ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
            }
        };
        if matches!(
            target,
            ExpertLayerResidencyTarget::PreserveComplete
                | ExpertLayerResidencyTarget::PreservePartial
                | ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
        ) && let Some(residency) = current_residency
        {
            preserved_bytes = preserved_bytes
                .checked_add(residency.payload_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
        layer_targets.push(target);
    }
    let reserved_routed_overlay_bytes =
        checked_sum(routed_floor_bytes.iter().enumerate().filter_map(
            |(layer_index, floor_bytes)| (!complete_targets[layer_index]).then_some(*floor_bytes),
        ))?;
    let target_capacity_bytes = checked_sum(geometries.iter().map(|geometry| {
        if complete_targets[geometry.layer_index] {
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
        complete_layer_targets: complete_targets
            .iter()
            .enumerate()
            .filter_map(|(layer_index, is_complete)| is_complete.then_some(layer_index))
            .collect(),
        layer_targets,
        reserved_routed_overlay_bytes,
        expected_preserved_bytes: preserved_bytes,
        maximum_new_retained_bytes: target_capacity_bytes.saturating_sub(preserved_bytes),
        deterministic_release_order: release_order(current_by_layer),
        is_low_budget_partial_mode: false,
    })
}
