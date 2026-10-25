//! Deterministic expert topology planning from geometry and retained ownership.

use super::validation::routed_floor_payload_bytes;
use super::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPlan, ExpertResidencyPlanError, RetainedExpertPageClass, foundation_overlay,
    validation,
};
use crate::phase::MemoryPhase;

/// Produces a deterministic complete-foundation plus routed-overlay plan.
pub fn plan_expert_residency(
    phase: MemoryPhase,
    retained_expert_ceiling_bytes: u64,
    layer_geometries: &[ExpertLayerGeometry],
    current_residencies: &[CurrentExpertLayerResidency],
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let current_by_layer = validation::validate_inputs(
        retained_expert_ceiling_bytes,
        layer_geometries,
        current_residencies,
    )?;
    let complete_model_payload_bytes = validation::checked_sum(
        layer_geometries
            .iter()
            .map(|geometry| geometry.complete_layer_payload_bytes),
    )?;
    if complete_model_payload_bytes <= retained_expert_ceiling_bytes {
        return complete_model_plan(
            phase,
            retained_expert_ceiling_bytes,
            layer_geometries,
            &current_by_layer,
            complete_model_payload_bytes,
        );
    }
    // A full-fit completion preserves the existing pages and funds only their
    // missing experts. Below full fit, preserve paid-for generation pages
    // instead of displacing them, then fund complete layers only from the
    // leftover that preservation did not claim.
    if matches!(
        phase,
        MemoryPhase::GenerationPreparation | MemoryPhase::Decode
    ) && !current_residencies.is_empty()
    {
        return preserve_existing_expert_pages_for_generation(
            phase,
            retained_expert_ceiling_bytes,
            layer_geometries,
            &current_by_layer,
        );
    }

    let routed_floor_bytes = layer_geometries
        .iter()
        .map(routed_floor_payload_bytes)
        .collect::<Result<Vec<_>, _>>()?;
    let all_layer_routed_floor_bytes = validation::checked_sum(routed_floor_bytes.iter().copied())?;
    if all_layer_routed_floor_bytes > retained_expert_ceiling_bytes {
        return low_budget_partial_plan(
            phase,
            retained_expert_ceiling_bytes,
            layer_geometries,
            &current_by_layer,
        );
    }
    foundation_overlay::foundation_and_overlay_plan(
        phase,
        retained_expert_ceiling_bytes,
        layer_geometries,
        &current_by_layer,
        &routed_floor_bytes,
        all_layer_routed_floor_bytes,
    )
}

fn complete_model_plan(
    phase: MemoryPhase,
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
    complete_model_payload_bytes: u64,
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let mut preserved_bytes = 0_u64;
    let mut layer_targets = Vec::with_capacity(geometries.len());
    for current_residency in current_by_layer {
        if current_residency.is_some_and(|residency| {
            residency.class == RetainedExpertPageClass::StableCompleteLayer
        }) {
            layer_targets.push(ExpertLayerResidencyTarget::PreserveComplete);
        } else {
            layer_targets.push(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead);
        }
        if let Some(residency) = current_residency {
            preserved_bytes = preserved_bytes
                .checked_add(residency.payload_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
    }
    Ok(ExpertResidencyPlan {
        phase,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets: (0..geometries.len()).collect(),
        layer_targets,
        reserved_routed_overlay_bytes: 0,
        expected_preserved_bytes: preserved_bytes,
        maximum_new_retained_bytes: complete_model_payload_bytes.saturating_sub(preserved_bytes),
        deterministic_release_order: release_order(current_by_layer),
        is_low_budget_partial_mode: false,
    })
}

fn preserve_existing_expert_pages_for_generation(
    phase: MemoryPhase,
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let mut candidate_layers = current_by_layer
        .iter()
        .enumerate()
        .filter_map(|(layer_index, residency)| residency.map(|residency| (layer_index, residency)))
        .collect::<Vec<_>>();
    candidate_layers.sort_unstable_by(|left, right| compare_preservation_priority(*left, *right));
    let mut selected = vec![false; geometries.len()];
    let mut preserved_bytes = 0_u64;
    for (layer_index, residency) in candidate_layers {
        if residency.payload_bytes <= ceiling_bytes.saturating_sub(preserved_bytes) {
            selected[layer_index] = true;
            preserved_bytes = preserved_bytes
                .checked_add(residency.payload_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
    }
    let promoted = generation_completion_promotions(
        ceiling_bytes.saturating_sub(preserved_bytes),
        geometries,
        current_by_layer,
        &selected,
    )?;
    let layer_targets = current_by_layer
        .iter()
        .enumerate()
        .map(|(layer_index, residency)| {
            match (residency, selected[layer_index], promoted[layer_index]) {
                (_, _, true) => ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead,
                (Some(residency), true, false)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::PreserveComplete
                }
                (Some(_), true, false) => ExpertLayerResidencyTarget::PreservePartial,
                (Some(residency), false, false)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
                }
                (Some(_), false, false) => ExpertLayerResidencyTarget::ReleasePartial,
                (None, _, false) => ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
            }
        })
        .collect();
    Ok(ExpertResidencyPlan {
        phase,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets: (0..geometries.len())
            .filter(|&layer_index| {
                promoted[layer_index]
                    || (selected[layer_index]
                        && current_by_layer[layer_index].is_some_and(|residency| {
                            residency.class == RetainedExpertPageClass::StableCompleteLayer
                        }))
            })
            .collect(),
        layer_targets,
        reserved_routed_overlay_bytes: 0,
        expected_preserved_bytes: preserved_bytes,
        maximum_new_retained_bytes: ceiling_bytes.saturating_sub(preserved_bytes),
        deterministic_release_order: release_order(current_by_layer),
        is_low_budget_partial_mode: false,
    })
}

/// Layers whose complete page the generation leftover can fund after preservation.
///
/// Decode multiplies a complete layer in place on the device, but every
/// incomplete layer forces a per-token route copy to the host plus SSD reads
/// for its misses. Leaving the leftover entitlement idle below full fit is
/// what kept a 21 GB ceiling at 50 decode tokens per second with 4 GB unused
/// (the 2026-10-25 flight: 2.9 GB of funded new retention, zero layers
/// seated). Completions are funded cheapest first because each completed
/// layer removes the same per-token synchronization regardless of its size,
/// so the count of complete layers is what decode speed follows. Every layer
/// that stays unretained keeps its routed floor reserved, so a promotion never
/// starves the per-token routed pages an incomplete layer still needs.
fn generation_completion_promotions(
    leftover_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
    selected: &[bool],
) -> Result<Vec<bool>, ExpertResidencyPlanError> {
    let mut completion_candidates = Vec::new();
    let mut unretained_routed_floor_bytes = 0_u64;
    for (layer_index, geometry) in geometries.iter().enumerate() {
        let (completion_cost_bytes, routed_floor_bytes) = match current_by_layer[layer_index] {
            None => {
                let routed_floor_bytes = routed_floor_payload_bytes(geometry)?;
                unretained_routed_floor_bytes = unretained_routed_floor_bytes
                    .checked_add(routed_floor_bytes)
                    .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
                (geometry.complete_layer_payload_bytes, routed_floor_bytes)
            }
            Some(residency)
                if selected[layer_index]
                    && residency.class == RetainedExpertPageClass::ElasticRoutedExperts =>
            {
                (
                    geometry
                        .complete_layer_payload_bytes
                        .saturating_sub(residency.payload_bytes),
                    0,
                )
            }
            Some(_) => continue,
        };
        completion_candidates.push((completion_cost_bytes, layer_index, routed_floor_bytes));
    }
    completion_candidates.sort_unstable();
    let mut promoted = vec![false; geometries.len()];
    let mut remaining_leftover_bytes = leftover_bytes;
    for (completion_cost_bytes, layer_index, routed_floor_bytes) in completion_candidates {
        let reserve_after_promotion_bytes =
            unretained_routed_floor_bytes.saturating_sub(routed_floor_bytes);
        if completion_cost_bytes.saturating_add(reserve_after_promotion_bytes)
            <= remaining_leftover_bytes
        {
            promoted[layer_index] = true;
            remaining_leftover_bytes -= completion_cost_bytes;
            unretained_routed_floor_bytes = reserve_after_promotion_bytes;
        }
    }
    Ok(promoted)
}

fn low_budget_partial_plan(
    phase: MemoryPhase,
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let mut candidate_layers = current_by_layer
        .iter()
        .enumerate()
        .filter_map(|(layer_index, residency)| residency.map(|residency| (layer_index, residency)))
        .collect::<Vec<_>>();
    candidate_layers.sort_unstable_by(|left, right| compare_preservation_priority(*left, *right));
    let mut selected = vec![false; geometries.len()];
    let mut preserved_bytes = 0_u64;
    for (layer_index, residency) in candidate_layers {
        if residency.payload_bytes <= ceiling_bytes.saturating_sub(preserved_bytes) {
            selected[layer_index] = true;
            preserved_bytes = preserved_bytes
                .checked_add(residency.payload_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
    }
    let layer_targets = current_by_layer
        .iter()
        .enumerate()
        .map(
            |(layer_index, residency)| match (residency, selected[layer_index]) {
                (Some(residency), true)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::PreserveComplete
                }
                (Some(_), true) => ExpertLayerResidencyTarget::PreservePartial,
                (Some(residency), false)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
                }
                (Some(_), false) => ExpertLayerResidencyTarget::ReleasePartial,
                (None, _) => ExpertLayerResidencyTarget::StreamOperationLocal,
            },
        )
        .collect();
    Ok(ExpertResidencyPlan {
        phase,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets: selected
            .iter()
            .enumerate()
            .filter_map(|(layer_index, is_selected)| {
                (*is_selected
                    && current_by_layer[layer_index].is_some_and(|residency| {
                        residency.class == RetainedExpertPageClass::StableCompleteLayer
                    }))
                .then_some(layer_index)
            })
            .collect(),
        layer_targets,
        reserved_routed_overlay_bytes: 0,
        expected_preserved_bytes: preserved_bytes,
        maximum_new_retained_bytes: 0,
        deterministic_release_order: release_order(current_by_layer),
        is_low_budget_partial_mode: true,
    })
}

pub(super) fn compare_partial_coverage(
    left: (usize, &CurrentExpertLayerResidency),
    right: (usize, &CurrentExpertLayerResidency),
) -> std::cmp::Ordering {
    let left_score = u128::from(left.1.covered_weighted_demand) * u128::from(right.1.payload_bytes);
    let right_score =
        u128::from(right.1.covered_weighted_demand) * u128::from(left.1.payload_bytes);
    left_score
        .cmp(&right_score)
        .then_with(|| right.0.cmp(&left.0))
}

fn compare_preservation_priority(
    left: (usize, &CurrentExpertLayerResidency),
    right: (usize, &CurrentExpertLayerResidency),
) -> std::cmp::Ordering {
    let left_is_complete = left.1.class == RetainedExpertPageClass::StableCompleteLayer;
    let right_is_complete = right.1.class == RetainedExpertPageClass::StableCompleteLayer;
    left_is_complete
        .cmp(&right_is_complete)
        .then_with(|| compare_partial_coverage(right, left))
}

pub(super) fn release_order(
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
) -> Vec<usize> {
    let mut partial_layers = current_by_layer
        .iter()
        .enumerate()
        .filter_map(|(layer_index, residency)| {
            let residency = residency.as_ref().copied()?;
            (residency.class == RetainedExpertPageClass::ElasticRoutedExperts)
                .then_some((layer_index, residency))
        })
        .collect::<Vec<_>>();
    partial_layers.sort_unstable_by(|left, right| compare_partial_coverage(*left, *right));
    let mut release_order = partial_layers
        .into_iter()
        .map(|(layer_index, _)| layer_index)
        .collect::<Vec<_>>();
    release_order.extend(current_by_layer.iter().enumerate().rev().filter_map(
        |(layer_index, residency)| {
            residency
                .is_some_and(|residency| {
                    residency.class == RetainedExpertPageClass::StableCompleteLayer
                })
                .then_some(layer_index)
        },
    ));
    release_order
}
