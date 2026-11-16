//! Deterministic expert topology planning from geometry and retained ownership.

use super::validation::routed_floor_payload_bytes;
use super::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerState, ExpertResidencyPlan,
    ExpertResidencyPlanError, RetainedExpertPageClass, foundation_overlay, validation,
};

/// Plans where each layer's experts live while a reply is written: a
/// deterministic complete foundation plus a routed overlay. No plan is made
/// while a prompt is read; the `astro_expert_cache` capacity is the only rule there.
pub fn plan_expert_residency(
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
        return Ok(ExpertResidencyPlan::from_layer_states(vec![
            ExpertLayerState::Kept;
            layer_geometries.len()
        ]));
    }
    // Below full fit, what a reply finds kept is preserved before anything
    // else, and complete layers are funded only from the room that leaves.
    if !current_residencies.is_empty() {
        return preserve_existing_expert_pages_for_generation(
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
        // Not even the experts tokens route to fit for every layer, and
        // nothing is kept yet: every layer is read as tokens need it.
        return Ok(ExpertResidencyPlan::from_layer_states(vec![
            ExpertLayerState::Absent;
            layer_geometries.len()
        ]));
    }
    foundation_overlay::foundation_and_overlay_plan(
        retained_expert_ceiling_bytes,
        layer_geometries,
        &routed_floor_bytes,
        all_layer_routed_floor_bytes,
    )
}

fn preserve_existing_expert_pages_for_generation(
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
    let layer_states = current_by_layer
        .iter()
        .enumerate()
        .map(|(layer_index, residency)| {
            match (residency, selected[layer_index], promoted[layer_index]) {
                // The room left funds the whole layer.
                (_, _, true) => ExpertLayerState::Kept,
                (Some(residency), true, false)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerState::Kept
                }
                // What is held of the layer stays; a layer holding nothing
                // keeps the experts the reply routes to.
                (Some(_), true, false) | (None, _, false) => ExpertLayerState::Hot,
                // What is held does not fit the capacity.
                (Some(_), false, false) => ExpertLayerState::Absent,
            }
        })
        .collect();
    Ok(ExpertResidencyPlan::from_layer_states(layer_states))
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
