//! The foundation-and-overlay strategy for a reply that begins with nothing
//! kept.
//!
//! When the capacity cannot hold every complete layer, every layer keeps
//! the experts tokens route to, and the cheapest layers are kept whole on
//! top of that until the capacity is used up.

use super::{ExpertLayerGeometry, ExpertLayerState, ExpertResidencyPlan, ExpertResidencyPlanError};

pub(super) fn foundation_and_overlay_plan(
    capacity_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    routed_floor_bytes: &[u64],
    all_layer_routed_floor_bytes: u64,
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let mut cost_to_keep_whole_by_layer = Vec::with_capacity(geometries.len());
    for geometry in geometries {
        cost_to_keep_whole_by_layer.push((
            geometry.complete_layer_payload_bytes - routed_floor_bytes[geometry.layer_index],
            geometry.layer_index,
        ));
    }
    cost_to_keep_whole_by_layer.sort_unstable();
    let mut layer_states = vec![ExpertLayerState::Hot; geometries.len()];
    let mut planned_bytes = all_layer_routed_floor_bytes;
    for (cost_to_keep_whole_bytes, layer_index) in cost_to_keep_whole_by_layer {
        if cost_to_keep_whole_bytes <= capacity_bytes.saturating_sub(planned_bytes) {
            layer_states[layer_index] = ExpertLayerState::Kept;
            planned_bytes = planned_bytes
                .checked_add(cost_to_keep_whole_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
    }
    if planned_bytes > capacity_bytes {
        return Err(ExpertResidencyPlanError::PlannedResidencyExceedsCeiling);
    }
    Ok(ExpertResidencyPlan::from_layer_states(layer_states))
}
