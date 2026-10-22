//! Fail-closed validation and checked byte arithmetic for residency planning.

use super::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertResidencyPlanError,
    RetainedExpertPageClass,
};

pub(super) fn validate_inputs<'a>(
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_residencies: &'a [CurrentExpertLayerResidency],
) -> Result<Vec<Option<&'a CurrentExpertLayerResidency>>, ExpertResidencyPlanError> {
    if geometries.is_empty() {
        return Err(ExpertResidencyPlanError::EmptyGeometry);
    }
    for (expected_layer_index, geometry) in geometries.iter().enumerate() {
        validate_geometry(expected_layer_index, geometry)?;
    }
    let mut current_by_layer = vec![None; geometries.len()];
    let mut previous_layer_index = None;
    let mut current_payload_bytes = 0_u64;
    for residency in current_residencies {
        if residency.layer_index >= geometries.len() {
            return Err(ExpertResidencyPlanError::CurrentLayerOutOfRange {
                layer_index: residency.layer_index,
            });
        }
        if previous_layer_index.is_some_and(|previous| previous >= residency.layer_index) {
            return Err(ExpertResidencyPlanError::DuplicateOrUnorderedCurrentLayer {
                layer_index: residency.layer_index,
            });
        }
        previous_layer_index = Some(residency.layer_index);
        validate_current_residency(&geometries[residency.layer_index], residency)?;
        current_payload_bytes = current_payload_bytes
            .checked_add(residency.payload_bytes)
            .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        current_by_layer[residency.layer_index] = Some(residency);
    }
    if current_payload_bytes > ceiling_bytes {
        return Err(ExpertResidencyPlanError::CurrentResidencyExceedsCeiling);
    }
    Ok(current_by_layer)
}

fn validate_geometry(
    expected_layer_index: usize,
    geometry: &ExpertLayerGeometry,
) -> Result<(), ExpertResidencyPlanError> {
    if geometry.layer_index != expected_layer_index {
        return Err(ExpertResidencyPlanError::NonContiguousLayerIndex {
            expected_layer_index,
            actual_layer_index: geometry.layer_index,
        });
    }
    if geometry.expert_capacity == 0
        || geometry.expert_payload_bytes == 0
        || geometry.complete_layer_payload_bytes == 0
        || geometry.experts_per_token == 0
    {
        return Err(ExpertResidencyPlanError::ZeroGeometry {
            layer_index: geometry.layer_index,
        });
    }
    let expected_complete_bytes = geometry
        .expert_payload_bytes
        .checked_mul(
            u64::try_from(geometry.expert_capacity)
                .map_err(|_| ExpertResidencyPlanError::ByteCountOverflow)?,
        )
        .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
    if expected_complete_bytes != geometry.complete_layer_payload_bytes {
        return Err(ExpertResidencyPlanError::InconsistentCompletePayload {
            layer_index: geometry.layer_index,
        });
    }
    Ok(())
}

fn validate_current_residency(
    geometry: &ExpertLayerGeometry,
    residency: &CurrentExpertLayerResidency,
) -> Result<(), ExpertResidencyPlanError> {
    let ids_are_valid = !residency.retained_expert_ids.is_empty()
        && residency
            .retained_expert_ids
            .windows(2)
            .all(|ids| ids[0] < ids[1])
        && residency
            .retained_expert_ids
            .iter()
            .all(|expert_id| *expert_id < geometry.expert_capacity);
    if !ids_are_valid {
        return Err(ExpertResidencyPlanError::InvalidRetainedExpertIds {
            layer_index: residency.layer_index,
        });
    }
    let expected_payload_bytes = geometry
        .expert_payload_bytes
        .checked_mul(
            u64::try_from(residency.retained_expert_ids.len())
                .map_err(|_| ExpertResidencyPlanError::ByteCountOverflow)?,
        )
        .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
    let class_is_consistent = match residency.class {
        RetainedExpertPageClass::StableCompleteLayer => {
            residency.retained_expert_ids.len() == geometry.expert_capacity
        }
        RetainedExpertPageClass::ElasticRoutedExperts => {
            residency.retained_expert_ids.len() < geometry.expert_capacity
        }
    };
    if expected_payload_bytes != residency.payload_bytes || !class_is_consistent {
        return Err(ExpertResidencyPlanError::InconsistentCurrentPayload {
            layer_index: residency.layer_index,
            payload_bytes: residency.payload_bytes,
            geometry_expert_payload_bytes: geometry.expert_payload_bytes,
            retained_count: residency.retained_expert_ids.len(),
            expected_payload_bytes,
        });
    }
    Ok(())
}

pub(super) fn routed_floor_payload_bytes(
    geometry: &ExpertLayerGeometry,
) -> Result<u64, ExpertResidencyPlanError> {
    geometry
        .expert_payload_bytes
        .checked_mul(
            u64::try_from(geometry.experts_per_token.min(geometry.expert_capacity))
                .map_err(|_| ExpertResidencyPlanError::ByteCountOverflow)?,
        )
        .ok_or(ExpertResidencyPlanError::ByteCountOverflow)
}

pub(super) fn checked_sum(
    mut byte_counts: impl Iterator<Item = u64>,
) -> Result<u64, ExpertResidencyPlanError> {
    byte_counts.try_fold(0_u64, |total_bytes, byte_count| {
        total_bytes
            .checked_add(byte_count)
            .ok_or(ExpertResidencyPlanError::ByteCountOverflow)
    })
}
