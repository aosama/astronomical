use astronomical_memory_policy::residency::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, RetainedExpertPageClass,
    minimum_decode_warm_slot_count,
};

pub(crate) fn complete_layers_within_decode_warm_budget(
    geometries: &[ExpertLayerGeometry],
    current_residencies: &[CurrentExpertLayerResidency],
    planned_layer_indexes: &[usize],
    available_new_retention_bytes: u64,
) -> Vec<usize> {
    let retained_bytes = |layer_index| {
        current_residencies
            .iter()
            .find(|residency| residency.layer_index == layer_index)
            .map_or(0, |residency| residency.payload_bytes)
    };
    let is_complete = |layer_index| {
        current_residencies.iter().any(|residency| {
            residency.layer_index == layer_index
                && residency.class == RetainedExpertPageClass::StableCompleteLayer
        })
    };
    let completion_bytes = |layer_index: usize| {
        geometries[layer_index]
            .complete_layer_payload_bytes
            .saturating_sub(retained_bytes(layer_index))
    };
    let completes_every_layer = geometries.iter().all(|geometry| {
        is_complete(geometry.layer_index) || planned_layer_indexes.contains(&geometry.layer_index)
    });
    let complete_remaining_bytes = planned_layer_indexes
        .iter()
        .try_fold(0_u64, |total_bytes, &layer_index| {
            total_bytes.checked_add(completion_bytes(layer_index))
        });
    // Complete layers need no routed warm floor. Reserving that floor for
    // layers this very seating pass completes strands a full-fit entitlement.
    if completes_every_layer
        && complete_remaining_bytes.is_some_and(|bytes| bytes <= available_new_retention_bytes)
    {
        return planned_layer_indexes.to_vec();
    }
    let mut remaining_budget_bytes = available_new_retention_bytes;
    let mut selected_layer_indexes = Vec::new();
    for &layer_index in planned_layer_indexes {
        let remaining_floor_bytes = geometries
            .iter()
            .filter(|geometry| {
                geometry.layer_index != layer_index
                    && !is_complete(geometry.layer_index)
                    && !selected_layer_indexes.contains(&geometry.layer_index)
            })
            .fold(0_u64, |floor_bytes, geometry| {
                floor_bytes.saturating_add(
                    geometry
                        .expert_payload_bytes
                        .saturating_mul(
                            minimum_decode_warm_slot_count(geometry.experts_per_token) as u64
                        )
                        .min(geometry.complete_layer_payload_bytes)
                        .saturating_sub(retained_bytes(geometry.layer_index)),
                )
            });
        let required_new_bytes = completion_bytes(layer_index);
        if required_new_bytes.saturating_add(remaining_floor_bytes) > remaining_budget_bytes {
            continue;
        }
        remaining_budget_bytes -= required_new_bytes;
        selected_layer_indexes.push(layer_index);
    }
    selected_layer_indexes
}

pub(crate) trait DecodeExpertSeating {
    type Error: std::fmt::Display;

    fn seat_complete_layer(&mut self, layer_index: usize) -> Result<Option<u64>, Self::Error>;
    fn materialize_complete_layers(&mut self) -> Result<(), Self::Error>;
}

pub(crate) fn seat_and_materialize_complete_layers<Seating: DecodeExpertSeating>(
    seating: &mut Seating,
    layer_indexes: &[usize],
) -> Result<u64, Seating::Error> {
    let mut seated_payload_bytes = 0_u64;
    let seating_outcome = (|| {
        for &layer_index in layer_indexes {
            let Some(payload_bytes) = seating.seat_complete_layer(layer_index)? else {
                break;
            };
            seated_payload_bytes = seated_payload_bytes.saturating_add(payload_bytes);
        }
        Ok(seated_payload_bytes)
    })();
    // Successful earlier seats still need evaluation when a later read fails.
    let materialization_outcome = seating.materialize_complete_layers();
    match (seating_outcome, materialization_outcome) {
        (Err(seating_error), Err(materialization_error)) => {
            tracing::warn!(
                error = %materialization_error,
                "materialization also failed after an expert seating failure"
            );
            Err(seating_error)
        }
        (Err(seating_error), Ok(())) => Err(seating_error),
        (Ok(_), Err(materialization_error)) => Err(materialization_error),
        (Ok(payload_bytes), Ok(())) => Ok(payload_bytes),
    }
}
