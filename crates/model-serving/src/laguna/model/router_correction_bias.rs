//! Validates optional resident router correction bias before model execution.

use std::collections::HashMap;

use astronomical_runtime_integration::MlxArray;

use crate::laguna::artifacts::{LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId};

use super::bound_linear::is_floating_weight;
use super::error::LagunaExecutionError;

pub(super) fn bind_optional_router_correction_bias(
    tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    vectors: &mut HashMap<LagunaTensorId, MlxArray>,
    layer_index: usize,
    expert_count: u32,
) -> Result<(), LagunaExecutionError> {
    let tensor_id = LagunaTensorId::Layer {
        layer_index,
        role: LagunaLayerTensorRole::RouterCorrectionBias,
        component: LagunaTensorComponent::Weight,
    };
    let Some(correction_bias) = tensors.remove(&tensor_id) else {
        return Ok(());
    };
    let expected_expert_count = i32::try_from(expert_count).map_err(|_| {
        LagunaExecutionError::invalid_geometry(
            "router correction-bias expert count exceeds the MLX integer range",
        )
    })?;
    if correction_bias.shape() != [expected_expert_count]
        || !is_floating_weight(correction_bias.dtype())
    {
        return Err(LagunaExecutionError::invalid_geometry(
            "router correction bias must be one floating-point value per expert",
        ));
    }
    vectors.insert(tensor_id, correction_bias);
    Ok(())
}
