//! Sort stacked token-to-expert assignments so gathered matrix products read
//! contiguous expert rows. Inverse order maps original token/top-K slots back
//! onto the sorted axis.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use super::error::SparseExpertError;
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

const OPERATION: &str = "sort stacked expert assignments";

/// Sorted gather inputs plus the inverse permutation for reduction.
pub struct SortedExpertAssignments {
    pub sorted_states: MlxArray,
    pub sorted_indices: MlxArray,
    pub inverse_order: MlxArray,
}

/// Sorts assignments by expert id and gathers the matching token rows.
///
/// Empty assignment sets (`N = 0`) return empty `[0, 1, D]` states and empty
/// index vectors so a later gathered product and reduction can complete without
/// a family-specific branch.
pub fn sort_expert_assignments(
    runtime: &MlxRuntime,
    expanded_states: &MlxArray,
    selected_indices: &MlxArray,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<SortedExpertAssignments, MlxRuntimeError> {
    performance_attribution
        .measure_operation(PerformanceOperation::ExpertAssignmentPreparation, |_| {
            sort_expert_assignments_inner(runtime, expanded_states, selected_indices)
        })
}

fn sort_expert_assignments_inner(
    runtime: &MlxRuntime,
    expanded_states: &MlxArray,
    selected_indices: &MlxArray,
) -> Result<SortedExpertAssignments, MlxRuntimeError> {
    let selected_index_shape = selected_indices.shape();
    let expanded_state_shape = expanded_states.shape();
    let hidden_dimension = *expanded_state_shape
        .last()
        .ok_or_else(|| geometry_error("expanded hidden states must include a hidden dimension"))?;
    let assignment_count = i32::try_from(selected_indices.element_count())
        .map_err(|_| geometry_error("expert assignment count exceeds the MLX integer range"))?;
    if assignment_count == 0 {
        return empty_sorted_assignments(runtime, hidden_dimension, expanded_states);
    }
    let expert_count_per_token = selected_index_shape.last().copied().ok_or_else(|| {
        geometry_error("selected expert indices must include an expert dimension")
    })?;
    if selected_index_shape.len() < 2
        || expanded_state_shape.len() < 3
        || expert_count_per_token <= 0
        || expanded_state_shape[expanded_state_shape.len() - 2] != 1
    {
        return Err(geometry_error(
            "expert assignments and expanded hidden states have invalid shapes",
        ));
    }
    let token_count = assignment_count / expert_count_per_token;
    let expanded_token_count = expanded_state_shape[..expanded_state_shape.len() - 2]
        .iter()
        .try_fold(1_i32, |token_product, dimension| {
            token_product.checked_mul(*dimension)
        })
        .ok_or_else(|| geometry_error("expanded hidden-state shape overflows"))?;
    if assignment_count % expert_count_per_token != 0
        || expanded_token_count != token_count
        || hidden_dimension <= 0
    {
        return Err(geometry_error(
            "expert assignments do not match expanded hidden-state tokens",
        ));
    }

    let flattened_indices = runtime.reshape(selected_indices, &[assignment_count])?;
    let sorted_order = runtime.argsort_axis(&flattened_indices, 0)?;
    let inverse_order = runtime.argsort_axis(&sorted_order, 0)?;
    let assignments_per_token = runtime.array_from_u32(
        &[u32::try_from(expert_count_per_token).map_err(|_| {
            geometry_error("expert count per token exceeds the unsigned integer range")
        })?],
        &[],
    )?;
    let sorted_token_indices = runtime.floor_divide(&sorted_order, &assignments_per_token)?;
    let flattened_states = runtime.reshape(expanded_states, &[token_count, 1, hidden_dimension])?;
    let sorted_states = runtime.take_axis(&flattened_states, &sorted_token_indices, 0)?;
    let sorted_indices = runtime.take_axis(&flattened_indices, &sorted_order, 0)?;
    Ok(SortedExpertAssignments {
        sorted_states,
        sorted_indices,
        inverse_order,
    })
}

/// Restores sorted gather outputs to original token and top-K order.
///
/// Production sorted reduction does not need this. Characterization and
/// operations-based references use it to prove the inverse permutation.
pub fn restore_expert_assignment_order(
    runtime: &MlxRuntime,
    sorted_expert_outputs: &MlxArray,
    inverse_order: &MlxArray,
    selected_index_shape: &[i32],
) -> Result<MlxArray, MlxRuntimeError> {
    let assignment_count = selected_index_shape
        .iter()
        .try_fold(1_i32, |assignment_product, dimension| {
            assignment_product.checked_mul(*dimension)
        })
        .ok_or_else(|| geometry_error("selected expert-index shape overflows"))?;
    if assignment_count == 0 {
        let output_dimension = sorted_expert_outputs.shape().last().copied().unwrap_or(0);
        let mut restored_shape = selected_index_shape.to_vec();
        restored_shape.push(output_dimension);
        return runtime.zeros(&restored_shape, sorted_expert_outputs.dtype());
    }
    let sorted_output_shape = sorted_expert_outputs.shape();
    if sorted_output_shape.len() != 3
        || sorted_output_shape[0] != assignment_count
        || sorted_output_shape[1] != 1
        || inverse_order.shape() != [assignment_count]
    {
        return Err(geometry_error(
            "sorted expert outputs and inverse order have incompatible shapes",
        ));
    }
    let output_dimension = sorted_output_shape[2];
    let original_order_outputs = runtime.take_axis(sorted_expert_outputs, inverse_order, 0)?;
    let mut restored_shape = selected_index_shape.to_vec();
    restored_shape.push(1);
    restored_shape.push(output_dimension);
    let restored_outputs = runtime.reshape(&original_order_outputs, &restored_shape)?;
    runtime.squeeze_axis(&restored_outputs, -2)
}

fn empty_sorted_assignments(
    runtime: &MlxRuntime,
    hidden_dimension: i32,
    expanded_states: &MlxArray,
) -> Result<SortedExpertAssignments, MlxRuntimeError> {
    if hidden_dimension <= 0 {
        return Err(geometry_error(
            "expanded hidden states must have a positive hidden dimension",
        ));
    }
    Ok(SortedExpertAssignments {
        sorted_states: runtime.zeros(&[0, 1, hidden_dimension], expanded_states.dtype())?,
        sorted_indices: runtime.array_from_u32(&[], &[0])?,
        inverse_order: runtime.array_from_u32(&[], &[0])?,
    })
}

fn geometry_error(description: &'static str) -> MlxRuntimeError {
    SparseExpertError::InvalidAssignmentGeometry { description }.into_runtime_error(OPERATION)
}
