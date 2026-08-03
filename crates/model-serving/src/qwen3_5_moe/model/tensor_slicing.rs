use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::Qwen3_5MoEExecutionError;

pub(super) fn slice_last_dimension(
    runtime: &MlxRuntime,
    input: &MlxArray,
    final_dimension_start_index: i32,
    final_dimension_stop_index: i32,
) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
    let input_shape = input.shape();
    let input_rank = input_shape.len();
    let mut slice_start_indices = vec![0; input_rank];
    let mut slice_stop_indices = input_shape;
    let slice_strides = vec![1; input_rank];
    let final_axis = input_rank
        .checked_sub(1)
        .ok_or(Qwen3_5MoEExecutionError::InvalidInput {
            description: "cannot slice the final dimension of a scalar",
        })?;
    slice_start_indices[final_axis] = final_dimension_start_index;
    slice_stop_indices[final_axis] = final_dimension_stop_index;
    Ok(runtime.slice(
        input,
        &slice_start_indices,
        &slice_stop_indices,
        &slice_strides,
    )?)
}
