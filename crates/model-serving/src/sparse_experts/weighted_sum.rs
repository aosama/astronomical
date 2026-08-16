//! Weighted reduction of routed expert outputs.
//!
//! Sorted reduction consumes ordered gather outputs plus the inverse permutation
//! and never materializes `[tokens, top_k, hidden]`. Unsorted reduction multiplies
//! the original assignment tensor by the original scores.

use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    MlxRuntime, MlxRuntimeError,
};

use super::error::SparseExpertError;
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

const OPERATION: &str = "reduce stacked expert outputs";
const SORTED_EXPERT_WEIGHTED_SUM_SOURCE: &str = r#"
    auto output_index = thread_position_in_grid.x;
    auto token_index = output_index / output_dimension;
    auto output_dimension_index = output_index % output_dimension;
    float weighted_sum = 0.0f;
    for (int selected_expert_index = 0;
         selected_expert_index < selected_expert_count;
         ++selected_expert_index) {
        auto assignment_index =
            token_index * selected_expert_count + selected_expert_index;
        auto sorted_assignment_index = inverse_order[assignment_index];
        weighted_sum +=
            (float)sorted_outputs[sorted_assignment_index * output_dimension +
                                  output_dimension_index] *
            (float)scores[assignment_index];
    }
    weighted_outputs[output_index] = (OutputT)weighted_sum;
"#;

/// Builds the retained Metal kernel for sorted expert reduction.
pub fn sorted_expert_weighted_sum_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    MlxMetalKernel::new(
        "sorted_expert_weighted_sum",
        &["sorted_outputs", "inverse_order", "scores"],
        &["weighted_outputs"],
        SORTED_EXPERT_WEIGHTED_SUM_SOURCE,
    )
}

/// Reduces unsorted `[..., K, D]` outputs by original assignment scores.
pub fn unsorted_expert_weighted_sum(
    runtime: &MlxRuntime,
    selected_expert_outputs: &MlxArray,
    selected_scores: &MlxArray,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, MlxRuntimeError> {
    performance_attribution.measure_operation(PerformanceOperation::ExpertWeightedReduction, |_| {
        unsorted_expert_weighted_sum_inner(runtime, selected_expert_outputs, selected_scores)
    })
}

/// Applies router scores to expert inputs while preserving the activation dtype.
///
/// Some model contracts place routing weights before the expert projections.
/// Float32 score arithmetic must not widen the gathered matrix input because
/// that can select a different, slower matrix kernel for every expert projection.
pub fn router_weighted_expert_inputs(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    selected_scores: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let expanded_hidden_states = runtime.expand_dims(hidden_states, -2)?;
    let expanded_scores = runtime.expand_dims(selected_scores, -1)?;
    let weighted_hidden_states = runtime.multiply(&expanded_hidden_states, &expanded_scores)?;
    let activation_dtype_inputs = runtime.astype(&weighted_hidden_states, hidden_states.dtype())?;
    runtime.expand_dims(&activation_dtype_inputs, -2)
}

/// Reduces sorted gather outputs with original scores through the inverse map.
pub fn sorted_expert_weighted_sum(
    runtime: &MlxRuntime,
    weighted_sum_kernel: &MlxMetalKernel,
    sorted_expert_outputs: &MlxArray,
    inverse_order: &MlxArray,
    selected_scores: &MlxArray,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, MlxRuntimeError> {
    performance_attribution.measure_operation(PerformanceOperation::ExpertWeightedReduction, |_| {
        sorted_expert_weighted_sum_inner(
            runtime,
            weighted_sum_kernel,
            sorted_expert_outputs,
            inverse_order,
            selected_scores,
        )
    })
}

fn unsorted_expert_weighted_sum_inner(
    runtime: &MlxRuntime,
    selected_expert_outputs: &MlxArray,
    selected_scores: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let output_shape = selected_expert_outputs.shape();
    let score_shape = selected_scores.shape();
    if output_shape.len() < 2 || score_shape.len() < 1 {
        return Err(geometry_error(
            "unsorted expert outputs and scores must include an assignment axis",
        ));
    }
    let selected_expert_count = *score_shape
        .last()
        .ok_or_else(|| geometry_error("selected scores must not be scalar"))?;
    if selected_expert_count == 0 {
        return empty_weighted_outputs(runtime, &score_shape, selected_expert_outputs);
    }
    let expanded_scores = runtime.expand_dims(selected_scores, -1)?;
    let weighted_outputs = runtime.multiply(selected_expert_outputs, &expanded_scores)?;
    let float32_accumulated_output = runtime.sum_axis(&weighted_outputs, -2, false)?;
    // Router probabilities remain Float32 through multiplication and reduction
    // for numerical stability. Restore the expert activation dtype only after
    // accumulation so one-token MoE decode cannot widen every following layer
    // to Float32 and disable MLX's low-precision matrix kernels.
    runtime.astype(&float32_accumulated_output, selected_expert_outputs.dtype())
}

fn sorted_expert_weighted_sum_inner(
    runtime: &MlxRuntime,
    weighted_sum_kernel: &MlxMetalKernel,
    sorted_expert_outputs: &MlxArray,
    inverse_order: &MlxArray,
    selected_scores: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let sorted_output_shape = sorted_expert_outputs.shape();
    let score_shape = selected_scores.shape();
    if sorted_output_shape.len() != 3
        || sorted_output_shape[1] != 1
        || score_shape.len() < 2
        || inverse_order.dtype() != MlxDtype::UInt32
    {
        return Err(geometry_error(
            "sorted expert outputs, inverse order, and scores have invalid shapes or dtypes",
        ));
    }
    let selected_expert_count = *score_shape
        .last()
        .ok_or_else(|| geometry_error("selected scores must not be scalar"))?;
    let assignment_count = score_shape
        .iter()
        .try_fold(1_i32, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| geometry_error("selected-score shape overflows"))?;
    if selected_expert_count == 0 || assignment_count == 0 {
        return empty_weighted_outputs(runtime, &score_shape, sorted_expert_outputs);
    }
    if sorted_output_shape[0] != assignment_count || inverse_order.shape() != [assignment_count] {
        return Err(geometry_error(
            "sorted expert outputs and inverse order must match selected scores",
        ));
    }
    let output_dimension = sorted_output_shape[2];
    let mut weighted_output_shape = score_shape;
    weighted_output_shape.pop();
    weighted_output_shape.push(output_dimension);
    let output_element_count = assignment_count
        .checked_div(selected_expert_count)
        .and_then(|token_count| token_count.checked_mul(output_dimension))
        .ok_or_else(|| geometry_error("weighted expert output shape overflows"))?;
    let mut kernel_outputs = runtime.apply_metal_kernel(
        weighted_sum_kernel,
        &[sorted_expert_outputs, inverse_order, selected_scores],
        &[MlxMetalKernelOutput::new(
            weighted_output_shape,
            sorted_expert_outputs.dtype(),
        )],
        [output_element_count, 1, 1],
        [256.min(output_element_count), 1, 1],
        &[
            MlxMetalKernelTemplateArgument::Dtype {
                name: "OutputT",
                dtype: sorted_expert_outputs.dtype(),
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "selected_expert_count",
                integer_template_argument: selected_expert_count,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "output_dimension",
                integer_template_argument: output_dimension,
            },
        ],
    )?;
    kernel_outputs
        .pop()
        .ok_or_else(|| geometry_error("sorted expert weighted-sum kernel returned no output"))
}

fn empty_weighted_outputs(
    runtime: &MlxRuntime,
    score_shape: &[i32],
    expert_outputs: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let output_dimension = expert_outputs.shape().last().copied().unwrap_or(0);
    let mut weighted_output_shape = score_shape.to_vec();
    if !weighted_output_shape.is_empty() {
        weighted_output_shape.pop();
    }
    weighted_output_shape.push(output_dimension);
    runtime.zeros(&weighted_output_shape, expert_outputs.dtype())
}

fn geometry_error(description: &'static str) -> MlxRuntimeError {
    SparseExpertError::InvalidAssignmentGeometry { description }.into_runtime_error(OPERATION)
}
