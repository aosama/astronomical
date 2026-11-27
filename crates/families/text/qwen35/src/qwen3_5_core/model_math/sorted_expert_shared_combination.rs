//! One Metal pass that reduces sorted routed-expert outputs and adds the gated
//! shared expert, replacing the sorted weighted-sum kernel plus the compiled
//! sigmoid-multiply-add graph on the prefill path.
//!
//! The two-step path writes the `[tokens, hidden]` sparse sum to memory and
//! reads it back in a second launch. This kernel keeps that sum in a register.
//! It must stay bit for bit equal to the two-step path, so every intermediate
//! is rounded to bfloat16 exactly where the two-step path stores it: the
//! Float32 weighted sum (the same Metal statements as the sorted weighted-sum
//! kernel) becomes bfloat16 before the add, the sigmoid is MLX's own `Sigmoid`
//! expression evaluated on bfloat16, and the multiply and add each run on
//! bfloat16 operands as MLX's compiled graph does.

use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::metal_kernels::ASTRO_FAMILY_QWEN35_SORTED_EXPERT_SHARED_COMBINATION;
use crate::sparse_experts::SORTED_EXPERT_REDUCTION_THREADS_PER_THREADGROUP;
use model_primitives::metal_kernel_declaration::declare_metal_kernel;

const COMBINATION_OPERATION: &str = "combine sorted and shared experts";

/// Builds the fused sorted-reduction and shared-expert combination kernel
/// (`metal/astro_family_qwen35_sorted_expert_shared_combination.metal`).
pub fn qwen3_5_sorted_expert_shared_combination_kernel() -> Result<MlxMetalKernel, MlxRuntimeError>
{
    declare_metal_kernel(&ASTRO_FAMILY_QWEN35_SORTED_EXPERT_SHARED_COMBINATION, &[])
}

/// The routed and shared expert tensors one fused combination consumes.
pub struct SortedExpertSharedCombinationInputs<'arrays> {
    /// Sorted down-projection outputs, `[assignments, 1, hidden]`.
    pub sorted_expert_outputs: &'arrays MlxArray,
    /// Sorted row of each original assignment, `[assignments]` UInt32.
    pub inverse_order: &'arrays MlxArray,
    /// Router scores in original assignment order, `[..., top_k]`.
    pub selected_scores: &'arrays MlxArray,
    /// Shared expert down-projection output, `[..., hidden]`.
    pub shared_expert_output: &'arrays MlxArray,
    /// Shared expert gate logits, one per token, `[..., 1]`.
    pub shared_expert_gate_logits: &'arrays MlxArray,
}

/// Whether these inputs have the dtypes the fused kernel is proven for.
///
/// This checks dtypes only; shapes are validated when the kernel launches.
/// The bit-for-bit contract is proven for bfloat16 activations alone, the
/// dtype every Qwen3.5 MoE artifact runs in: Float16 rounds the sigmoid and
/// the multiply-add at different points, and mixed dtypes make MLX promote
/// inside the compiled graph, so both keep the two-step path.
#[must_use]
pub fn can_fuse_sorted_expert_shared_combination(
    combination_inputs: &SortedExpertSharedCombinationInputs<'_>,
) -> bool {
    combination_inputs.sorted_expert_outputs.dtype() == MlxDtype::BFloat16
        && combination_inputs.shared_expert_output.dtype() == MlxDtype::BFloat16
        && combination_inputs.shared_expert_gate_logits.dtype() == MlxDtype::BFloat16
        && combination_inputs.inverse_order.dtype() == MlxDtype::UInt32
        && matches!(
            combination_inputs.selected_scores.dtype(),
            MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
        )
}

/// The fused combination when these inputs have the dtypes its bit-for-bit
/// contract is proven for, and `None` otherwise, for a caller that then
/// reduces first and combines after.
pub fn fused_sorted_expert_shared_combination_if_proven(
    runtime: &MlxRuntime,
    kernel: &MlxMetalKernel,
    inputs: &SortedExpertSharedCombinationInputs<'_>,
) -> Result<Option<MlxArray>, MlxRuntimeError> {
    if !can_fuse_sorted_expert_shared_combination(inputs) {
        return Ok(None);
    }
    Ok(Some(qwen3_5_sorted_expert_shared_combination(
        runtime, kernel, inputs,
    )?))
}

/// Reduces sorted expert outputs and adds the sigmoid-gated shared expert.
pub fn qwen3_5_sorted_expert_shared_combination(
    runtime: &MlxRuntime,
    combination_kernel: &MlxMetalKernel,
    combination_inputs: &SortedExpertSharedCombinationInputs<'_>,
) -> Result<MlxArray, MlxRuntimeError> {
    let geometry = validated_combination_geometry(combination_inputs)?;
    let output_element_count = geometry
        .token_count
        .checked_mul(geometry.output_dimension)
        .ok_or_else(|| combination_error("combined expert output shape overflows"))?;
    let mut kernel_outputs = runtime.apply_metal_kernel(
        combination_kernel,
        &[
            combination_inputs.sorted_expert_outputs,
            combination_inputs.inverse_order,
            combination_inputs.selected_scores,
            combination_inputs.shared_expert_output,
            combination_inputs.shared_expert_gate_logits,
        ],
        &[MlxMetalKernelOutput::new(
            combination_inputs.shared_expert_output.shape(),
            MlxDtype::BFloat16,
        )],
        [output_element_count, 1, 1],
        [
            SORTED_EXPERT_REDUCTION_THREADS_PER_THREADGROUP.min(output_element_count),
            1,
            1,
        ],
        &[
            MlxMetalKernelTemplateArgument::Dtype {
                name: "OutputT",
                dtype: MlxDtype::BFloat16,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "selected_expert_count",
                integer_template_argument: geometry.selected_expert_count,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "output_dimension",
                integer_template_argument: geometry.output_dimension,
            },
        ],
    )?;
    kernel_outputs
        .pop()
        .ok_or_else(|| combination_error("the fused expert combination kernel returned no output"))
}

struct CombinationGeometry {
    token_count: i32,
    selected_expert_count: i32,
    output_dimension: i32,
}

fn validated_combination_geometry(
    combination_inputs: &SortedExpertSharedCombinationInputs<'_>,
) -> Result<CombinationGeometry, MlxRuntimeError> {
    if !can_fuse_sorted_expert_shared_combination(combination_inputs) {
        return Err(combination_error(
            "the fused expert combination needs bfloat16 activations and UInt32 inverse order",
        ));
    }
    let sorted_output_shape = combination_inputs.sorted_expert_outputs.shape();
    let score_shape = combination_inputs.selected_scores.shape();
    if sorted_output_shape.len() != 3 || sorted_output_shape[1] != 1 || score_shape.len() < 2 {
        return Err(combination_error(
            "sorted outputs must be [assignments, 1, hidden] and scores must carry a top-k axis",
        ));
    }
    let (token_shape, selected_expert_count) = score_shape.split_at(score_shape.len() - 1);
    let selected_expert_count = selected_expert_count[0];
    let token_count = checked_token_count(token_shape)?;
    let output_dimension = sorted_output_shape[2];
    let assignment_count = token_count
        .checked_mul(selected_expert_count)
        .ok_or_else(|| combination_error("the assignment count overflows"))?;
    if token_count == 0
        || selected_expert_count == 0
        || sorted_output_shape[0] != assignment_count
        || combination_inputs.inverse_order.shape() != [assignment_count]
    {
        return Err(combination_error(
            "sorted outputs and inverse order must hold every token's top-k assignments",
        ));
    }
    validate_shared_expert_shapes(combination_inputs, token_shape, output_dimension)?;
    Ok(CombinationGeometry {
        token_count,
        selected_expert_count,
        output_dimension,
    })
}

/// Product of the token axes, refusing a count that does not fit `i32`.
fn checked_token_count(token_shape: &[i32]) -> Result<i32, MlxRuntimeError> {
    let mut token_count: i32 = 1;
    for token_axis_length in token_shape {
        token_count = token_count
            .checked_mul(*token_axis_length)
            .ok_or_else(|| combination_error("the token shape overflows"))?;
    }
    Ok(token_count)
}

/// The shared expert output must be `[..., hidden]` and its gate `[..., 1]`
/// over the same token axes as the router scores.
fn validate_shared_expert_shapes(
    combination_inputs: &SortedExpertSharedCombinationInputs<'_>,
    token_shape: &[i32],
    output_dimension: i32,
) -> Result<(), MlxRuntimeError> {
    let mut expected_shared_output_shape = token_shape.to_vec();
    expected_shared_output_shape.push(output_dimension);
    let mut expected_gate_logit_shape = token_shape.to_vec();
    expected_gate_logit_shape.push(1);
    if combination_inputs.shared_expert_output.shape() != expected_shared_output_shape
        || combination_inputs.shared_expert_gate_logits.shape() != expected_gate_logit_shape
    {
        return Err(combination_error(
            "the shared expert output and gate must match the routed tokens",
        ));
    }
    Ok(())
}

fn combination_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: COMBINATION_OPERATION,
        description: description.to_owned(),
    }
}
