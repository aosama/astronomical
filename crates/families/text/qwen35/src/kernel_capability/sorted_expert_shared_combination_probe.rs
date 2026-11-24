//! Bounded capability probe for the fused sorted-expert and shared-expert
//! combination kernel.
//!
//! The fused kernel is only allowed to replace the two-step path when it is
//! bit for bit equal to it, and that equality depends on how this machine's
//! Metal compiler rounds bfloat16 arithmetic. So the probe does not check
//! against hand-computed values: it runs both paths on one small fixed input
//! and demands identical bits, plus finite, not-all-zero results so two
//! silently dropped dispatches cannot agree on zeros. The proof covers the
//! probe's own template specialization, so the probe uses production's eight
//! experts per token; the output width is a loop bound the combination tail
//! does not depend on.

use mlx_bridge::{MlxArray, MlxCompiledElementwiseGraphs, MlxDtype, MlxMetalKernel};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::{CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError};
use crate::qwen3_5_core::model_math::routing::{
    qwen3_5_moe_sorted_expert_weighted_sum, qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
use crate::qwen3_5_core::model_math::sorted_expert_shared_combination::{
    SortedExpertSharedCombinationInputs, qwen3_5_sorted_expert_shared_combination,
    qwen3_5_sorted_expert_shared_combination_kernel,
};

/// Registry identifier of the fused sorted-expert and shared-expert
/// combination family.
pub const SORTED_EXPERT_SHARED_COMBINATION: CustomMetalKernelFamily =
    CustomMetalKernelFamily::new("sorted_expert_shared_combination");

/// Supplies this family's probes for the once-per-process probe run.
pub fn capability_probes(runtime: &MlxRuntime) -> Vec<Box<dyn CustomMetalKernelProbe + '_>> {
    vec![Box::new(SortedExpertSharedCombinationProbe::new(runtime))]
}

pub struct SortedExpertSharedCombinationProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl<'runtime> SortedExpertSharedCombinationProbe<'runtime> {
    #[must_use]
    pub const fn new(runtime: &'runtime MlxRuntime) -> Self {
        Self { runtime }
    }
}

impl CustomMetalKernelProbe for SortedExpertSharedCombinationProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        SORTED_EXPERT_SHARED_COMBINATION
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let combination_kernel =
            qwen3_5_sorted_expert_shared_combination_kernel().map_err(|error| {
                KernelCapabilityError::Compilation {
                    description: error.to_string(),
                }
            })?;
        let (fused_bits, two_step_bits) =
            fused_and_two_step_probe_bits(self.runtime, &combination_kernel).map_err(|error| {
                KernelCapabilityError::Execution {
                    description: error.to_string(),
                }
            })?;
        validate_bit_identical_probe_outputs(&fused_bits, &two_step_bits)
    }
}

/// Experts per token in the probe; production's top-k, so the probe compiles
/// the same `selected_expert_count` specialization the MoE layers run.
const PROBE_SELECTED_EXPERT_COUNT: usize = 8;
const PROBE_TOKEN_COUNT: usize = 2;
const PROBE_OUTPUT_DIMENSION: usize = 2;

/// Probe arrays: two tokens routed to eight experts each, two output channels,
/// with gate logits on both sides of zero so both sigmoid branches run.
struct ProbeArrays {
    sorted_expert_outputs: MlxArray,
    inverse_order: MlxArray,
    selected_scores: MlxArray,
    shared_expert_output: MlxArray,
    shared_expert_gate_logits: MlxArray,
}

fn probe_arrays(runtime: &MlxRuntime) -> Result<ProbeArrays, MlxRuntimeError> {
    let assignment_count = PROBE_TOKEN_COUNT * PROBE_SELECTED_EXPERT_COUNT;
    // Fixed, sign-mixed, non-integer values so the Float32 sum rounds when it
    // narrows to bfloat16; a fixed reversal stands in for a sorted order.
    let mut sorted_output_values = Vec::with_capacity(assignment_count * PROBE_OUTPUT_DIMENSION);
    for sorted_value_index in 0..assignment_count * PROBE_OUTPUT_DIMENSION {
        let alternating_sign = if sorted_value_index % 3 == 0 {
            -1.0
        } else {
            1.0
        };
        sorted_output_values.push(alternating_sign * (0.375 + sorted_value_index as f32 * 0.21875));
    }
    let mut inverse_order_rows = Vec::with_capacity(assignment_count);
    let mut router_score_values = Vec::with_capacity(assignment_count);
    for assignment_index in 0..assignment_count {
        inverse_order_rows.push((assignment_count - 1 - assignment_index) as u32);
        router_score_values
            .push(0.03 + (assignment_index % PROBE_SELECTED_EXPERT_COUNT) as f32 * 0.025);
    }
    let sorted_expert_outputs = runtime.array_from_f32(
        &sorted_output_values,
        &[assignment_count as i32, 1, PROBE_OUTPUT_DIMENSION as i32],
    )?;
    let shared_expert_output = runtime.array_from_f32(&[1.5, -2.25, 3.0, 0.5], &[2, 2])?;
    let shared_expert_gate_logits = runtime.array_from_f32(&[0.75, -1.25], &[2, 1])?;
    Ok(ProbeArrays {
        sorted_expert_outputs: runtime.astype(&sorted_expert_outputs, MlxDtype::BFloat16)?,
        inverse_order: runtime.array_from_u32(&inverse_order_rows, &[assignment_count as i32])?,
        selected_scores: runtime.array_from_f32(
            &router_score_values,
            &[PROBE_TOKEN_COUNT as i32, PROBE_SELECTED_EXPERT_COUNT as i32],
        )?,
        shared_expert_output: runtime.astype(&shared_expert_output, MlxDtype::BFloat16)?,
        shared_expert_gate_logits: runtime
            .astype(&shared_expert_gate_logits, MlxDtype::BFloat16)?,
    })
}

/// Runs the fused kernel and the two-step path it replaces on the probe
/// arrays and reads both back as Float32 bit patterns; widening bfloat16 to
/// Float32 is exact, so equal patterns mean equal bfloat16 bits.
fn fused_and_two_step_probe_bits(
    runtime: &MlxRuntime,
    combination_kernel: &MlxMetalKernel,
) -> Result<(Vec<u32>, Vec<u32>), MlxRuntimeError> {
    let probe_arrays = probe_arrays(runtime)?;
    let fused_output = qwen3_5_sorted_expert_shared_combination(
        runtime,
        combination_kernel,
        &SortedExpertSharedCombinationInputs {
            sorted_expert_outputs: &probe_arrays.sorted_expert_outputs,
            inverse_order: &probe_arrays.inverse_order,
            selected_scores: &probe_arrays.selected_scores,
            shared_expert_output: &probe_arrays.shared_expert_output,
            shared_expert_gate_logits: &probe_arrays.shared_expert_gate_logits,
        },
    )?;
    let sparse_expert_output = qwen3_5_moe_sorted_expert_weighted_sum(
        runtime,
        &qwen3_5_moe_sorted_expert_weighted_sum_kernel()?,
        &probe_arrays.sorted_expert_outputs,
        &probe_arrays.inverse_order,
        &probe_arrays.selected_scores,
    )?;
    let two_step_output = runtime.apply_compiled_sparse_shared_expert_combination(
        &MlxCompiledElementwiseGraphs::new()?,
        &sparse_expert_output,
        &probe_arrays.shared_expert_output,
        &probe_arrays.shared_expert_gate_logits,
    )?;
    Ok((
        float32_bit_patterns(runtime, &fused_output)?,
        float32_bit_patterns(runtime, &two_step_output)?,
    ))
}

fn float32_bit_patterns(
    runtime: &MlxRuntime,
    bfloat16_output: &MlxArray,
) -> Result<Vec<u32>, MlxRuntimeError> {
    let widened_output = runtime.astype(bfloat16_output, MlxDtype::Float32)?;
    let widened_values = widened_output.to_vec_f32()?;
    let mut bit_patterns = Vec::with_capacity(widened_values.len());
    for widened_value in widened_values {
        bit_patterns.push(widened_value.to_bits());
    }
    Ok(bit_patterns)
}

/// Accepts only identical, finite outputs from the two paths that are not all
/// zero.
fn validate_bit_identical_probe_outputs(
    fused_bits: &[u32],
    two_step_bits: &[u32],
) -> Result<(), KernelCapabilityError> {
    if fused_bits != two_step_bits {
        return Err(KernelCapabilityError::OutputMismatch {
            description: format!(
                "the fused expert combination produced bits {fused_bits:?} where the two-step path produced {two_step_bits:?}"
            ),
        });
    }
    let mut has_non_zero_output = false;
    for bit_pattern in fused_bits {
        let combined_value = f32::from_bits(*bit_pattern);
        if !combined_value.is_finite() {
            return Err(KernelCapabilityError::OutputMismatch {
                description: format!("the expert combination produced {combined_value}"),
            });
        }
        has_non_zero_output |= combined_value != 0.0;
    }
    if !has_non_zero_output {
        return Err(KernelCapabilityError::OutputMismatch {
            description: "both expert combination paths produced only zeros".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "sorted_expert_shared_combination_probe_tests.rs"]
mod sorted_expert_shared_combination_probe_tests;
