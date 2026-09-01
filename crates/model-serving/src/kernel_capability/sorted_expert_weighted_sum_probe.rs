//! Bounded capability probe for the sorted expert weighted-sum Metal kernel.
//!
//! The probe launches the real kernel on a minimal two-token, three-expert
//! shape with fixed inputs and validates the reduction against hand-computed
//! expected values. Compile failures, launch failures, and wrong values —
//! including the all-zeros signature of a silently dropped dispatch — each
//! map to a distinct typed capability error.

use astronomical_runtime_integration::{MlxMetalKernel, MlxRuntime, MlxRuntimeError};

use super::{
    CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError, validate_probe_outputs,
};
use crate::performance_attribution::PerformanceAttribution;
use crate::sparse_experts::{
    sort_expert_assignments, sorted_expert_weighted_sum, sorted_expert_weighted_sum_kernel,
};

// Expected reduction of the fixed probe inputs, proven by the existing
// direct-MLX sorted-reduction contracts: the inverse order maps token 0 to
// sorted-output rows [150,151], [100,101], [140,141] with scores
// [0.1, 0.2, 0.7] -> [133, 134], and token 1 to rows [110,111], [130,131],
// [120,121] with scores [0.25, 0.25, 0.5] -> [120, 121].
const PROBE_EXPECTED_WEIGHTED_OUTPUTS: [f32; 4] = [133.0, 134.0, 120.0, 121.0];

pub struct SortedExpertWeightedSumProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl<'runtime> SortedExpertWeightedSumProbe<'runtime> {
    #[must_use]
    pub const fn new(runtime: &'runtime MlxRuntime) -> Self {
        Self { runtime }
    }
}

impl CustomMetalKernelProbe for SortedExpertWeightedSumProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        CustomMetalKernelFamily::SortedExpertWeightedSum
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let mut probe_attribution = PerformanceAttribution::disabled();
        let kernel = sorted_expert_weighted_sum_kernel().map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let probe_output =
            sorted_expert_weighted_sum_probe_journey(self.runtime, &kernel, &mut probe_attribution)
                .map_err(|error| KernelCapabilityError::Execution {
                    description: error.to_string(),
                })?;
        validate_probe_outputs(&probe_output, &PROBE_EXPECTED_WEIGHTED_OUTPUTS)
    }
}

/// Runs the full probe journey: sort fixed assignments, gather outputs, and
/// reduce through the real Metal kernel, then read the values back to the
/// host so validation covers actual execution rather than lazy graph state.
fn sorted_expert_weighted_sum_probe_journey(
    runtime: &MlxRuntime,
    kernel: &MlxMetalKernel,
    probe_attribution: &mut PerformanceAttribution,
) -> Result<Vec<f32>, MlxRuntimeError> {
    let expanded_states = runtime
        .array_from_f32(&[10.0, 11.0, 20.0, 21.0], &[1, 2, 2])
        .and_then(|hidden_states| runtime.expand_dims(&hidden_states, -2))
        .and_then(|expanded| runtime.expand_dims(&expanded, -3))?;
    let selected_indices = runtime.array_from_u32(&[5, 0, 4, 1, 3, 2], &[1, 2, 3])?;
    let sorted_assignments = sort_expert_assignments(
        runtime,
        &expanded_states,
        &selected_indices,
        probe_attribution,
    )?;
    let sorted_expert_outputs = runtime.array_from_f32(
        &[
            100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0, 140.0, 141.0, 150.0, 151.0,
        ],
        &[6, 1, 2],
    )?;
    let selected_scores = runtime.array_from_f32(&[0.1, 0.2, 0.7, 0.25, 0.25, 0.5], &[1, 2, 3])?;
    let weighted_outputs = sorted_expert_weighted_sum(
        runtime,
        kernel,
        &sorted_expert_outputs,
        &sorted_assignments.inverse_order,
        &selected_scores,
        probe_attribution,
    )?;
    weighted_outputs.to_vec_f32()
}
