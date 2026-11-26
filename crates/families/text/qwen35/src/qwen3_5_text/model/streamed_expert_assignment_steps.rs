//! The steps around one streamed expert chunk's projections: ordering the
//! routed assignments for the gather, sharing the gate and up input rotation,
//! and shaping the rows the weighted reduction consumes.

use jangtq2_qwen35::configuration::Jangtq2CodebookRotation;
use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use super::paged_execution::RoutedExpertAssignmentOutputs;
use super::routing::qwen3_5_moe_sort_expert_assignments;
use crate::qwen3_5_text::model::Qwen3_5Model;
use crate::sparse_experts;
use crate::{PerformanceAttribution, PerformanceOperation};

/// Routed assignments ordered by expert: the sorted states, the sorted expert
/// indices, and the inverse order that restores the original assignments.
pub(super) type SortedAssignments = (MlxArray, MlxArray, MlxArray);

impl Qwen3_5Model {
    /// Sorts the routed assignments by expert when the sorted reduction applies
    /// for this many assignments and this worker's kernel, `None` otherwise.
    pub(super) fn sorted_assignments_when_worthwhile(
        &self,
        expanded_states: &MlxArray,
        selected_expert_indices: &MlxArray,
    ) -> Result<Option<SortedAssignments>, TextExecutionError> {
        if !sparse_experts::should_use_sorted_expert_reduction(
            selected_expert_indices.element_count(),
            self.sorted_expert_weighted_sum_kernel.is_some(),
        ) {
            return Ok(None);
        }
        Ok(Some(qwen3_5_moe_sort_expert_assignments(
            &self.runtime,
            expanded_states,
            selected_expert_indices,
        )?))
    }

    /// One rotated copy of the activations for the gate and up projections
    /// together, when both were quantized under the Hadamard-32 rotation.
    pub(super) fn shared_gate_up_activations(
        &self,
        expert_inputs: &MlxArray,
        gate_activation_rotation: Jangtq2CodebookRotation,
        up_activation_rotation: Jangtq2CodebookRotation,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        if !gate_and_up_share_hadamard32_rotation(gate_activation_rotation, up_activation_rotation)
        {
            return Ok(None);
        }
        // Both projections use the same input and transform, so build
        // one rotated tensor and share it instead of repeating the
        // blockwise Hadamard work for gate and up independently.
        Ok(Some(performance_attribution.measure_operation(
            PerformanceOperation::Jangtq2ActivationRotationHadamard32,
            |_| {
                jangtq2_qwen35::mlx_decode::rotate_activations_hadamard32(
                    &self.runtime,
                    expert_inputs,
                )
            },
        )?))
    }

    /// Wraps the down projection's rows: sorted rows travel with the inverse
    /// order the sorted reduction needs, unsorted rows lose their singleton
    /// axis.
    pub(super) fn routed_assignment_outputs_from(
        &self,
        selected_outputs: MlxArray,
        sorted_assignments: Option<SortedAssignments>,
        indices_are_sorted: bool,
    ) -> Result<RoutedExpertAssignmentOutputs, TextExecutionError> {
        if indices_are_sorted {
            let Some((_, _, inverse_order)) = sorted_assignments else {
                return Err(TextExecutionError::InvalidInput {
                    description: "sorted routed assignments lost their inverse order",
                });
            };
            Ok(RoutedExpertAssignmentOutputs::Sorted {
                sorted_outputs: selected_outputs,
                inverse_order,
            })
        } else {
            let assignment_outputs = self.runtime.squeeze_axis(&selected_outputs, -2)?;
            Ok(RoutedExpertAssignmentOutputs::Unsorted { assignment_outputs })
        }
    }
}

/// Whether the gate and up projections were both quantized under the
/// Hadamard-32 rotation, so one rotated copy of the activations serves both.
pub(super) fn gate_and_up_share_hadamard32_rotation(
    gate_activation_rotation: Jangtq2CodebookRotation,
    up_activation_rotation: Jangtq2CodebookRotation,
) -> bool {
    gate_activation_rotation == Jangtq2CodebookRotation::Hadamard32
        && up_activation_rotation == Jangtq2CodebookRotation::Hadamard32
}

/// The rotation the gate and up projections each still apply to their input:
/// none once a shared rotated copy exists, their own otherwise.
pub(super) fn remaining_gate_up_rotations(
    has_shared_rotated_copy: bool,
    gate_activation_rotation: Jangtq2CodebookRotation,
    up_activation_rotation: Jangtq2CodebookRotation,
) -> (Jangtq2CodebookRotation, Jangtq2CodebookRotation) {
    if has_shared_rotated_copy {
        (
            Jangtq2CodebookRotation::NoRotation,
            Jangtq2CodebookRotation::NoRotation,
        )
    } else {
        (gate_activation_rotation, up_activation_rotation)
    }
}

#[cfg(test)]
#[path = "streamed_expert_assignment_steps_tests.rs"]
mod streamed_expert_assignment_steps_tests;
