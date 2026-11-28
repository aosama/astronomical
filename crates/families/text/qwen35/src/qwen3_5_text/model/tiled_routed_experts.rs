//! A prompt chunk's routed experts through the tiled sorted-expert kernels:
//! gate, up and their SwiGLU in one kernel straight from the token rows
//! (gate and up are never written), then the tiled down projection over the
//! sorted activations, all with MLX's bits.
//!
//! On a 35B-A3B layer over a 2,048-token chunk this took 17.2% less GPU time
//! than the route before it (the sorted-expert matmul for gate and for up,
//! MLX's compiled SwiGLU, MLX's sorted `gather_qmm` for down), faster in 12
//! of 12 rounds, with the same down outputs to the bit
//! (`cargo measure-sorted-expert-tiled-layer-verdict`, M5 Pro, 2026-11-27).

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use super::streamed_expert_assignment_steps::SortedAssignments;
use super::token_row_gate_up::{MINIMUM_AVERAGE_ROWS_PER_EXPERT, experts_per_token, is_refusal};
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_text::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;
use crate::qwen3_5_text::model::Qwen3_5Model;
use crate::sorted_expert_matmul::{
    ExpertStack, SortedExpertProjectionError, TILED_WEIGHT_BITS,
    sorted_expert_swiglu_from_token_rows, tiled_sorted_expert_matmul,
};
use jangtq2_qwen35::configuration::Jangtq2CodebookRotation;

impl Qwen3_5Model {
    /// The chunk's routed down outputs `[assignments, 1, hidden]` in sorted
    /// order, or `None` when it keeps the routes before the tiled kernels: no
    /// proven kernels for the stacks' width, an unsorted or short chunk,
    /// rotated activations, or weights the kernels do not read (they refuse
    /// them rather than misread them).
    pub(super) fn tiled_routed_expert_outputs(
        &self,
        hidden_states: &MlxArray,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        sorted_assignments: Option<&SortedAssignments>,
        selected_expert_indices: &MlxArray,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        let accelerators = &self.optional_accelerators;
        let (Some(kernels), Some((_, sorted_indices, _, sorted_order))) = (
            accelerators.sorted_expert_tiled_kernels.as_ref(),
            sorted_assignments,
        ) else {
            return Ok(None);
        };
        if accelerators
            .sorted_expert_matmul_paused_for_measurement
            .get()
            || !self.takes_tiled_routed_experts(paged_expert_weights, selected_expert_indices)
        {
            return Ok(None);
        }
        let (Some(gate), Some(up), Some(down)) = (
            tiled_expert_stack_of(&paged_expert_weights.gate_projection),
            tiled_expert_stack_of(&paged_expert_weights.up_projection),
            tiled_expert_stack_of(&paged_expert_weights.down_projection),
        ) else {
            return Ok(None);
        };
        let experts_per_token = experts_per_token(selected_expert_indices)?;
        let routed_outputs = match sorted_expert_swiglu_from_token_rows(
            &self.runtime,
            kernels,
            (hidden_states, sorted_order, sorted_indices),
            experts_per_token,
            (&gate, &up),
        ) {
            Ok(activated) => tiled_sorted_expert_matmul(
                &self.runtime,
                kernels,
                &activated,
                &down,
                sorted_indices,
            ),
            Err(projection_error) => Err(projection_error),
        };
        match routed_outputs {
            Ok(routed_outputs) => Ok(Some(routed_outputs)),
            Err(refusal) if is_refusal(&refusal) => Ok(None),
            Err(projection_error) => Err(tiled_kernel_error(projection_error)),
        }
    }

    /// Whether the chunk is one the kernels were measured on: no activation
    /// rotation on any of the three projections, and at least
    /// `MINIMUM_AVERAGE_ROWS_PER_EXPERT` rows an expert on average.
    fn takes_tiled_routed_experts(
        &self,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_expert_indices: &MlxArray,
    ) -> bool {
        let rotations_absent = [
            paged_expert_weights.gate_activation_rotation,
            paged_expert_weights.up_activation_rotation,
            paged_expert_weights.down_activation_rotation,
        ]
        .iter()
        .all(|rotation| *rotation == Jangtq2CodebookRotation::NoRotation);
        rotations_absent
            && selected_expert_indices.element_count()
                >= MINIMUM_AVERAGE_ROWS_PER_EXPERT * self.config.expert_count() as usize
    }
}

/// The kernels' view of an affine group-64 stack at a width they read
/// (`TILED_WEIGHT_BITS`); `None` for any other packing, which keeps MLX's
/// route. Whether this GPU proved that width is the kernels' own refusal.
fn tiled_expert_stack_of(affine_weights: &Qwen3_5AffineWeights) -> Option<ExpertStack<'_>> {
    match affine_weights {
        Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            quantization_bits,
            quantization_group_size: 64,
        } if TILED_WEIGHT_BITS.contains(quantization_bits) => Some(ExpertStack {
            packed_weight,
            scales: quantization_scales,
            biases: quantization_biases,
        }),
        _ => None,
    }
}

fn tiled_kernel_error(projection_error: SortedExpertProjectionError) -> TextExecutionError {
    TextExecutionError::Runtime(runtime_integration::MlxRuntimeError::RuntimeOperation {
        operation: "multiply the routed experts through the tiled sorted-expert kernels",
        description: projection_error.to_string(),
    })
}
