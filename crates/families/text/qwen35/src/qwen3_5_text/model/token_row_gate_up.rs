//! A prompt chunk's routed gate and up projections through Astronomical's
//! sorted-expert matmul, read straight from the token rows.
//!
//! The usual route takes every token's row once per assignment in sorted
//! order (a 64 MiB copy for a 2,048-token chunk of a 35B-A3B layer) and runs
//! MLX's `gather_qmm` over that copy for gate and for up. The sorted-expert
//! matmul reads each sorted row from the token's own row through the sort
//! order, with MLX's bits, so the copy is never made; over four layers of
//! gate and up it took 13% less GPU time
//! (`cargo measure-sorted-expert-token-rows-verdict`). The down projection
//! keeps MLX's kernel, which the measurements still found faster there.

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use super::streamed_expert_assignment_steps::SortedAssignments;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_text::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;
use crate::qwen3_5_text::model::Qwen3_5Model;
use crate::sorted_expert_matmul::{
    ExpertStack, SortedExpertProjectionError, sorted_expert_quantized_matmul_from_token_rows,
};
use jangtq2_qwen35::configuration::Jangtq2CodebookRotation;

/// The fewest sorted rows per expert, on average, at which a chunk takes the
/// sorted-expert matmul. Its gain was measured on 2,048-token chunks (64 rows
/// an expert); much shorter chunks leave most of each tile empty, and keep
/// MLX's route until a measurement says otherwise.
pub(super) const MINIMUM_AVERAGE_ROWS_PER_EXPERT: usize = 32;

impl Qwen3_5Model {
    /// `(gate, up)` outputs `[assignments, 1, intermediate]` in sorted order,
    /// or `None` when this chunk keeps MLX's route: no proven kernel, an
    /// unsorted or short chunk, rotated activations, or weights the kernel
    /// does not read (it refuses them rather than misread them).
    pub(super) fn token_row_gate_and_up(
        &self,
        hidden_states: &MlxArray,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        sorted_assignments: Option<&SortedAssignments>,
        selected_expert_indices: &MlxArray,
    ) -> Result<Option<(MlxArray, MlxArray)>, TextExecutionError> {
        let Some(kernels) = self
            .optional_accelerators
            .sorted_expert_matmul_kernels
            .as_ref()
        else {
            return Ok(None);
        };
        if self
            .optional_accelerators
            .sorted_expert_matmul_paused_for_measurement
            .get()
        {
            return Ok(None);
        }
        let Some((_, sorted_indices, _, sorted_order)) = sorted_assignments else {
            return Ok(None);
        };
        let assignment_count = selected_expert_indices.element_count();
        let rotations_absent = paged_expert_weights.gate_activation_rotation
            == Jangtq2CodebookRotation::NoRotation
            && paged_expert_weights.up_activation_rotation == Jangtq2CodebookRotation::NoRotation;
        if !rotations_absent
            || assignment_count
                < MINIMUM_AVERAGE_ROWS_PER_EXPERT * self.config.expert_count() as usize
        {
            return Ok(None);
        }
        let (Some(gate_stack), Some(up_stack)) = (
            expert_stack_of(&paged_expert_weights.gate_projection),
            expert_stack_of(&paged_expert_weights.up_projection),
        ) else {
            return Ok(None);
        };
        let experts_per_token =
            *selected_expert_indices
                .shape()
                .last()
                .ok_or(TextExecutionError::InvalidInput {
                    description: "routed expert indices have no top-k axis",
                })?;
        let routing = TokenRowRouting {
            token_rows: hidden_states,
            sorted_order,
            sorted_indices,
            experts_per_token,
        };
        match (
            routing.project(&self.runtime, kernels, &gate_stack),
            routing.project(&self.runtime, kernels, &up_stack),
        ) {
            (Ok(gate), Ok(up)) => Ok(Some((gate, up))),
            (Err(refusal), _) | (_, Err(refusal)) if is_refusal(&refusal) => Ok(None),
            (Err(projection_error), _) | (_, Err(projection_error)) => {
                Err(kernel_error(projection_error))
            }
        }
    }
}

/// What both projections of a chunk read: its token rows and its sort.
struct TokenRowRouting<'chunk> {
    token_rows: &'chunk MlxArray,
    sorted_order: &'chunk MlxArray,
    sorted_indices: &'chunk MlxArray,
    experts_per_token: i32,
}

impl TokenRowRouting<'_> {
    fn project(
        &self,
        runtime: &runtime_integration::MlxRuntime,
        kernels: &crate::sorted_expert_matmul::SortedExpertMatmulKernels,
        experts: &ExpertStack<'_>,
    ) -> Result<MlxArray, SortedExpertProjectionError> {
        sorted_expert_quantized_matmul_from_token_rows(
            runtime,
            kernels,
            (self.token_rows, self.sorted_order, self.sorted_indices),
            self.experts_per_token,
            experts,
        )
    }
}

/// The 4-bit stack a quantized projection holds, the only packing the kernel
/// reads; `None` for every other weight kind.
fn expert_stack_of(affine_weights: &Qwen3_5AffineWeights) -> Option<ExpertStack<'_>> {
    match affine_weights {
        Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            quantization_bits: 4,
            quantization_group_size: 64,
        } => Some(ExpertStack {
            packed_weight,
            scales: quantization_scales,
            biases: quantization_biases,
        }),
        _ => None,
    }
}

/// A shape or packing the kernel does not take: the chunk keeps MLX's route.
fn is_refusal(projection_error: &SortedExpertProjectionError) -> bool {
    matches!(
        projection_error,
        SortedExpertProjectionError::Refused(_)
            | SortedExpertProjectionError::UnsupportedQuantization { .. }
    )
}

fn kernel_error(projection_error: SortedExpertProjectionError) -> TextExecutionError {
    TextExecutionError::Runtime(runtime_integration::MlxRuntimeError::RuntimeOperation {
        operation: "multiply gate and up through the sorted-expert matmul",
        description: projection_error.to_string(),
    })
}
