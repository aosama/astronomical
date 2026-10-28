//! Gathered matrix multiplication over canonical stacked expert arrays.
//!
//! A sparse Mixture-of-Experts layer does not run every expert for every token.
//! The router chooses a small list of expert IDs for each token, and a gathered
//! matrix multiplication multiplies each token row by only those expert matrices.
//! MLX performs that selection inside `gather_mm` or `gather_qmm`; selecting the
//! full matrices first would create a potentially enormous temporary tensor.
//!
//! Model families remain responsible for everything that is not identical math:
//! routing formulas, top-K policy, expert storage, paging, residency, and the
//! decision to sort assignments. This module receives already-canonical arrays
//! and owns only the shared dense-versus-affine dispatch and ordering contract.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};
use mlx_c_rust::{MlxArray, MlxCError};

/// Tells MLX whether the supplied expert-ID array is sorted in ascending order.
///
/// This is a semantic type instead of a bare `bool` so a call site says what it
/// believes about the indices. `SortedByExpert` is an explicit assertion, not a
/// request to sort. This function never rearranges IDs. Callers must select that
/// variant only after applying the permutation produced by
/// [`super::sort_expert_assignments`] to both IDs and activation rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertAssignmentOrder {
    /// IDs retain router order and may jump between experts arbitrarily.
    Original,
    /// IDs and matching activation rows are already grouped by expert ID.
    SortedByExpert,
}

impl ExpertAssignmentOrder {
    /// Converts the readable Rust contract to MLX's `sorted_indices` flag.
    ///
    /// MLX uses this promise to choose a gathered kernel that can exploit
    /// contiguous expert reads. Giving MLX `true` for unsorted IDs is therefore
    /// a correctness-contract violation even if a small example appears to work.
    #[must_use]
    pub const fn uses_sorted_indices(self) -> bool {
        matches!(self, Self::SortedByExpert)
    }
}

/// One canonical stacked expert projection, independent of model-family storage.
///
/// This enum deliberately borrows arrays. It does not copy, retain, page, or
/// reinterpret model weights. The family owner keeps every array alive while the
/// lazy MLX graph references it.
pub enum StackedExpertProjection<'a> {
    /// Dense weights already transposed to `[experts, input, output]`.
    ///
    /// MLX matrix multiplication expects the contracted input dimension before
    /// the output dimension. Checkpoints commonly store dense linear weights as
    /// `[experts, output, input]`, so the family adapter performs one lazy axis
    /// transpose before constructing this variant.
    Dense { transposed_weights: &'a MlxArray },
    /// MLX affine weights in `[experts, output, packed_input]` form.
    ///
    /// Packed affine weights remain in MLX's quantized layout. Scales and biases
    /// describe each quantization group; `group_size` and `bits` tell MLX how to
    /// decode the packed input axis. `gather_qmm` consumes these arrays directly,
    /// so this path never dequantizes complete expert matrices into a temporary.
    Affine {
        packed_weights: &'a MlxArray,
        scales: &'a MlxArray,
        biases: &'a MlxArray,
        group_size: i32,
        bits: i32,
    },
}

/// Applies one stacked expert projection without materializing selected weights.
///
/// `activations` contain one matrix row per assignment (or broadcastable token
/// rows), while `selected_expert_indices` chooses the expert matrix for each row.
/// Only right-hand-side indices are supplied because the gathered side is the
/// stacked expert-weight tensor. MLX derives left-hand-side batch indices from
/// the activation shape and broadcasts both index arrays to one assignment shape.
///
/// Attribution measures graph construction for the selected MLX gathered
/// operation. Evaluation remains lazy and is attributed at the request's later
/// evaluation boundary, consistent with the rest of Astronomical's MLX graph.
pub fn gather_expert_projection(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    projection: StackedExpertProjection<'_>,
    selected_expert_indices: &MlxArray,
    assignment_order: ExpertAssignmentOrder,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, MlxRuntimeError> {
    // Keep attribution around the exact neutral operation. Every family now
    // reports gathered projection construction under the same catalog entry,
    // while its surrounding router and shared-expert work remain family-owned.
    performance_attribution
        .measure_operation(
            PerformanceOperation::GatheredExpertExecution,
            |_| match projection {
                StackedExpertProjection::Dense { transposed_weights } => gather_dense_projection(
                    runtime,
                    activations,
                    transposed_weights,
                    selected_expert_indices,
                    assignment_order,
                ),
                StackedExpertProjection::Affine {
                    packed_weights,
                    scales,
                    biases,
                    group_size,
                    bits,
                } => gather_affine_projection(
                    runtime,
                    activations,
                    &AffineExpertProjectionInputs {
                        packed_weights,
                        scales,
                        biases,
                        group_size,
                        bits,
                    },
                    selected_expert_indices,
                    assignment_order,
                ),
            },
        )
        .map_err(MlxRuntimeError::from)
}

/// The affine-quantized expert arrays one gathered projection reads,
/// grouped so `gather_affine_projection` receives one named payload
/// instead of a row of positional arrays (the pattern
/// `PrefillGatheredDequantInputs` established for the prefill kernel).
struct AffineExpertProjectionInputs<'a> {
    packed_weights: &'a MlxArray,
    scales: &'a MlxArray,
    biases: &'a MlxArray,
    group_size: i32,
    bits: i32,
}

/// Gathers the dense expert matrices for the routed assignments.
///
/// No explicit left index is needed: each activation batch row already
/// corresponds to its assignment position, and the right index chooses one
/// matrix from the expert axis.
fn gather_dense_projection(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    transposed_weights: &MlxArray,
    selected_expert_indices: &MlxArray,
    assignment_order: ExpertAssignmentOrder,
) -> Result<MlxArray, MlxCError> {
    runtime.gather_dense_matmul(
        activations,
        transposed_weights,
        None,
        Some(selected_expert_indices),
        assignment_order.uses_sorted_indices(),
    )
}

/// Gathers the affine-quantized expert matrices for the routed assignments.
///
/// As in the dense path, activation rows are already arranged for their
/// assignment positions, so only experts are gathered. Checkpoint affine
/// weights are `[expert, output, packed_input]`, so MLX logically transposes
/// each selected matrix for x @ Wᵀ.
fn gather_affine_projection(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    affine_projection_inputs: &AffineExpertProjectionInputs<'_>,
    selected_expert_indices: &MlxArray,
    assignment_order: ExpertAssignmentOrder,
) -> Result<MlxArray, MlxCError> {
    runtime.gather_quantized_matmul_affine(
        activations,
        affine_projection_inputs.packed_weights,
        affine_projection_inputs.scales,
        affine_projection_inputs.biases,
        None,
        Some(selected_expert_indices),
        true,
        affine_projection_inputs.group_size,
        affine_projection_inputs.bits,
        assignment_order.uses_sorted_indices(),
    )
}
