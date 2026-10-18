//! Sibling rotated projections fused into one packed matmul.
//!
//! Decode on the folded pack is bound by how fast the 2-bit matrix-vector
//! kernel streams weights, and that rate falls with the row count of a
//! launch: a 10,240-row projection streams well below the 248,320-row
//! vocabulary head because its 1,280 threadgroups neither fill the graphics
//! processor nor amortize the launch ramp. Sibling projections that already
//! share one rotated input (gated-delta query/key/value and output gate,
//! full-attention query, key, and value, MLP gate and up) are therefore
//! stacked along the output axis at load, turning two or three short
//! launches into one longer launch per token: fewer launches and a higher
//! streaming rate per launch.
//!
//! The stacking changes no arithmetic. MLX's quantized matrix-vector kernel
//! computes every output row as an independent dot product over the
//! unchanged input width, so each member's slice of the fused output is
//! bit-identical to the member's own matmul, and the upstream parity
//! trajectory is preserved. Each member's output is a last-axis slice of the
//! fused output: a free view for the one-row decode shape, and a strided
//! view the next consumer materializes for prefill, where that copy is a
//! small fraction of the prefill matmul itself.

use astronomical_mlx_c_rust::{MlxArray, MlxCError};
use astronomical_runtime_integration::MlxRuntime;

use super::fwht::RotatedActivations;
use super::weights::{PackedProjection, slice_last_dimension_raw};

/// `MEMBER_COUNT` sibling rotated projections of one input width, fused into
/// a single packed matmul whose output rows are the members' rows stacked in
/// member order.
#[derive(Debug)]
pub(crate) struct FusedPackedProjection<const MEMBER_COUNT: usize> {
    stacked_projection: PackedProjection,
    /// Each member's exclusive end row within the fused output, so member
    /// `index` occupies rows `member_end_rows[index - 1]..member_end_rows[index]`.
    member_end_rows: [i32; MEMBER_COUNT],
}

impl<const MEMBER_COUNT: usize> FusedPackedProjection<MEMBER_COUNT> {
    /// Stacks the members' packed weights, scales, and biases along the
    /// output axis. The stacked arrays are lazy until the loader evaluates
    /// every fused projection of the pack in one pass (see
    /// [`Self::stacked_arrays`]), so no concatenation enters a forward graph
    /// and load pays one synchronization instead of one per layer.
    pub(crate) fn fuse(
        runtime: &MlxRuntime,
        members: [PackedProjection; MEMBER_COUNT],
    ) -> Result<Self, MlxCError> {
        let input_width = members[0].input_width();
        let mut member_end_rows = [0_i32; MEMBER_COUNT];
        let mut stacked_row_count = 0_i32;
        for (member_index, member) in members.iter().enumerate() {
            if member.input_width() != input_width {
                return Err(MlxCError {
                    operation: "fuse sibling rotated projections",
                    description: format!(
                        "member {member_index} has input width {} but the first member has {input_width}",
                        member.input_width()
                    ),
                });
            }
            stacked_row_count += member.output_rows();
            member_end_rows[member_index] = stacked_row_count;
        }
        let packed_weights: Vec<&MlxArray> = members
            .iter()
            .map(PackedProjection::packed_weight)
            .collect();
        let quantization_scales: Vec<&MlxArray> = members
            .iter()
            .map(PackedProjection::quantization_scales)
            .collect();
        let quantization_biases: Vec<&MlxArray> = members
            .iter()
            .map(PackedProjection::quantization_biases)
            .collect();
        let stacked_projection = PackedProjection::new(
            runtime.concatenate_axis(&packed_weights, 0)?,
            runtime.concatenate_axis(&quantization_scales, 0)?,
            runtime.concatenate_axis(&quantization_biases, 0)?,
            input_width,
        );
        Ok(Self {
            stacked_projection,
            member_end_rows,
        })
    }

    /// The three stacked arrays the loader materializes together.
    pub(crate) fn stacked_arrays(&self) -> [&MlxArray; 3] {
        [
            self.stacked_projection.packed_weight(),
            self.stacked_projection.quantization_scales(),
            self.stacked_projection.quantization_biases(),
        ]
    }

    /// Runs the one fused matmul and returns each member's output in member
    /// order, as last-axis views of the fused output.
    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        rotated_activations: &RotatedActivations,
    ) -> Result<[MlxArray; MEMBER_COUNT], MlxCError> {
        let fused_output = self.forward_fused(runtime, rotated_activations)?;
        let mut member_outputs = Vec::with_capacity(MEMBER_COUNT);
        let mut member_start_row = 0_i32;
        for member_end_row in self.member_end_rows {
            member_outputs.push(slice_last_dimension_raw(
                runtime,
                &fused_output,
                member_start_row,
                member_end_row,
            )?);
            member_start_row = member_end_row;
        }
        member_outputs.try_into().map_err(|_| MlxCError {
            operation: "split a fused rotated projection",
            description: format!("the fused output did not split into {MEMBER_COUNT} members"),
        })
    }

    /// Runs the fused matmul alone, for callers that time the launch rather
    /// than consume the members.
    pub(crate) fn forward_fused(
        &self,
        runtime: &MlxRuntime,
        rotated_activations: &RotatedActivations,
    ) -> Result<MlxArray, MlxCError> {
        self.stacked_projection
            .forward(runtime, rotated_activations)
    }

    pub(crate) const fn input_width(&self) -> i32 {
        self.stacked_projection.input_width()
    }
}
