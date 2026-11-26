//! Which kernel multiplies rows by a projection kept in MLX's own layout.
//!
//! MLX's quantized matmul is the fastest way to serve one row and a prompt
//! chunk, and the slowest for a handful of rows: over the dense 27B's 64
//! layers it takes 52 ms at one row, 132 ms at 8 and 200 ms at 16, the rows
//! of a dense draft tree. The row-batched kernel reads the same layout and
//! takes 92 ms at 8 rows and 96 ms at 16 (and 91 ms at 4, where MLX's 71 ms
//! still wins). So a pass of 8 to 32 rows takes the row-batched kernel and
//! every other pass keeps MLX's, with no weight converted or copied.
//!
//! Register order would serve those rows in about 65 ms, but it costs every
//! reply token and every prompt chunk about 8% (issue #2568, measured by
//! `cargo measure-register-order-served-rows`), so a model that keeps
//! MLX's layout takes this instead.
//!
//! It applies only once a model has declared the row-batched kernels, which
//! a model does when it hosts a drafter and this GPU's capability probe has
//! run them.

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use super::base::Qwen3_5ModelBase;
use super::register_ordered_projections::projection_failure;
use crate::row_batched_matmul::{
    row_batched_quantized_matmul, row_batched_quantized_matmul_accepts,
};

/// The fewest rows the row-batched kernel serves faster than MLX's.
const FEWEST_ROW_BATCHED_ROWS: i32 = 8;
/// The most rows one row-batched dispatch takes.
const MOST_ROW_BATCHED_ROWS: i32 = 32;

/// MLX's packed 4-bit weight with its scales and biases, as bound.
pub struct QuantizedProjectionParts<'weights> {
    pub packed_weight: &'weights MlxArray,
    pub scales: &'weights MlxArray,
    pub biases: &'weights MlxArray,
    pub group_size: i32,
    pub bits: i32,
}

/// Whether a pass of `row_count` rows is one the row-batched kernel serves
/// faster than MLX's.
#[must_use]
pub const fn row_batched_kernel_serves(row_count: i32) -> bool {
    row_count >= FEWEST_ROW_BATCHED_ROWS && row_count <= MOST_ROW_BATCHED_ROWS
}

impl Qwen3_5ModelBase {
    /// `activations` [..., input] times a projection in MLX's layout,
    /// through the kernel that serves this many rows fastest.
    pub fn quantized_rows_matmul(
        &self,
        activations: &MlxArray,
        projection: &QuantizedProjectionParts<'_>,
    ) -> Result<MlxArray, TextExecutionError> {
        let activations_shape = activations.shape();
        let row_count: i32 = activations_shape[..activations_shape.len().saturating_sub(1)]
            .iter()
            .product();
        if let Some(kernels) = &self.optional_accelerators.register_ordered_matmul_kernels
            && row_batched_kernel_serves(row_count)
            && row_batched_quantized_matmul_accepts(
                activations,
                projection.packed_weight,
                projection.scales,
                projection.biases,
            )
        {
            return row_batched_quantized_matmul(
                &self.runtime,
                kernels,
                activations,
                projection.packed_weight,
                projection.scales,
                projection.biases,
            )
            .map_err(projection_failure);
        }
        Ok(self.runtime.quantized_matmul_affine_on_platform(
            activations,
            projection.packed_weight,
            projection.scales,
            projection.biases,
            true,
            projection.group_size,
            projection.bits,
        )?)
    }
}

#[cfg(test)]
#[path = "several_row_projection_tests.rs"]
mod several_row_projection_tests;
