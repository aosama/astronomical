//! Rows times a projection kept in MLX's own layout.
//!
//! MLX's quantized matmul serves every row count of such a projection. Until
//! MLX 0.32.5 it was slow for a handful of rows (over the dense 27B's 64
//! layers, 132 ms at 8 rows and 200 ms at 16 against 52 ms at one), and a
//! pass of 8 to 32 rows took this repository's row-batched kernel instead
//! (92 to 96 ms). MLX 0.32.5 added its own kernels for 3 to 16 rows: the
//! same passes now cost 62 ms at 8 rows and 61 ms at 16, and 101 ms at 32
//! where the row-batched kernel takes 167 (`cargo measure-k-row-verify-cost`),
//! so every pass goes to MLX.
//!
//! MLX's few-row kernels are fast but not row-exact: a row among four or
//! more differs from the same row alone by a rounding step (`cargo
//! measure-batched-kernel-row-exactness`). A model whose drafted reply must
//! equal its plain one converts to register order, which costs the same at
//! 8 to 16 rows and gives every row its one-row bits (issue #2599).

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use super::base::Qwen3_5ModelBase;

/// MLX's packed weight with its scales and biases, as bound.
pub struct QuantizedProjectionParts<'weights> {
    pub packed_weight: &'weights MlxArray,
    pub scales: &'weights MlxArray,
    pub biases: &'weights MlxArray,
    pub group_size: i32,
    pub bits: i32,
}

impl Qwen3_5ModelBase {
    /// `activations` [..., input] times a projection in MLX's layout.
    pub fn quantized_rows_matmul(
        &self,
        activations: &MlxArray,
        projection: &QuantizedProjectionParts<'_>,
    ) -> Result<MlxArray, TextExecutionError> {
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
