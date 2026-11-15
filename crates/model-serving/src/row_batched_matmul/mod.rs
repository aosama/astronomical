//! Row-batched 4-bit affine matmul for 1 to 32 token rows on M5 matrix units.
//!
//! Speculative decoding verifies a block of drafted tokens in one target
//! pass, which only pays when that pass costs about as much as one row. Stock
//! MLX does not give that on M5 (see `kernel_source`); this kernel does by
//! reading each weight group once for every row.

#[cfg_attr(not(feature = "direct-mlx"), allow(dead_code))]
pub(crate) mod dispatch_plan;
#[cfg(feature = "direct-mlx")]
pub(crate) mod kernel_source;
#[cfg(feature = "direct-mlx")]
pub mod row_batched_projection;

#[cfg(test)]
mod dispatch_plan_tests;

pub use dispatch_plan::{ROW_BATCHED_MATMUL_MAXIMUM_ROWS, RowBatchedDispatchRefusal};
#[cfg(feature = "direct-mlx")]
pub use row_batched_projection::{
    RowBatchedMatmulKernels, RowBatchedProjection, RowBatchedProjectionError,
};
