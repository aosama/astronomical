//! Row-batched 4-bit affine matmul for 1 to 32 token rows on M5 matrix units,
//! and its register-ordered successor for 1 to 128 rows per dispatch.
//!
//! Speculative decoding verifies a block of drafted tokens in one target
//! pass, which only pays when that pass costs about as much as one row. Stock
//! MLX does not give that on M5 (see `kernel_source`); these kernels do by
//! reading each weight group once for every row.

pub mod dispatch_plan;
#[cfg(feature = "direct-mlx")]
pub mod kernel_source;
#[cfg(feature = "direct-mlx")]
pub mod register_order_restoration;
#[cfg(feature = "direct-mlx")]
pub mod register_order_source;
#[cfg(feature = "direct-mlx")]
pub mod register_ordered_matmul;
pub mod register_ordered_plan;
#[cfg(feature = "direct-mlx")]
pub mod row_batched_quantized_matmul;

#[cfg(test)]
mod dispatch_plan_tests;
#[cfg(test)]
mod register_ordered_plan_tests;

pub use dispatch_plan::{ROW_BATCHED_MATMUL_MAXIMUM_ROWS, RowBatchedDispatchRefusal};
#[cfg(feature = "direct-mlx")]
pub use register_order_restoration::RestoredMlxRows;
#[cfg(feature = "direct-mlx")]
pub use register_ordered_matmul::{
    ProjectionInput, RegisterOrderedWeight, register_ordered_matmul,
    register_ordered_matmul_any_rows, register_ordered_projection,
};
pub use register_ordered_plan::{REGISTER_ORDERED_ROWS_PER_DISPATCH, register_ordered_slice_count};
#[cfg(feature = "direct-mlx")]
pub use row_batched_quantized_matmul::{
    RowBatchedMatmulKernels, RowBatchedProjectionError, row_batched_quantized_matmul,
    row_batched_quantized_matmul_accepts,
};
