//! Routes projections over a few token rows through the row-batched kernel.
//!
//! Drafting verifies 8 rows at once, where MLX's quantized matmul runs its
//! one-row kernel once per row (134 ms over the 27B's 64 layers at 8 rows);
//! the row-batched kernel reads each weight group once for every row (96 ms).
//! A model routes nothing here until `enable_row_batched_projections` has
//! proven the kernel on this GPU, so every other model, prompt chunk and
//! one-row decode step keeps MLX's own kernels.

use astronomical_runtime_integration::MlxRuntimeError;
use mlx_bridge::MlxArray;

use super::Qwen3_5ModelBase;
use crate::kernel_capability::{CustomMetalKernelFamily, worker_process_kernel_capabilities};
use crate::performance_attribution::PerformanceAttribution;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::row_batched_matmul::{
    RowBatchedMatmulKernels, row_batched_quantized_matmul, row_batched_quantized_matmul_accepts,
};

/// The fewest rows the kernel takes. Measured over the 27B's projections on
/// an M5 Pro: MLX is faster up to 5 rows (83 against 94 ms over 64 layers at
/// 5), the row-batched kernel from 6 (94 against 109).
pub const ROW_BATCHED_PROJECTION_MINIMUM_ROWS: i32 = 6;

impl Qwen3_5ModelBase {
    /// Lets projections over `ROW_BATCHED_PROJECTION_MINIMUM_ROWS` to 32 rows
    /// use the row-batched kernel, once this worker's capability probe has
    /// run it on this GPU. Returns whether it is now enabled; a GPU without M5
    /// matrix units keeps MLX's kernels and returns false.
    pub fn enable_row_batched_projections(
        &mut self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<bool, MlxRuntimeError> {
        let capabilities =
            worker_process_kernel_capabilities(&self.runtime, performance_attribution);
        if !capabilities.is_custom_kernel_supported(CustomMetalKernelFamily::ROW_BATCHED_MATMUL) {
            return Ok(false);
        }
        self.optional_accelerators.row_batched_matmul_kernels =
            Some(RowBatchedMatmulKernels::declare()?);
        Ok(true)
    }

    /// The row-batched projection of `activations` when it is enabled and
    /// takes this call; None sends the caller to MLX's quantized matmul.
    pub(crate) fn row_batched_projection(
        &self,
        activations: &MlxArray,
        packed_weight: &MlxArray,
        scales: &MlxArray,
        biases: &MlxArray,
    ) -> Option<Result<MlxArray, Qwen3_5ExecutionError>> {
        let kernels = self
            .optional_accelerators
            .row_batched_matmul_kernels
            .as_ref()?;
        let activation_shape = activations.shape();
        let row_count: i32 = activation_shape[..activation_shape.len().saturating_sub(1)]
            .iter()
            .product();
        if row_count < ROW_BATCHED_PROJECTION_MINIMUM_ROWS
            || !row_batched_quantized_matmul_accepts(activations, packed_weight, scales, biases)
        {
            return None;
        }
        Some(
            row_batched_quantized_matmul(
                &self.runtime,
                kernels,
                activations,
                packed_weight,
                scales,
                biases,
            )
            .map_err(|projection_error| {
                Qwen3_5ExecutionError::from(MlxRuntimeError::RuntimeOperation {
                    operation: "project rows through the row-batched matmul",
                    description: projection_error.to_string(),
                })
            }),
        )
    }
}
