//! Keeping the `mlx_buffer_cache` from growing through a training.
//!
//! Every pass leaves freed GPU buffers in the `mlx_buffer_cache`, and passes
//! of different shapes leave buffers nothing reuses. Serving empties the
//! `mlx_buffer_cache` after each request; a training runs thousands of
//! passes inside one request, and left alone the `mlx_buffer_cache` grew
//! until the Mac swapped (found live on Laya: steps slowed from 6 to 40
//! seconds and the desktop froze). Every family's whole-model training
//! therefore reclaims it after each pass once it holds more than the
//! runtime's reclaim threshold.

use astronomical_runtime_integration::MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES;

use crate::decision_models::training::gpu_math::{GpuMath, GpuResult, GpuTrainingError};

/// Returns the `mlx_buffer_cache` to the system once it is worth the GPU wait.
pub(crate) fn reclaim_mlx_buffer_cache(math: GpuMath<'_>) -> GpuResult<()> {
    match math
        .runtime
        .synchronize_gpu_stream_and_reclaim_mlx_buffer_cache_above_threshold(
            MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES,
        ) {
        Ok(()) => Ok(()),
        Err(reclaim_error) => Err(GpuTrainingError(reclaim_error.to_string())),
    }
}
