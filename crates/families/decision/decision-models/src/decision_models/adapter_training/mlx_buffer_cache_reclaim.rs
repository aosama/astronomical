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

use std::sync::atomic::{AtomicUsize, Ordering};

use runtime_integration::MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES;

use crate::decision_models::training::gpu_math::{GpuMath, GpuResult, GpuTrainingError};

/// How many passes in this process found the `mlx_buffer_cache` over the
/// threshold and cleared it. A journey that exists to exercise a clear
/// between passes (the CUDA graph executables kept across it crashed, issue
/// #2699) reads it to prove a clear happened at all.
static RECLAIMED_PASS_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Returns the `mlx_buffer_cache` to the system once it is worth the GPU wait.
pub fn reclaim_mlx_buffer_cache(math: GpuMath<'_>) -> GpuResult<()> {
    match math
        .runtime
        .reclaim_mlx_buffer_cache_if_above_threshold(MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES)
    {
        Ok(was_reclaimed) => {
            // A true adds one, a false adds nothing.
            RECLAIMED_PASS_COUNT.fetch_add(usize::from(was_reclaimed), Ordering::Relaxed);
            Ok(())
        }
        Err(reclaim_error) => Err(GpuTrainingError(reclaim_error.to_string())),
    }
}

/// How many passes so far cleared the `mlx_buffer_cache`, process-wide.
#[must_use]
pub fn reclaimed_pass_count() -> usize {
    RECLAIMED_PASS_COUNT.load(Ordering::Relaxed)
}

#[cfg(test)]
#[path = "mlx_buffer_cache_reclaim_tests.rs"]
mod tests;
