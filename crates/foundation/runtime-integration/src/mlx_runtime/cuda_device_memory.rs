//! The memory MLX allocates from on a CUDA build: the GPU's own (issue #2699).
//!
//! # Why the limits are capped at the GPU
//!
//! Astronomical sizes MLX's two limits from the machine's memory. That is
//! right where the GPU shares it, on Apple silicon and on the CPU backend. A
//! CUDA GPU has memory of its own, often a fraction of the machine's, and
//! the limits then came out several times larger than the GPU.
//!
//! MLX's CUDA allocator does not refuse an allocation past its limit. Going
//! over only makes it release its MLX buffer cache first; it then asks CUDA
//! anyway, and CUDA fails with out-of-memory. So everything sized from the
//! configured limit (training micro-batches above all) was sized for memory
//! the GPU does not have, and Laya's first training step ran out of it.
//!
//! On a CUDA build the native limits are therefore capped at the share of
//! the GPU's memory MLX itself chooses, 95% (`mlx/backend/cuda/allocator.cpp`),
//! and never above what was free when the GPU was first read: a GPU that
//! also drives a display, or that other processes use, has less than its
//! total to give.
//!
//! # Who sees the cap
//!
//! MLX holds the capped limits, and `MlxRuntime::configured_memory_limit_bytes`,
//! which reads them back from MLX, reports them; that is what training sizes
//! its micro-batches from. `MlxRuntime::memory_limits` keeps returning the
//! limits callers asked for, so the one-configuration rule still compares
//! like with like. Work sized from those requested limits, and the worker's
//! admission ceiling, still follow the machine's memory on CUDA (issue
//! #2706).

use std::sync::OnceLock;

use mlx_bridge::{MlxDeviceHandle, raw};

use crate::{MlxMemoryLimits, MlxRuntimeError};

/// MLX's own default memory limit on CUDA, as a share of the GPU's memory.
const MLX_CUDA_MEMORY_LIMIT_PERCENT_OF_DEVICE: usize = 95;
/// The device-information keys under which MLX's CUDA backend reports the
/// GPU's memory and how much of it is free, in bytes.
const CUDA_TOTAL_MEMORY_DEVICE_INFO_KEY: &str = "total_memory";
const CUDA_FREE_MEMORY_DEVICE_INFO_KEY: &str = "free_memory";

/// `memory_limits` as MLX should hold them on this build: unchanged without
/// the CUDA backend, and capped at the GPU's share otherwise.
///
/// # Errors
///
/// Fails when MLX cannot report the CUDA GPU's memory.
pub(crate) fn native_memory_limits(
    memory_limits: MlxMemoryLimits,
) -> Result<MlxMemoryLimits, MlxRuntimeError> {
    Ok(memory_limits_within_device(
        memory_limits,
        cuda_device_memory_ceiling_bytes()?,
    ))
}

/// The limits capped at `device_memory_ceiling_bytes`, when there is one.
/// Capping both keeps the MLX buffer cache limit at or below the active
/// limit, as `MlxMemoryLimits::new` requires.
pub(crate) fn memory_limits_within_device(
    memory_limits: MlxMemoryLimits,
    device_memory_ceiling_bytes: Option<usize>,
) -> MlxMemoryLimits {
    let Some(device_memory_ceiling_bytes) = device_memory_ceiling_bytes else {
        return memory_limits;
    };
    MlxMemoryLimits {
        active_memory_limit_bytes: memory_limits
            .active_memory_limit_bytes
            .min(device_memory_ceiling_bytes),
        mlx_buffer_cache_limit_bytes: memory_limits
            .mlx_buffer_cache_limit_bytes
            .min(device_memory_ceiling_bytes),
    }
}

/// The ceiling for a GPU MLX reports `total_memory_bytes` and
/// `free_memory_bytes` for: MLX's own 95% of the total, and no more than is
/// free. Refuses a GPU reported with no memory, which no limit could fit.
pub(crate) fn device_memory_ceiling(
    total_memory_bytes: usize,
    free_memory_bytes: usize,
) -> Result<usize, String> {
    if total_memory_bytes == 0 {
        return Err("MLX reported a CUDA GPU with no memory".to_owned());
    }
    // In u128, so the share is exact for any total.
    let mlx_share_bytes = (total_memory_bytes as u128
        * MLX_CUDA_MEMORY_LIMIT_PERCENT_OF_DEVICE as u128
        / 100) as usize;
    Ok(mlx_share_bytes.min(free_memory_bytes))
}

/// The CUDA GPU's ceiling for MLX, read once per process: the first read
/// happens before any model is loaded, when free memory is what this
/// process can have. `None` without the CUDA backend. A failed read is not
/// kept, so a later runtime tries again rather than inheriting the failure.
fn cuda_device_memory_ceiling_bytes() -> Result<Option<usize>, MlxRuntimeError> {
    static DEVICE_MEMORY_CEILING: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(device_memory_ceiling) = DEVICE_MEMORY_CEILING.get() {
        return Ok(*device_memory_ceiling);
    }
    match read_cuda_device_memory_ceiling_bytes() {
        Ok(device_memory_ceiling) => {
            // Two runtimes starting together may both read; the first
            // answer is kept, and both read the same device anyway.
            let _ = DEVICE_MEMORY_CEILING.set(device_memory_ceiling);
            Ok(*DEVICE_MEMORY_CEILING
                .get()
                .unwrap_or(&device_memory_ceiling))
        }
        Err(description) => Err(MlxRuntimeError::RuntimeOperation {
            operation: "read the CUDA GPU's memory",
            description,
        }),
    }
}

fn read_cuda_device_memory_ceiling_bytes() -> Result<Option<usize>, String> {
    if !mlx_bridge::cuda_is_available().map_err(describe_mlx_error)? {
        return Ok(None);
    }
    // MLX's default GPU, the device every runtime stream is created on.
    let gpu_device =
        MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_GPU, 0).map_err(describe_mlx_error)?;
    let device_information = gpu_device.information().map_err(describe_mlx_error)?;
    let total_memory_bytes = device_information
        .size_value(CUDA_TOTAL_MEMORY_DEVICE_INFO_KEY)
        .map_err(describe_mlx_error)?;
    let free_memory_bytes = device_information
        .size_value(CUDA_FREE_MEMORY_DEVICE_INFO_KEY)
        .map_err(describe_mlx_error)?;
    device_memory_ceiling(total_memory_bytes, free_memory_bytes).map(Some)
}

fn describe_mlx_error(mlx_error: mlx_bridge::MlxCError) -> String {
    mlx_error.to_string()
}

#[cfg(test)]
#[path = "cuda_device_memory_tests.rs"]
mod tests;
