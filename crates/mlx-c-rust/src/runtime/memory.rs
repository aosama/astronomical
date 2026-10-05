//! Safe wrappers over MLX's process-global memory and Metal controls.
//!
//! The policy owner in `astronomical-runtime-integration` decides when to call
//! these; this module only makes the calls safe and typed. Every function here
//! mirrors exactly one official MLX-C control.

use crate::{MlxCError, MlxStream, error::check_status, raw};

/// Applies the graph-evaluation active-memory guidance and returns the
/// previous limit in bytes.
pub fn set_memory_limit(active_memory_limit_bytes: usize) -> Result<usize, MlxCError> {
    let mut previous_limit_bytes = 0;
    // SAFETY: The output points to initialized writable storage and the
    // non-terminating handler is installed before policy runs.
    let status =
        unsafe { raw::mlx_set_memory_limit(&mut previous_limit_bytes, active_memory_limit_bytes) };
    check_status(status, "set the MLX active memory limit")?;
    Ok(previous_limit_bytes)
}

/// Applies the allocator-cache retention limit and returns the previous limit
/// in bytes.
pub fn set_cache_limit(allocator_cache_limit_bytes: usize) -> Result<usize, MlxCError> {
    let mut previous_limit_bytes = 0;
    // SAFETY: The output points to initialized writable storage and the cache
    // limit is bounded by the active-memory limit the caller validated.
    let status =
        unsafe { raw::mlx_set_cache_limit(&mut previous_limit_bytes, allocator_cache_limit_bytes) };
    check_status(status, "set the MLX allocator cache memory limit")?;
    Ok(previous_limit_bytes)
}

/// Reads the allocator limit currently enforced by MLX.
pub fn memory_limit_bytes() -> Result<usize, MlxCError> {
    read_metric("read the MLX memory limit", raw::mlx_get_memory_limit)
}

/// Samples process-local MLX active-memory bytes.
pub fn active_memory_bytes() -> Result<usize, MlxCError> {
    read_metric("read MLX active memory", raw::mlx_get_active_memory)
}

/// Samples process-local MLX allocator-cache bytes.
pub fn cache_memory_bytes() -> Result<usize, MlxCError> {
    read_metric("read MLX allocator-cache memory", raw::mlx_get_cache_memory)
}

/// Samples process-local MLX peak active-memory bytes.
pub fn peak_memory_bytes() -> Result<usize, MlxCError> {
    read_metric("read MLX peak memory", raw::mlx_get_peak_memory)
}

/// Resets MLX's process-local peak active-memory counter.
pub fn reset_peak_memory() -> Result<(), MlxCError> {
    // SAFETY: The process-global error handler is installed during runtime
    // initialization and `mlx_reset_peak_memory` has no pointer arguments.
    let status = unsafe { raw::mlx_reset_peak_memory() };
    check_status(status, "reset MLX peak memory")
}

/// Releases reclaimable MLX allocator-cache allocations.
pub fn clear_allocator_cache() -> Result<(), MlxCError> {
    // SAFETY: The process-global error handler is installed during runtime
    // initialization and `mlx_clear_cache` has no pointer arguments.
    let status = unsafe { raw::mlx_clear_cache() };
    check_status(status, "clear the MLX allocator cache")
}

/// Waits for submitted work on the given stream to complete.
pub fn synchronize(stream: &MlxStream) -> Result<(), MlxCError> {
    // SAFETY: The caller's stream owner keeps the stream live for the call.
    let status = unsafe { raw::mlx_synchronize(stream.raw()) };
    check_status(status, "synchronize the MLX GPU stream")
}

fn read_metric(
    operation: &'static str,
    metric_reader: unsafe extern "C" fn(*mut usize) -> i32,
) -> Result<usize, MlxCError> {
    let mut metric_bytes = 0;
    // SAFETY: Every accepted reader is an MLX C memory getter with the same
    // ABI and receives a valid pointer to writable `usize` storage.
    let status = unsafe { metric_reader(&mut metric_bytes) };
    check_status(status, operation)?;
    Ok(metric_bytes)
}
