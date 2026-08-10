use std::sync::Mutex;

use crate::{MlxMemoryLimits, MlxMemorySnapshot, MlxRuntime, MlxRuntimeError, raw};

use super::{check_status, error_handling::lock_unpoisoned};

static RUNTIME_MEMORY_LIMITS: Mutex<Option<MlxMemoryLimits>> = Mutex::new(None);

impl MlxRuntime {
    /// Replaces the two process-local MLX memory limits without reinitializing the runtime.
    ///
    /// Native controls are updated before either Rust-side policy is changed. If a later
    /// native control fails, set_memory_limits restores the previous native values and this
    /// method leaves both the runtime instance and process-global policy untouched.
    pub fn update_memory_limits(
        &mut self,
        memory_limits: MlxMemoryLimits,
    ) -> Result<(), MlxRuntimeError> {
        let mut configured_limits = lock_unpoisoned(&RUNTIME_MEMORY_LIMITS);
        if let Some(existing_limits) = *configured_limits
            && existing_limits != self.memory_limits
        {
            return Err(MlxRuntimeError::RuntimeAlreadyConfigured {
                active_memory_limit_bytes: existing_limits.active_memory_limit_bytes,
                allocator_cache_memory_limit_bytes: existing_limits
                    .allocator_cache_memory_limit_bytes,
            });
        }
        if self.memory_limits == memory_limits {
            return Ok(());
        }
        set_memory_limits(memory_limits)?;
        *configured_limits = Some(memory_limits);
        self.memory_limits = memory_limits;
        Ok(())
    }

    /// Reads the active allocator limit currently enforced by MLX.
    pub fn configured_memory_limit_bytes(&self) -> Result<usize, MlxRuntimeError> {
        read_memory_metric(raw::mlx_get_memory_limit, "read the MLX memory limit")
    }

    /// Samples process-local MLX allocator bytes.
    pub fn memory_snapshot(&self) -> Result<MlxMemorySnapshot, MlxRuntimeError> {
        Ok(MlxMemorySnapshot {
            active_memory_bytes: read_memory_metric(
                raw::mlx_get_active_memory,
                "read MLX active memory",
            )?,
            allocator_cache_memory_bytes: read_memory_metric(
                raw::mlx_get_cache_memory,
                "read MLX allocator-cache memory",
            )?,
            peak_memory_bytes: read_memory_metric(
                raw::mlx_get_peak_memory,
                "read MLX peak memory",
            )?,
        })
    }

    /// Resets MLX's process-local peak active-memory counter.
    pub fn reset_peak_memory(&self) -> Result<(), MlxRuntimeError> {
        // SAFETY: The process-global error handler is installed during runtime
        // initialization and `mlx_reset_peak_memory` has no pointer arguments.
        let status = unsafe { raw::mlx_reset_peak_memory() };
        check_status(status, "reset MLX peak memory")
    }

    /// Releases reclaimable MLX allocator-cache allocations.
    pub fn clear_allocator_cache(&self) -> Result<(), MlxRuntimeError> {
        // SAFETY: The process-global error handler is installed during runtime
        // initialization and `mlx_clear_cache` has no pointer arguments.
        let status = unsafe { raw::mlx_clear_cache() };
        check_status(status, "clear the MLX allocator cache")
    }

    /// Waits for submitted work on the runtime GPU stream to complete.
    pub fn synchronize_gpu_stream(&self) -> Result<(), MlxRuntimeError> {
        // SAFETY: The runtime owns this live GPU stream for the worker lifetime.
        let synchronization_status = unsafe { raw::mlx_synchronize(self.gpu_stream.raw()) };
        check_status(synchronization_status, "synchronize the MLX GPU stream")
    }

    /// Waits for the runtime GPU stream before releasing reclaimable allocations.
    ///
    /// Request finalization must use this operation because decode submits one
    /// token ahead asynchronously. Clearing the allocator while that submission
    /// remains in flight can race Metal buffer completion.
    pub fn synchronize_gpu_stream_and_clear_allocator_cache(&self) -> Result<(), MlxRuntimeError> {
        self.synchronize_gpu_stream()?;
        self.clear_allocator_cache()
    }
}

pub(super) fn configure_runtime_memory_limits(
    memory_limits: MlxMemoryLimits,
) -> Result<(), MlxRuntimeError> {
    let mut configured_limits = lock_unpoisoned(&RUNTIME_MEMORY_LIMITS);
    if let Some(existing_limits) = *configured_limits {
        if existing_limits != memory_limits {
            return Err(MlxRuntimeError::RuntimeAlreadyConfigured {
                active_memory_limit_bytes: existing_limits.active_memory_limit_bytes,
                allocator_cache_memory_limit_bytes: existing_limits
                    .allocator_cache_memory_limit_bytes,
            });
        }
    } else {
        set_memory_limits(memory_limits)?;
        *configured_limits = Some(memory_limits);
    }
    Ok(())
}

/// Applies the active-allocation and allocator-retention MLX process controls.
///
/// - `mlx_set_memory_limit` configures MLX graph-evaluation memory guidance.
/// - `mlx_set_cache_limit` controls reclaimable allocator retention.
///
/// Astronomical deliberately leaves MLX's per-buffer wired residency disabled.
/// MLX allocation-pressure reclamation removes cached buffers from that residency
/// set while Metal command buffers use unretained references. On affected macOS
/// releases, that removal can panic IOGPU with a prepare-count underflow.
///
/// Neither control runs `sysctl` or changes `iogpu.wired_limit_mb`. Rollback
/// preserves the previous process policy if the allocator-cache control fails.
fn set_memory_limits(memory_limits: MlxMemoryLimits) -> Result<(), MlxRuntimeError> {
    let mut previous_memory_limit_bytes = 0;
    // SAFETY: The output points to initialized writable storage and the
    // non-terminating handler is installed before this function is reached.
    let memory_status = unsafe {
        raw::mlx_set_memory_limit(
            &mut previous_memory_limit_bytes,
            memory_limits.active_memory_limit_bytes,
        )
    };
    check_status(memory_status, "set the MLX active memory limit")?;

    let mut previous_allocator_cache_limit_bytes = 0;
    // SAFETY: The output points to initialized writable storage and the cache
    // limit is bounded by the active-memory limit validated above.
    let allocator_cache_status = unsafe {
        raw::mlx_set_cache_limit(
            &mut previous_allocator_cache_limit_bytes,
            memory_limits.allocator_cache_memory_limit_bytes,
        )
    };
    if let Err(allocator_cache_error) = check_status(
        allocator_cache_status,
        "set the MLX allocator cache memory limit",
    ) {
        let mut ignored_current_limit_bytes = 0;
        // SAFETY: This best-effort rollback restores the previous valid limit
        // using writable output storage after the handler has been installed.
        unsafe {
            raw::mlx_set_memory_limit(
                &mut ignored_current_limit_bytes,
                previous_memory_limit_bytes,
            );
        }
        return Err(allocator_cache_error);
    }
    Ok(())
}

fn read_memory_metric(
    metric_reader: unsafe extern "C" fn(*mut usize) -> i32,
    operation: &'static str,
) -> Result<usize, MlxRuntimeError> {
    let mut metric_bytes = 0;
    // SAFETY: Every accepted reader is an MLX C memory getter with the same
    // ABI and receives a valid pointer to writable `usize` storage.
    let status = unsafe { metric_reader(&mut metric_bytes) };
    check_status(status, operation)?;
    Ok(metric_bytes)
}
