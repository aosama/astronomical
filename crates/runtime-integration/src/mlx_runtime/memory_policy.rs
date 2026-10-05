use std::sync::Mutex;

use crate::{
    MlxMemoryLimits, MlxMemorySnapshot, MlxRuntime, MlxRuntimeError,
    allocator_cache_exceeds_reclaim_threshold,
};
use astronomical_mlx_c_rust::{
    clear_allocator_cache, reset_peak_memory, set_cache_limit, set_memory_limit, synchronize,
};

use super::error_handling::lock_unpoisoned;

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
        astronomical_mlx_c_rust::memory_limit_bytes().map_err(MlxRuntimeError::from)
    }

    /// Samples process-local MLX allocator bytes.
    pub fn memory_snapshot(&self) -> Result<MlxMemorySnapshot, MlxRuntimeError> {
        Ok(MlxMemorySnapshot {
            active_memory_bytes: astronomical_mlx_c_rust::active_memory_bytes()
                .map_err(MlxRuntimeError::from)?,
            allocator_cache_memory_bytes: astronomical_mlx_c_rust::cache_memory_bytes()
                .map_err(MlxRuntimeError::from)?,
            peak_memory_bytes: astronomical_mlx_c_rust::peak_memory_bytes()
                .map_err(MlxRuntimeError::from)?,
        })
    }

    /// Resets MLX's process-local peak active-memory counter.
    pub fn reset_peak_memory(&self) -> Result<(), MlxRuntimeError> {
        reset_peak_memory().map_err(MlxRuntimeError::from)
    }

    /// Releases reclaimable MLX allocator-cache allocations.
    pub fn clear_allocator_cache(&self) -> Result<(), MlxRuntimeError> {
        clear_allocator_cache().map_err(MlxRuntimeError::from)
    }

    /// Waits for submitted work on the runtime GPU stream to complete.
    pub fn synchronize_gpu_stream(&self) -> Result<(), MlxRuntimeError> {
        synchronize(self.context.gpu_stream()).map_err(MlxRuntimeError::from)
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

    /// Clears the allocator cache only when it is large enough to justify IOGPU work.
    ///
    /// Prefill already retires the chunk tape through `evaluate_arrays`. A stream
    /// drain after every chunk was a second full GPU wait and showed up as lost
    /// prompt tokens per second; the chunk tape is evaluated once and moves on.
    pub fn synchronize_gpu_stream_and_reclaim_allocator_cache_above_threshold(
        &self,
        reclaim_threshold_bytes: usize,
    ) -> Result<(), MlxRuntimeError> {
        let memory_snapshot = self.memory_snapshot()?;
        if !allocator_cache_exceeds_reclaim_threshold(
            memory_snapshot.allocator_cache_memory_bytes(),
            reclaim_threshold_bytes,
        ) {
            return Ok(());
        }
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
    let previous_memory_limit_bytes =
        set_memory_limit(memory_limits.active_memory_limit_bytes).map_err(MlxRuntimeError::from)?;

    if let Err(allocator_cache_error) =
        set_cache_limit(memory_limits.allocator_cache_memory_limit_bytes)
            .map_err(MlxRuntimeError::from)
    {
        // Best-effort rollback restores the previous valid limit after the
        // handler has been installed; nothing depends on the restore result.
        let _ = set_memory_limit(previous_memory_limit_bytes);
        return Err(allocator_cache_error);
    }
    Ok(())
}
