//! Live MLX memory policy: the two process-local limits, their once-per-process
//! enforcement, snapshots, and the allocator-cache reclamation cadence.
//!
//! # The two limits
//!
//! MLX exposes two independent process-local memory controls, and Astronomical
//! sets both at worker start:
//!
//! - the **active-memory limit** (`mlx_set_memory_limit`) guides graph
//!   evaluation; the Astronomical allocator patch turns it into a real,
//!   reported ceiling (the source of
//!   [`MlxRuntimeError::ActiveMemoryLimitExceeded`]);
//! - the **allocator-cache limit** (`mlx_set_cache_limit`) bounds how many
//!   freed allocations MLX retains for reuse instead of returning them to the
//!   operating system.
//!
//! # Reclamation cadence — a measured three-way tradeoff
//!
//! When to clear the allocator cache was settled by measurement, and the
//! conclusions are memorialized on the two reclamation methods below: clearing
//! after every prefill chunk stalls WindowServer; never clearing reproduced a
//! 20 GB throughput cliff; clearing above the
//! [`ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES`] floor is the balance the
//! product ships.
//!
//! # The residency-set decision
//!
//! Astronomical deliberately leaves MLX's per-buffer wired residency disabled
//! in this policy. MLX allocation-pressure reclamation removes cached buffers
//! from that residency set while Metal command buffers use unretained
//! references; on affected macOS releases that removal can panic IOGPU with a
//! prepare-count underflow. The worker's wired-residency claim (a single
//! up-front budget in `apps/inference-worker/src/worker_startup.rs`) achieves
//! the same protection without the per-buffer hazard, and an offline contract
//! test asserts this module never re-enables the per-buffer path.

use std::sync::Mutex;

use crate::{
    MlxMemoryLimits, MlxMemorySnapshot, MlxRuntime, MlxRuntimeError,
    allocator_cache_exceeds_reclaim_threshold,
};
use mlx_c_rust::{
    clear_allocator_cache, reset_peak_memory, set_cache_limit, set_memory_limit, synchronize,
};

use super::error_classification;

/// The process-global memory policy. `None` until the first configuration.
static RUNTIME_MEMORY_LIMITS: Mutex<Option<MlxMemoryLimits>> = Mutex::new(None);

impl MlxRuntime {
    /// Replaces the two process-local MLX memory limits without reinitializing the runtime.
    ///
    /// This is the live-ceiling control: the menu raises or lowers the model
    /// RAM ceiling while a worker keeps serving. Ordering and rollback are
    /// the whole design:
    ///
    /// - Native controls are updated before either Rust-side policy record is
    ///   changed, so a failure mid-swap never leaves Rust believing a limit
    ///   the native runtime rejected.
    /// - If the second native control fails, the first is restored to its
    ///   previous value and both the process-global record and this instance
    ///   keep the old policy.
    /// - Re-applying the limits this instance already holds is a no-op, which
    ///   is what lets engine components re-run initialization sequences
    ///   safely.
    ///
    /// # Errors
    ///
    /// Returns [`MlxRuntimeError::RuntimeAlreadyConfigured`] when a
    /// *different* configuration is already active in this process — the
    /// one-worker-one-configuration invariant is a product guarantee, so a
    /// mismatched second configure is a programming error to surface, not a
    /// state to overwrite.
    pub fn update_memory_limits(
        &mut self,
        memory_limits: MlxMemoryLimits,
    ) -> Result<(), MlxRuntimeError> {
        let mut configured_limits = error_classification::lock_unpoisoned(&RUNTIME_MEMORY_LIMITS);
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
    ///
    /// Deliberately queries the native runtime rather than the Rust-side
    /// record: if the two ever disagree, the native value is the truth the
    /// graphics processor operates under, and telemetry built on this call
    /// will expose the disagreement instead of hiding it.
    pub fn configured_memory_limit_bytes(&self) -> Result<usize, MlxRuntimeError> {
        mlx_c_rust::memory_limit_bytes().map_err(MlxRuntimeError::from)
    }

    /// Samples process-local MLX allocator bytes.
    ///
    /// One native round-trip produces a consistent triple (active, cache,
    /// peak) rather than three independent reads that could straddle a
    /// reclamation event.
    pub fn memory_snapshot(&self) -> Result<MlxMemorySnapshot, MlxRuntimeError> {
        Ok(MlxMemorySnapshot {
            active_memory_bytes: mlx_c_rust::active_memory_bytes()
                .map_err(MlxRuntimeError::from)?,
            allocator_cache_memory_bytes: mlx_c_rust::cache_memory_bytes()
                .map_err(MlxRuntimeError::from)?,
            peak_memory_bytes: mlx_c_rust::peak_memory_bytes().map_err(MlxRuntimeError::from)?,
        })
    }

    /// Resets MLX's process-local peak active-memory counter.
    ///
    /// Used to bracket a measurement window (one request, one chunk, one
    /// throughput journey) so "peak" means the peak of that window, not of
    /// the process lifetime.
    pub fn reset_peak_memory(&self) -> Result<(), MlxRuntimeError> {
        reset_peak_memory().map_err(MlxRuntimeError::from)
    }

    /// Releases reclaimable MLX allocator-cache allocations.
    ///
    /// The cache exists to make reallocation cheap; releasing it trades that
    /// speed for memory. Call this only at points where the next allocations
    /// are unpredictable (request finalization) or the cache has grown past
    /// the reclaim threshold.
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
    /// The threshold itself lives in [`crate::mlx_runtime_types`] with the
    /// measurement history behind its value.
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

/// The initialization-time application of the memory policy.
///
/// Enforces the same one-configuration invariant as
/// [`MlxRuntime::update_memory_limits`]: the first call installs, an identical
/// repeat is a no-op, and a conflicting second call is rejected.
pub(super) fn configure_runtime_memory_limits(
    memory_limits: MlxMemoryLimits,
) -> Result<(), MlxRuntimeError> {
    let mut configured_limits = error_classification::lock_unpoisoned(&RUNTIME_MEMORY_LIMITS);
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
/// Neither control changes any kernel setting; both are process-local MLX
/// guidance. Rollback preserves the previous process policy if the
/// allocator-cache control fails.
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
