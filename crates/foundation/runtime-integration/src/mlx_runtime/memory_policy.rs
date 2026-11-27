//! Live MLX memory policy: the two process-local limits, their once-per-process
//! enforcement, snapshots, and the MLX buffer cache reclamation cadence.
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
//! - the **MLX buffer cache limit** (`mlx_set_cache_limit`) bounds how many
//!   freed allocations MLX retains for reuse instead of returning them to the
//!   operating system.
//!
//! # Reclamation cadence — a measured three-way tradeoff
//!
//! When to clear the MLX buffer cache was settled by measurement, and the
//! conclusions are memorialized on the two reclamation methods below: clearing
//! after every prefill chunk stalls WindowServer; never clearing reproduced a
//! 20 GB throughput cliff; clearing above the
//! [`MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES`] floor is the balance the
//! product ships.
//!
//! # The residency-set decision
//!
//! Astronomical deliberately leaves MLX's per-buffer wired residency disabled
//! in this policy. MLX allocation-pressure reclamation removes `mlx_buffer_cache` buffers
//! from that residency set while Metal command buffers use unretained
//! references; on affected macOS releases that removal can panic IOGPU with a
//! prepare-count underflow. The worker's wired-residency claim (a single
//! up-front budget in `apps/inference-worker/src/worker_startup.rs`) achieves
//! the same protection without the per-buffer hazard, and a unit contract
//! test asserts this module never re-enables the per-buffer path.

use std::sync::Mutex;

use crate::{
    MlxMemoryLimits, MlxMemorySnapshot, MlxRuntime, MlxRuntimeError,
    mlx_buffer_cache_exceeds_reclaim_threshold,
};
// The bridge keeps MLX's own function names, where "cache" alone means the
// MLX buffer cache; they are imported under names that say so.
use mlx_bridge::{
    cache_memory_bytes as mlx_buffer_cache_bytes, clear_allocator_cache as clear_mlx_buffer_cache,
    reset_peak_memory, set_cache_limit as set_mlx_buffer_cache_limit, set_memory_limit,
    synchronize,
};

use super::error_classification;

/// The process-global memory policy and how many runtimes currently hold it.
///
/// MLX limits are process-global, so two live runtimes must agree on them; a
/// runtime that would install different limits beside a live one is refused.
/// Once every runtime has been dropped, nothing depends on the installed
/// limits any more, so the next runtime installs its own. Without that, a
/// worker that loaded one model, changed its ceiling with no model loaded,
/// and then loaded another was refused until restart (#1906).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ProcessMemoryPolicy {
    pub(super) installed_limits: Option<MlxMemoryLimits>,
    pub(super) live_runtime_count: usize,
}

impl ProcessMemoryPolicy {
    pub(super) const UNCONFIGURED: Self = Self {
        installed_limits: None,
        live_runtime_count: 0,
    };
}

static PROCESS_MEMORY_POLICY: Mutex<ProcessMemoryPolicy> =
    Mutex::new(ProcessMemoryPolicy::UNCONFIGURED);

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
        let mut process_policy = error_classification::lock_unpoisoned(&PROCESS_MEMORY_POLICY);
        if let Some(existing_limits) = process_policy.installed_limits
            && existing_limits != self.memory_limits
        {
            return Err(MlxRuntimeError::RuntimeAlreadyConfigured {
                active_memory_limit_bytes: existing_limits.active_memory_limit_bytes,
                mlx_buffer_cache_limit_bytes: existing_limits.mlx_buffer_cache_limit_bytes,
            });
        }
        if self.memory_limits == memory_limits {
            return Ok(());
        }
        set_memory_limits(memory_limits)?;
        process_policy.installed_limits = Some(memory_limits);
        self.memory_limits = memory_limits;
        Ok(())
    }

    /// Moves the active-memory ceiling and keeps this runtime's MLX buffer
    /// cache limit, lowered to the new ceiling when it would otherwise exceed
    /// it, because `MlxMemoryLimits::new` refuses a cache larger than the
    /// ceiling.
    ///
    /// # Errors
    ///
    /// Returns the validation or runtime error of the underlying update.
    pub fn apply_live_memory_ceiling(
        &mut self,
        requested_active_memory_limit_bytes: usize,
    ) -> Result<(), MlxRuntimeError> {
        let kept_mlx_buffer_cache_limit_bytes = self
            .memory_limits
            .mlx_buffer_cache_limit_bytes()
            .min(requested_active_memory_limit_bytes);
        let memory_limits = MlxMemoryLimits::new(
            requested_active_memory_limit_bytes,
            kept_mlx_buffer_cache_limit_bytes,
        )?;
        self.update_memory_limits(memory_limits)
    }

    /// Reads the active allocator limit currently enforced by MLX.
    ///
    /// Deliberately queries the native runtime rather than the Rust-side
    /// record: if the two ever disagree, the native value is the truth the
    /// graphics processor operates under, and telemetry built on this call
    /// will expose the disagreement instead of hiding it.
    pub fn configured_memory_limit_bytes(&self) -> Result<usize, MlxRuntimeError> {
        mlx_bridge::memory_limit_bytes().map_err(MlxRuntimeError::from)
    }

    /// Samples process-local MLX allocator bytes.
    ///
    /// One native round-trip produces a consistent triple (active, MLX buffer cache,
    /// peak) rather than three independent reads that could straddle a
    /// reclamation event.
    pub fn memory_snapshot(&self) -> Result<MlxMemorySnapshot, MlxRuntimeError> {
        Ok(MlxMemorySnapshot {
            active_memory_bytes: mlx_bridge::active_memory_bytes()
                .map_err(MlxRuntimeError::from)?,
            mlx_buffer_cache_bytes: mlx_buffer_cache_bytes().map_err(MlxRuntimeError::from)?,
            peak_memory_bytes: mlx_bridge::peak_memory_bytes().map_err(MlxRuntimeError::from)?,
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

    /// Releases reclaimable MLX buffer cache allocations.
    ///
    /// The MLX buffer cache exists to make reallocation cheap; releasing it trades that
    /// speed for memory. Call this only at points where the next allocations
    /// are unpredictable (request finalization) or the MLX buffer cache has grown past
    /// the reclaim threshold.
    pub fn clear_mlx_buffer_cache(&self) -> Result<(), MlxRuntimeError> {
        clear_mlx_buffer_cache().map_err(MlxRuntimeError::from)
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
    pub fn synchronize_gpu_stream_and_clear_mlx_buffer_cache(&self) -> Result<(), MlxRuntimeError> {
        self.synchronize_gpu_stream()?;
        self.clear_mlx_buffer_cache()
    }

    /// Clears the MLX buffer cache only when it is large enough to justify IOGPU work.
    ///
    /// Prefill already retires the chunk tape through `evaluate_arrays`. A stream
    /// drain after every chunk was a second full GPU wait and showed up as lost
    /// prompt tokens per second; the chunk tape is evaluated once and moves on.
    /// The threshold itself lives in [`crate::mlx_runtime_types`] with the
    /// measurement history behind its value.
    pub fn synchronize_gpu_stream_and_reclaim_mlx_buffer_cache_above_threshold(
        &self,
        reclaim_threshold_bytes: usize,
    ) -> Result<(), MlxRuntimeError> {
        let memory_snapshot = self.memory_snapshot()?;
        if !mlx_buffer_cache_exceeds_reclaim_threshold(
            memory_snapshot.mlx_buffer_cache_bytes(),
            reclaim_threshold_bytes,
        ) {
            return Ok(());
        }
        self.synchronize_gpu_stream()?;
        self.clear_mlx_buffer_cache()
    }
}

/// The initialization-time application of the memory policy, which also
/// counts the new runtime as live.
///
/// Beside a live runtime the same one-configuration rule as
/// [`MlxRuntime::update_memory_limits`] holds: identical limits are shared, and
/// different ones are refused. With no runtime alive the new limits replace
/// whatever the last runtime left behind.
pub(super) fn configure_runtime_memory_limits(
    memory_limits: MlxMemoryLimits,
) -> Result<(), MlxRuntimeError> {
    admit_runtime_memory_limits(&PROCESS_MEMORY_POLICY, memory_limits, set_memory_limits)
}

/// Releases one live runtime's hold on the process policy. Paired with every
/// successful [`configure_runtime_memory_limits`], including when the
/// runtime's construction fails after the limits were admitted.
pub(super) fn release_runtime_memory_limits() {
    release_runtime_from(&PROCESS_MEMORY_POLICY);
}

/// The admission rule behind [`configure_runtime_memory_limits`], with the
/// policy and the native setter passed in so tests never touch the process
/// state other tests share. The check, the install and the count change happen
/// under one lock, so two runtimes starting at once cannot both replace it.
pub(super) fn admit_runtime_memory_limits(
    process_policy: &Mutex<ProcessMemoryPolicy>,
    memory_limits: MlxMemoryLimits,
    apply_native_limits: fn(MlxMemoryLimits) -> Result<(), MlxRuntimeError>,
) -> Result<(), MlxRuntimeError> {
    let mut process_policy = error_classification::lock_unpoisoned(process_policy);
    match process_policy.installed_limits {
        Some(existing_limits) if existing_limits == memory_limits => {}
        Some(existing_limits) if process_policy.live_runtime_count > 0 => {
            return Err(MlxRuntimeError::RuntimeAlreadyConfigured {
                active_memory_limit_bytes: existing_limits.active_memory_limit_bytes,
                mlx_buffer_cache_limit_bytes: existing_limits.mlx_buffer_cache_limit_bytes,
            });
        }
        _ => {
            apply_native_limits(memory_limits)?;
            process_policy.installed_limits = Some(memory_limits);
        }
    }
    process_policy.live_runtime_count += 1;
    Ok(())
}

/// The release rule behind [`release_runtime_memory_limits`]; saturates so a
/// stray release can never wrap the count and wedge every later runtime.
pub(super) fn release_runtime_from(process_policy: &Mutex<ProcessMemoryPolicy>) {
    let mut process_policy = error_classification::lock_unpoisoned(process_policy);
    process_policy.live_runtime_count = process_policy.live_runtime_count.saturating_sub(1);
}

/// Applies the active-allocation and allocator-retention MLX process controls.
///
/// - `mlx_set_memory_limit` configures MLX graph-evaluation memory guidance.
/// - `mlx_set_cache_limit` controls reclaimable allocator retention.
///
/// Astronomical deliberately leaves MLX's per-buffer wired residency disabled.
/// MLX allocation-pressure reclamation removes `mlx_buffer_cache` buffers from that residency
/// set while Metal command buffers use unretained references. On affected macOS
/// releases, that removal can panic IOGPU with a prepare-count underflow.
///
/// Neither control changes any kernel setting; both are process-local MLX
/// guidance. Rollback preserves the previous process policy if the
/// MLX buffer cache control fails.
fn set_memory_limits(memory_limits: MlxMemoryLimits) -> Result<(), MlxRuntimeError> {
    let previous_memory_limit_bytes =
        set_memory_limit(memory_limits.active_memory_limit_bytes).map_err(MlxRuntimeError::from)?;

    if let Err(mlx_buffer_cache_error) =
        set_mlx_buffer_cache_limit(memory_limits.mlx_buffer_cache_limit_bytes)
            .map_err(MlxRuntimeError::from)
    {
        // Best-effort rollback restores the previous valid limit after the
        // handler has been installed; nothing depends on the restore result.
        let _ = set_memory_limit(previous_memory_limit_bytes);
        return Err(mlx_buffer_cache_error);
    }
    Ok(())
}
