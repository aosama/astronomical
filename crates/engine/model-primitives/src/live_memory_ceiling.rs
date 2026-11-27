//! Moving the MLX memory ceiling under an engine that keeps nothing between
//! its requests: embedding, decision and speech models.
//!
//! Their loaded model is the only thing held, so the floor is what the
//! allocator reports as active right now, and one rule serves all three.
//! Engine-specific refusals (a request in progress, a shared training state)
//! are checked by the engine before it calls this.

use memory_policy::live_memory_limit::{LiveMemoryLimitRefusal, live_ceiling_refusal};
use memory_policy::telemetry::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
use runtime_integration::MlxRuntime;

/// What a successful live ceiling change leaves the engine to report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppliedLiveCeiling {
    /// The memory the loaded model holds, which is the smallest ceiling it
    /// can run under.
    pub minimum_mlx_memory_ceiling_bytes: u64,
    /// The runtime's MLX buffer cache limit after the change.
    pub mlx_buffer_cache_limit_bytes: u64,
}

impl AppliedLiveCeiling {
    /// The report the worker publishes for this change, with the engine's own
    /// observation taken after it.
    #[must_use]
    pub fn into_adjustment(
        self,
        requested_mlx_memory_ceiling_bytes: u64,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    ) -> MlxMemoryLimitAdjustment {
        MlxMemoryLimitAdjustment::resident(
            requested_mlx_memory_ceiling_bytes,
            self.mlx_buffer_cache_limit_bytes,
            self.minimum_mlx_memory_ceiling_bytes,
            mlx_memory_telemetry,
        )
    }
}

/// Refuses a ceiling below the model's held memory, otherwise applies it.
///
/// # Errors
///
/// `BelowModelFloor` for a ceiling under the held memory, and `Failed` when
/// the platform cannot represent the ceiling or MLX rejects the change.
fn apply_ceiling_above_held_memory(
    runtime: &mut MlxRuntime,
    requested_mlx_memory_ceiling_bytes: u64,
) -> Result<AppliedLiveCeiling, LiveMemoryLimitRefusal> {
    let held_mlx_memory_bytes = runtime
        .memory_snapshot()
        .map_err(failed_with_runtime_error)?
        .active_memory_bytes() as u64;
    if let Some(refusal) =
        live_ceiling_refusal(requested_mlx_memory_ceiling_bytes, held_mlx_memory_bytes)
    {
        return Err(refusal);
    }
    let requested_active_memory_limit_bytes = usize::try_from(requested_mlx_memory_ceiling_bytes)
        .map_err(|_| LiveMemoryLimitRefusal::Failed {
        reason: "the requested MLX memory ceiling exceeds this platform".to_owned(),
    })?;
    runtime
        .apply_live_memory_ceiling(requested_active_memory_limit_bytes)
        .map_err(failed_with_runtime_error)?;
    Ok(AppliedLiveCeiling {
        minimum_mlx_memory_ceiling_bytes: held_mlx_memory_bytes,
        mlx_buffer_cache_limit_bytes: runtime.memory_limits().mlx_buffer_cache_limit_bytes() as u64,
    })
}

fn failed_with_runtime_error(
    runtime_error: runtime_integration::MlxRuntimeError,
) -> LiveMemoryLimitRefusal {
    LiveMemoryLimitRefusal::Failed {
        reason: runtime_error.to_string(),
    }
}

/// The one rule all three engines follow: with a loaded runtime the ceiling
/// moves under it, above what it holds; before the model loads the coming load
/// is simply given the new ceiling, as the image engine does. Either way the
/// engine's own record of its limits follows, so a later load builds from them.
///
/// # Errors
///
/// `BelowModelFloor`, or `Failed` when the platform cannot represent the
/// ceiling or MLX rejects the change.
pub fn move_engine_ceiling(
    loaded_runtime: Option<&mut MlxRuntime>,
    effective_mlx_memory_ceiling_bytes: &mut usize,
    mlx_buffer_cache_limit_bytes: &mut usize,
    requested_mlx_memory_ceiling_bytes: u64,
) -> Result<AppliedLiveCeiling, LiveMemoryLimitRefusal> {
    let applied_ceiling = match loaded_runtime {
        Some(runtime) => {
            apply_ceiling_above_held_memory(runtime, requested_mlx_memory_ceiling_bytes)?
        }
        None => AppliedLiveCeiling {
            // Nothing is loaded yet, so nothing is held.
            minimum_mlx_memory_ceiling_bytes: 1,
            // A cache limit above the ceiling would make the coming load's limits invalid.
            mlx_buffer_cache_limit_bytes: (*mlx_buffer_cache_limit_bytes as u64)
                .min(requested_mlx_memory_ceiling_bytes),
        },
    };
    let requested_ceiling_bytes =
        usize::try_from(requested_mlx_memory_ceiling_bytes).map_err(|_| {
            LiveMemoryLimitRefusal::Failed {
                reason: "the requested MLX memory ceiling exceeds this platform".to_owned(),
            }
        })?;
    *effective_mlx_memory_ceiling_bytes = requested_ceiling_bytes;
    *mlx_buffer_cache_limit_bytes = applied_ceiling.mlx_buffer_cache_limit_bytes as usize;
    Ok(applied_ceiling)
}
