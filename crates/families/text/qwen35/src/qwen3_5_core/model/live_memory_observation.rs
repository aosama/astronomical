//! The runtime readings and limit arithmetic the Qwen3.5 engine's live MLX
//! ceiling changes use (issue #2473).

use runtime_integration::{
    MlxMemoryLimits, MlxRuntime, mlx_buffer_cache_limit_after_active_limit_change,
};

use crate::InferenceEngineError;
use text_generation::engine_errors::{fatal_engine_error, fatal_runtime_error};

/// The live runtime facts one ceiling-change decision reads: limits currently
/// installed, the ceiling they imply, the model's geometry-safe minimum, and
/// currently active memory. Snapshotting them together keeps the decision
/// operating on one coherent instant of the runtime.
pub struct LiveMemoryObservation {
    pub current_mlx_memory_limits: MlxMemoryLimits,
    pub current_mlx_memory_ceiling_bytes: u64,
    pub minimum_mlx_memory_ceiling_bytes: u64,
    pub current_active_memory_bytes: u64,
}

/// Reads everything the ceiling-change decision needs before any limit moves.
pub fn observe_live_memory(
    runtime: &MlxRuntime,
    minimum_mlx_memory_ceiling_bytes: u64,
) -> Result<LiveMemoryObservation, InferenceEngineError> {
    let current_mlx_memory_limits = runtime.memory_limits();
    let current_mlx_memory_ceiling_bytes =
        u64::try_from(current_mlx_memory_limits.active_memory_limit_bytes())
            .map_err(|_| fatal_engine_error("current MLX memory ceiling exceeds the u64 range"))?;
    let current_active_memory_bytes = u64::try_from(
        runtime
            .memory_snapshot()
            .map_err(fatal_runtime_error)?
            .active_memory_bytes(),
    )
    .map_err(|_| fatal_engine_error("current MLX active memory exceeds the u64 range"))?;
    Ok(LiveMemoryObservation {
        current_mlx_memory_limits,
        current_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
        current_active_memory_bytes,
    })
}

/// The limits a requested ceiling installs, or `None` when it equals the
/// current ceiling and nothing should move. The platform-range check runs
/// first, so an unrepresentable request fails even when it matches.
pub fn requested_mlx_memory_limits_when_changed(
    requested_mlx_memory_ceiling_bytes: u64,
    current_mlx_memory_limits: MlxMemoryLimits,
) -> Result<Option<MlxMemoryLimits>, InferenceEngineError> {
    let current_mlx_memory_ceiling_bytes =
        current_mlx_memory_limits.active_memory_limit_bytes() as u64;
    let requested_mlx_memory_ceiling_bytes_as_usize =
        usize::try_from(requested_mlx_memory_ceiling_bytes).map_err(|_| {
            fatal_engine_error("requested MLX memory ceiling exceeds the platform range")
        })?;
    if requested_mlx_memory_ceiling_bytes == current_mlx_memory_ceiling_bytes {
        return Ok(None);
    }
    MlxMemoryLimits::new(
        requested_mlx_memory_ceiling_bytes_as_usize,
        mlx_buffer_cache_limit_after_active_limit_change(
            current_mlx_memory_limits,
            requested_mlx_memory_ceiling_bytes_as_usize,
        ),
    )
    .map(Some)
    .map_err(fatal_runtime_error)
}

/// Releases the MLX buffer cache under the old ceiling and reads the active
/// memory left, which a lowering compares against the requested ceiling.
pub fn post_reclamation_active_memory_bytes(
    runtime: &MlxRuntime,
) -> Result<u64, InferenceEngineError> {
    let post_reclamation_mlx_memory_snapshot = runtime
        .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
        .and_then(|()| runtime.memory_snapshot())
        .map_err(fatal_runtime_error)?;
    u64::try_from(post_reclamation_mlx_memory_snapshot.active_memory_bytes())
        .map_err(|_| fatal_engine_error("post-reclamation MLX active memory exceeds the u64 range"))
}
