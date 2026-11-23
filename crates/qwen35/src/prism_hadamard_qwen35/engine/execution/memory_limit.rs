//! The rules for a live MLX memory-limit change on the folded-ternary engine.
//!
//! They match the resident engine's: a change waits until no request is
//! running, a ceiling below what the loaded weights need is refused with the
//! typed rejection, and the `mlx_buffer_cache` limit follows the ceiling down
//! and back up to the configured limit, so a lowered and then raised ceiling
//! does not leave the `mlx_buffer_cache` smaller than configured.

use crate::InferenceEngineError;

/// Why a ceiling change cannot be applied now, if it cannot.
pub(super) fn ceiling_change_refusal(
    has_active_request: bool,
    requested_mlx_memory_ceiling_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
) -> Option<InferenceEngineError> {
    if has_active_request {
        return Some(InferenceEngineError::EngineBusy);
    }
    if requested_mlx_memory_ceiling_bytes < minimum_mlx_memory_ceiling_bytes {
        return Some(InferenceEngineError::MlxMemoryLimitRejected {
            requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            reason: "the requested ceiling is below what the folded-ternary weights need"
                .to_owned(),
        });
    }
    None
}

/// The `mlx_buffer_cache` limit under a ceiling: the configured limit, never
/// more than the ceiling itself.
pub(super) fn mlx_buffer_cache_bytes_for_ceiling(
    configured_mlx_buffer_cache_bytes: u64,
    mlx_memory_ceiling_bytes: u64,
) -> u64 {
    configured_mlx_buffer_cache_bytes.min(mlx_memory_ceiling_bytes)
}

#[cfg(test)]
#[path = "memory_limit_tests.rs"]
mod memory_limit_tests;
