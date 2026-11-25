//! The folded-ternary engine's own step in a live MLX memory-ceiling change;
//! the shared Qwen 3.5 change runs every other step (it waits until no
//! request is running and needs a loaded model).
//!
//! The dense weights stay whole, so a ceiling below what they need is refused
//! with the typed rejection, and the `mlx_buffer_cache` limit follows the
//! ceiling down and back up to the configured limit, so a lowered and then
//! raised ceiling does not leave the `mlx_buffer_cache` smaller than
//! configured.

use runtime_integration::MlxMemoryLimits;

use super::PrismHadamardQwen35InferenceExecution;
use crate::InferenceEngineError;
use text_generation::engine_errors::mlx_memory_ceiling_change_before_load_error;
use text_generation::generation_loop::mlx_memory_limit_change::TextMlxMemoryLimitHooks;
use text_generation::model::text_model::TextModel;

impl TextMlxMemoryLimitHooks for PrismHadamardQwen35InferenceExecution {
    fn change_model_mlx_memory_ceiling(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(u64, MlxMemoryLimits), InferenceEngineError> {
        let configured_mlx_buffer_cache_bytes = self.configured_mlx_buffer_cache_bytes;
        let model = self
            .execution_core
            .model
            .as_mut()
            .ok_or_else(mlx_memory_ceiling_change_before_load_error)?;
        let minimum_mlx_memory_ceiling_bytes = model.minimum_mlx_memory_ceiling_bytes()?;
        if let Some(refusal) = ceiling_change_refusal(
            requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
        ) {
            return Err(refusal);
        }
        let updated_mlx_memory_limits = memory_limits_for_ceiling(
            requested_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_bytes_for_ceiling(
                configured_mlx_buffer_cache_bytes,
                requested_mlx_memory_ceiling_bytes,
            ),
        )?;
        model
            .runtime
            .update_memory_limits(updated_mlx_memory_limits)
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary could not apply the MLX memory ceiling: {error}"),
            })?;
        Ok((minimum_mlx_memory_ceiling_bytes, updated_mlx_memory_limits))
    }
}

/// Why a ceiling cannot be applied, if it cannot: one below what the loaded
/// weights need.
fn ceiling_change_refusal(
    requested_mlx_memory_ceiling_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
) -> Option<InferenceEngineError> {
    if requested_mlx_memory_ceiling_bytes >= minimum_mlx_memory_ceiling_bytes {
        return None;
    }
    Some(InferenceEngineError::MlxMemoryLimitRejected {
        requested_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
        reason: "the requested ceiling is below what the folded-ternary weights need".to_owned(),
    })
}

/// The `mlx_buffer_cache` limit under a ceiling: the configured limit, never
/// more than the ceiling itself.
fn mlx_buffer_cache_bytes_for_ceiling(
    configured_mlx_buffer_cache_bytes: u64,
    mlx_memory_ceiling_bytes: u64,
) -> u64 {
    configured_mlx_buffer_cache_bytes.min(mlx_memory_ceiling_bytes)
}

/// The MLX limits for an accepted ceiling change. A ceiling at or above the
/// engine's minimum is positive and the `mlx_buffer_cache` limit never
/// exceeds it, so the limits always validate; the error path only guards
/// that rule.
fn memory_limits_for_ceiling(
    mlx_memory_ceiling_bytes: u64,
    mlx_buffer_cache_bytes: u64,
) -> Result<MlxMemoryLimits, InferenceEngineError> {
    MlxMemoryLimits::new(
        usize::try_from(mlx_memory_ceiling_bytes).unwrap_or(usize::MAX),
        usize::try_from(mlx_buffer_cache_bytes).unwrap_or(usize::MAX),
    )
    .map_err(|error| InferenceEngineError::Fatal {
        reason: format!("folded-ternary MLX limits failed: {error}"),
    })
}

#[cfg(test)]
#[path = "memory_limit_tests.rs"]
mod memory_limit_tests;
