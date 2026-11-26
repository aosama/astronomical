//! The streaming engine's own steps in a live MLX memory-ceiling change; the
//! shared Qwen3.5 change runs every other step.

use runtime_integration::MlxMemoryLimits;

use crate::{InferenceEngineError, safe_minimum_active_memory_ceiling_bytes};
use text_generation::engine_errors::mlx_memory_ceiling_change_before_load_error;
use text_generation::generation_loop::mlx_memory_limit_change::TextMlxMemoryLimitHooks;

use super::Qwen3_5EngineState;

/// Calculates the smallest safe idle MLX ceiling for a loaded model.
///
/// Retained expert pages are reclaimable. The remaining live model bytes plus
/// one maximum routed expert page are non-evictable for the next request.
#[must_use]
pub const fn safe_minimum_mlx_memory_ceiling_bytes(
    current_idle_active_mlx_memory_bytes: u64,
    evictable_retained_expert_payload_bytes: u64,
    maximum_expert_page_reserve_bytes: u64,
) -> u64 {
    safe_minimum_active_memory_ceiling_bytes(
        current_idle_active_mlx_memory_bytes,
        evictable_retained_expert_payload_bytes,
        maximum_expert_page_reserve_bytes,
    )
}

impl TextMlxMemoryLimitHooks for Qwen3_5EngineState {
    /// Any change of the ceiling is taken in place: this engine serves a
    /// model whether or not all of it fits, keeping more in memory after a
    /// raise and giving weights back before a lowering (issue #2585).
    fn refuse_mlx_memory_ceiling_change(
        &self,
        _requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    /// The streaming model gives back retained experts and kept dense layers
    /// before lowering, and moves its expert pager and memory governor with
    /// the installed limit, rolling every part back when a step fails.
    fn change_model_mlx_memory_ceiling(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(u64, MlxMemoryLimits), InferenceEngineError> {
        self.core
            .model
            .as_mut()
            .ok_or_else(mlx_memory_ceiling_change_before_load_error)?
            .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
    }
}
