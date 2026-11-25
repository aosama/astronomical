//! The streaming engine's own steps in a live MLX memory-ceiling change; the
//! shared Qwen3.5 change runs every other step.

use runtime_integration::MlxMemoryLimits;

use crate::{InferenceEngineError, safe_minimum_active_memory_ceiling_bytes};
use text_generation::engine_errors::{
    mlx_memory_ceiling_change_before_load_error, mlx_memory_ceiling_exceeds_u64_range_error,
};
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
    /// A raise that now admits complete expert residency must not be
    /// absorbed in place: this engine structurally never reports complete
    /// ownership, and the startup selection a replacement worker runs
    /// would load the resident engine under the persisted ceiling.
    /// Demanding the replacement keeps the live worker congruent with
    /// what a restart would choose.
    fn refuse_mlx_memory_ceiling_change(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        let current_mlx_memory_ceiling_bytes =
            u64::try_from(self.core.memory_limits.active_memory_limit_bytes())
                .map_err(mlx_memory_ceiling_exceeds_u64_range_error)?;
        if requested_mlx_memory_ceiling_bytes > current_mlx_memory_ceiling_bytes
            && self.core.model.is_some()
            && self
                .startup_residency_facts
                .is_some_and(|startup_residency_facts| {
                    startup_residency_facts.raised_ceiling_selects_complete_residency(
                        requested_mlx_memory_ceiling_bytes,
                    )
                })
        {
            return Err(InferenceEngineError::StreamingResidentPromotionRequired {
                reason:
                    "the raised ceiling admits complete expert residency; a replacement worker must load the resident engine"
                        .to_owned(),
            });
        }
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
