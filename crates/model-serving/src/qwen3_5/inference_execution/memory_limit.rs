use std::time::Instant;

use crate::{
    InferenceEngineError, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
    safe_minimum_active_memory_ceiling_bytes,
};

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

impl Qwen3_5EngineState {
    pub(super) fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        if self.active_request.is_some() {
            return Err(InferenceEngineError::EngineBusy);
        }
        let Some(model) = self.model.as_mut() else {
            return Err(super::fatal_engine_error(
                "cannot update the MLX memory ceiling before the model is loaded",
            ));
        };
        let old_mlx_memory_ceiling_bytes =
            u64::try_from(self.memory_limits.active_memory_limit_bytes()).map_err(|_| {
                super::fatal_engine_error("current MLX memory ceiling exceeds the u64 range")
            })?;
        let mlx_memory_snapshot_before_adjustment = model
            .runtime()
            .memory_snapshot()
            .map_err(super::qwen3_5_runtime_error)?;
        let expert_memory_mode_before_adjustment = model.expert_memory_mode();
        let adjustment_started_at = Instant::now();
        let (minimum_mlx_memory_ceiling_bytes, updated_mlx_memory_limits) =
            model.update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)?;
        self.adaptive_ram_growth_guard
            .update_active_memory_ceiling_bytes(
                updated_mlx_memory_limits.active_memory_limit_bytes(),
            )
            .map_err(|adaptive_ram_growth_guard_error| {
                super::fatal_engine_error(adaptive_ram_growth_guard_error.to_string())
            })?;
        self.memory_limits = updated_mlx_memory_limits;

        let mlx_memory_snapshot_after_adjustment = model
            .runtime()
            .memory_snapshot()
            .map_err(super::qwen3_5_runtime_error)?;
        let active_memory_bytes = u64::try_from(
            mlx_memory_snapshot_after_adjustment.active_memory_bytes(),
        )
        .map_err(|_| super::fatal_engine_error("MLX active memory exceeds the u64 range"))?;
        let allocator_cache_memory_bytes =
            u64::try_from(mlx_memory_snapshot_after_adjustment.allocator_cache_memory_bytes())
                .map_err(|_| {
                    super::fatal_engine_error("MLX allocator-cache memory exceeds the u64 range")
                })?;
        let peak_memory_bytes =
            u64::try_from(mlx_memory_snapshot_after_adjustment.peak_memory_bytes())
                .map_err(|_| super::fatal_engine_error("MLX peak memory exceeds the u64 range"))?;
        let active_memory_breakdown =
            model.finalized_active_memory_breakdown(active_memory_bytes, 0);
        // One claim serves both the telemetry and the adjustment event, derived
        // from the breakdown of this exact snapshot so neither publication can
        // straddle a promotion or demote (issue #337).
        let expert_residency =
            model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown);
        let mlx_memory_telemetry = Some(
            MlxMemoryTelemetry::new(
                active_memory_bytes,
                allocator_cache_memory_bytes,
                peak_memory_bytes,
                active_memory_breakdown,
            )
            .with_expert_residency_telemetry(expert_residency),
        );
        let expert_memory_mode = model.expert_memory_mode();
        let expert_residency_transition_occurred =
            expert_memory_mode != expert_memory_mode_before_adjustment;
        tracing::info!(
            old_mlx_memory_ceiling_bytes,
            new_mlx_memory_ceiling_bytes = requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            adjustment_elapsed_millis = adjustment_started_at.elapsed().as_millis(),
            active_mlx_memory_bytes_before = mlx_memory_snapshot_before_adjustment.active_memory_bytes(),
            active_mlx_memory_bytes_after = mlx_memory_snapshot_after_adjustment.active_memory_bytes(),
            allocator_cache_memory_bytes_before = mlx_memory_snapshot_before_adjustment.allocator_cache_memory_bytes(),
            allocator_cache_memory_bytes_after = mlx_memory_snapshot_after_adjustment.allocator_cache_memory_bytes(),
            expert_memory_mode = ?expert_memory_mode,
            expert_residency_transition_occurred,
            "applied live MLX memory ceiling adjustment"
        );
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            u64::try_from(updated_mlx_memory_limits.allocator_cache_memory_limit_bytes()).map_err(
                |_| {
                    super::fatal_engine_error(
                        "updated MLX allocator-cache limit exceeds the u64 range",
                    )
                },
            )?,
            minimum_mlx_memory_ceiling_bytes,
            expert_memory_mode,
            mlx_memory_telemetry,
        )
        .with_expert_residency_telemetry(expert_residency))
    }
}
