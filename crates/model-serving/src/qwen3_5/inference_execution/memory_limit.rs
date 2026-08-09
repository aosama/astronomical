use std::time::Instant;

use crate::{
    InferenceEngineError, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
    Qwen3_5FeedForwardArchitecture,
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
    current_idle_active_mlx_memory_bytes
        .saturating_sub(evictable_retained_expert_payload_bytes)
        .saturating_add(maximum_expert_page_reserve_bytes)
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
        let retained_complete_layer_count_before = model
            .expert_weight_memory_cache_statistics()
            .complete_layer_count;
        let adjustment_started_at = Instant::now();
        let (
            minimum_mlx_memory_ceiling_bytes,
            did_change_mlx_memory_ceiling,
            updated_mlx_memory_limits,
        ) = model.update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)?;
        self.adaptive_ram_growth_guard
            .update_active_memory_limit_bytes(updated_mlx_memory_limits.active_memory_limit_bytes())
            .map_err(|adaptive_ram_growth_guard_error| {
                super::fatal_engine_error(adaptive_ram_growth_guard_error.to_string())
            })?;
        self.memory_limits = updated_mlx_memory_limits;

        if did_change_mlx_memory_ceiling
            && requested_mlx_memory_ceiling_bytes > old_mlx_memory_ceiling_bytes
        {
            match model.config().feed_forward_architecture() {
                Qwen3_5FeedForwardArchitecture::Dense => {}
                Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
                    let mut disabled_performance_attribution =
                        crate::PerformanceAttribution::disabled();
                    if let Err(expert_layer_recovery_error) = model
                        .prewarm_complete_expert_layers_with_performance_attribution(
                            &mut disabled_performance_attribution,
                        )
                    {
                        tracing::warn!(
                            error = %expert_layer_recovery_error,
                            "could not recover complete expert layers after raising the MLX memory ceiling"
                        );
                    }
                }
            }
        }

        let mlx_memory_snapshot_after_adjustment = model
            .runtime()
            .memory_snapshot()
            .map_err(super::qwen3_5_runtime_error)?;
        let retained_complete_layer_count_after = model
            .expert_weight_memory_cache_statistics()
            .complete_layer_count;
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
        let mlx_memory_telemetry = Some(MlxMemoryTelemetry::new(
            active_memory_bytes,
            allocator_cache_memory_bytes,
            peak_memory_bytes,
            model.finalized_active_memory_breakdown(active_memory_bytes, 0),
        ));
        let expert_memory_mode = model.expert_memory_mode();
        tracing::info!(
            old_mlx_memory_ceiling_bytes,
            new_mlx_memory_ceiling_bytes = requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            adjustment_elapsed_millis = adjustment_started_at.elapsed().as_millis(),
            active_mlx_memory_bytes_before = mlx_memory_snapshot_before_adjustment.active_memory_bytes(),
            active_mlx_memory_bytes_after = mlx_memory_snapshot_after_adjustment.active_memory_bytes(),
            allocator_cache_memory_bytes_before = mlx_memory_snapshot_before_adjustment.allocator_cache_memory_bytes(),
            allocator_cache_memory_bytes_after = mlx_memory_snapshot_after_adjustment.allocator_cache_memory_bytes(),
            retained_complete_layer_count_before,
            retained_complete_layer_count_after,
            expert_memory_mode = ?expert_memory_mode,
            "applied live MLX memory ceiling adjustment"
        );
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            expert_memory_mode,
            mlx_memory_telemetry,
        ))
    }
}
