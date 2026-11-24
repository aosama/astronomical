use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;
use std::time::Instant;

use crate::InferenceEngineError;

use super::Qwen3_5EngineState;

impl Qwen3_5EngineState {
    pub(in crate::qwen3_5_resident::inference_execution) fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        if self.core.active_request.is_some() {
            return Err(InferenceEngineError::EngineBusy);
        }
        let live_retained_context_state_payload_bytes =
            self.live_retained_context_state_payload_bytes();
        let Some(model) = self.core.model.as_mut() else {
            return Err(super::fatal_engine_error(
                "cannot update the MLX memory ceiling before the model is loaded",
            ));
        };
        let old_mlx_memory_ceiling_bytes =
            u64::try_from(self.core.memory_limits.active_memory_limit_bytes()).map_err(
                super::super::generation_session_lifecycle::generation_error_constructors::current_mlx_memory_ceiling_exceeds_u64_range_error,
            )?;
        let mlx_memory_snapshot_before_adjustment = model
            .runtime()
            .memory_snapshot()
            .map_err(super::qwen3_5_runtime_error)?;
        let expert_memory_mode_before_adjustment = model.expert_memory_mode();
        let adjustment_started_at = Instant::now();
        let (minimum_mlx_memory_ceiling_bytes, updated_mlx_memory_limits) =
            model.update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)?;
        self.core.memory_limits = updated_mlx_memory_limits;
        model.govern_limit_change(updated_mlx_memory_limits.active_memory_limit_bytes() as u64);

        let mlx_memory_snapshot_after_adjustment = model
            .runtime()
            .memory_snapshot()
            .map_err(super::qwen3_5_runtime_error)?;
        let active_memory_bytes = u64::try_from(
            mlx_memory_snapshot_after_adjustment.active_memory_bytes(),
        )
        .map_err(super::super::generation_session_lifecycle::generation_error_constructors::mlx_active_memory_exceeds_u64_range_error)?;
        let mlx_buffer_cache_bytes =
            u64::try_from(mlx_memory_snapshot_after_adjustment.mlx_buffer_cache_bytes())
.map_err(super::super::generation_session_lifecycle::generation_error_constructors::mlx_buffer_cache_exceeds_u64_range_error)?;
        let peak_memory_bytes =
            u64::try_from(mlx_memory_snapshot_after_adjustment.peak_memory_bytes())
                .map_err(super::super::generation_session_lifecycle::generation_error_constructors::mlx_peak_memory_exceeds_u64_range_error)?;
        let active_memory_breakdown = model.finalized_active_memory_breakdown(
            active_memory_bytes,
            live_retained_context_state_payload_bytes,
            0,
        );
        // One claim serves both the telemetry and the adjustment event, derived
        // from the breakdown of this exact snapshot so neither publication can
        // straddle a promotion or demote (issue #337).
        let expert_residency =
            model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown);
        let mlx_memory_telemetry = Some(
            MlxMemoryTelemetry::new(
                active_memory_bytes,
                mlx_buffer_cache_bytes,
                peak_memory_bytes,
                active_memory_breakdown,
            )
            .with_expert_residency_telemetry(expert_residency)
            // Issue #510: the adjustment event lands after the ModelLoaded
            // sample, so a split-less telemetry here would overwrite the
            // load-complete split on the status the menu paints.
            .with_memory_ceiling_utilization(
                model.between_request_memory_ceiling_utilization_for_breakdown(
                    active_memory_bytes,
                    active_memory_breakdown,
                ),
            )
            .with_memory_ledger_reading(model.latest_memory_ledger_reading()),
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
            mlx_buffer_cache_bytes_before = mlx_memory_snapshot_before_adjustment.mlx_buffer_cache_bytes(),
            mlx_buffer_cache_bytes_after = mlx_memory_snapshot_after_adjustment.mlx_buffer_cache_bytes(),
            expert_memory_mode = ?expert_memory_mode,
            expert_residency_transition_occurred,
            "applied live MLX memory ceiling adjustment"
        );
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            u64::try_from(updated_mlx_memory_limits.mlx_buffer_cache_limit_bytes()).map_err(
                |_| {
                    super::fatal_engine_error(
                        "updated MLX buffer cache limit exceeds the u64 range",
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

use crate::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
