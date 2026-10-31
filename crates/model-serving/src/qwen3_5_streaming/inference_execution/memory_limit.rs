use std::time::Instant;

use crate::{InferenceEngineError, safe_minimum_active_memory_ceiling_bytes};

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
        self.require_no_resident_promotion(requested_mlx_memory_ceiling_bytes)?;
        let old_mlx_memory_ceiling_bytes = self.current_mlx_memory_ceiling_bytes()?;
        let Some(model) = self.model.as_mut() else {
            return Err(super::fatal_engine_error(
                "cannot update the MLX memory ceiling before the model is loaded",
            ));
        };
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
        let (active_memory_bytes, allocator_cache_memory_bytes, peak_memory_bytes) =
            snapshot_memory_triple(&mlx_memory_snapshot_after_adjustment)?;
        let (expert_residency, mlx_memory_telemetry_value) = adjustment_telemetry(
            model,
            active_memory_bytes,
            allocator_cache_memory_bytes,
            peak_memory_bytes,
        );
        let mlx_memory_telemetry = Some(mlx_memory_telemetry_value);
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

    /// A raise that now admits complete expert residency must not be
    /// absorbed in place: this engine structurally never reports complete
    /// ownership, and the startup selection a replacement worker runs
    /// would load the resident engine under the persisted ceiling.
    /// Demanding the replacement keeps the live worker congruent with
    /// what a restart would choose.
    fn require_no_resident_promotion(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        if requested_mlx_memory_ceiling_bytes > self.current_mlx_memory_ceiling_bytes()?
            && self.model.is_some()
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

    fn current_mlx_memory_ceiling_bytes(&self) -> Result<u64, InferenceEngineError> {
        u64::try_from(self.memory_limits.active_memory_limit_bytes()).map_err(|_| {
            super::fatal_engine_error("current MLX memory ceiling exceeds the u64 range")
        })
    }
}

/// Reads one memory snapshot's three byte counters as u64 values. Every
/// counter is a usize in the runtime; a value beyond the u64 range means the
/// accounting itself is broken, which is fatal rather than clampable.
fn snapshot_memory_triple(
    mlx_memory_snapshot: &astronomical_runtime_integration::MlxMemorySnapshot,
) -> Result<(u64, u64, u64), InferenceEngineError> {
    let active_memory_bytes = u64::try_from(mlx_memory_snapshot.active_memory_bytes())
        .map_err(|_| super::fatal_engine_error("MLX active memory exceeds the u64 range"))?;
    let allocator_cache_memory_bytes =
        u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes()).map_err(|_| {
            super::fatal_engine_error("MLX allocator-cache memory exceeds the u64 range")
        })?;
    let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes())
        .map_err(|_| super::fatal_engine_error("MLX peak memory exceeds the u64 range"))?;
    Ok((
        active_memory_bytes,
        allocator_cache_memory_bytes,
        peak_memory_bytes,
    ))
}

/// Builds the telemetry pair one adjustment publishes: the expert residency
/// claim and the MLX memory telemetry. One claim serves both the telemetry
/// and the adjustment event, derived from the breakdown of this exact
/// snapshot so neither publication can straddle a promotion or demote
/// (issue #337). The utilization split rides along because the adjustment
/// event lands after the ModelLoaded sample — a split-less telemetry here
/// would overwrite the load-complete split on the status the menu paints
/// (issue #510).
fn adjustment_telemetry(
    model: &crate::qwen3_5_streaming::model::Qwen3_5Model,
    active_memory_bytes: u64,
    allocator_cache_memory_bytes: u64,
    peak_memory_bytes: u64,
) -> (crate::ExpertResidencyTelemetry, MlxMemoryTelemetry) {
    let active_memory_breakdown = model.finalized_active_memory_breakdown(active_memory_bytes, 0);
    let expert_residency = model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown);
    let mlx_memory_telemetry = MlxMemoryTelemetry::new(
        active_memory_bytes,
        allocator_cache_memory_bytes,
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
    );
    (expert_residency, mlx_memory_telemetry)
}

use crate::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
