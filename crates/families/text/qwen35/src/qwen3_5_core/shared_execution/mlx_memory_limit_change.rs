//! A live MLX memory-ceiling change, for every Qwen3.5 engine: the steps all
//! of them take, in the order all of them take them, with the engine's own
//! rule at two points.
//!
//! A change waits until no request is running and needs a loaded model. The
//! engine may refuse it before anything moves (the streaming engine sends a
//! raise that admits complete expert residency to a replacement worker), and
//! each engine moves its model to the new ceiling its own way: the resident
//! engine reclaims before lowering and moves its memory governor, the
//! streaming engine moves its expert pager and kept layers, and the
//! folded-ternary engine caps its `mlx_buffer_cache` at the configured limit.
//! The adjustment then reports the memory reading taken after the change.

use std::time::Instant;

use ipc_protocol::ExpertMemoryMode;
use runtime_integration::{MlxMemoryLimits, MlxMemorySnapshot};

use crate::qwen3_5_core::engine_errors::{
    mlx_buffer_cache_limit_exceeds_u64_range_error, mlx_memory_ceiling_change_before_load_error,
    mlx_memory_ceiling_exceeds_u64_range_error,
};
use crate::{InferenceEngineError, MlxMemoryLimitAdjustment};

use super::generation_finalization_steps::{
    Qwen3_5FinalizationHooks, idle_mlx_memory_telemetry_for_snapshot,
};
use super::text_model::Qwen3_5TextModel;

/// The points in a ceiling change where the engines differ.
pub trait Qwen3_5MlxMemoryLimitHooks: Qwen3_5FinalizationHooks {
    /// Refuses, before anything moves, a change this engine must not absorb
    /// in place. Engines that can absorb every change they can serve keep
    /// this default.
    fn refuse_mlx_memory_ceiling_change(
        &self,
        _requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    /// Moves the loaded model to the requested ceiling, returning the
    /// smallest ceiling the model can serve under and the MLX limits now
    /// installed in its runtime.
    fn change_model_mlx_memory_ceiling(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(u64, MlxMemoryLimits), InferenceEngineError>;
}

/// Applies a live MLX memory-ceiling change and reports the memory after it.
pub fn update_mlx_memory_limit<Engine: Qwen3_5MlxMemoryLimitHooks>(
    engine: &mut Engine,
    requested_mlx_memory_ceiling_bytes: u64,
) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
    if engine.execution_core().active_request.is_some() {
        return Err(InferenceEngineError::EngineBusy);
    }
    engine.refuse_mlx_memory_ceiling_change(requested_mlx_memory_ceiling_bytes)?;
    let old_mlx_memory_ceiling_bytes = u64::try_from(
        engine
            .execution_core()
            .memory_limits
            .active_memory_limit_bytes(),
    )
    .map_err(mlx_memory_ceiling_exceeds_u64_range_error)?;
    let model = engine
        .execution_core()
        .model
        .as_ref()
        .ok_or_else(mlx_memory_ceiling_change_before_load_error)?;
    let mlx_memory_snapshot_before_adjustment = model.memory_snapshot_for_telemetry()?;
    let expert_memory_mode_before_adjustment = model.expert_memory_mode();
    let adjustment_started_at = Instant::now();
    let (minimum_mlx_memory_ceiling_bytes, updated_mlx_memory_limits) =
        engine.change_model_mlx_memory_ceiling(requested_mlx_memory_ceiling_bytes)?;
    engine.execution_core_mut().memory_limits = updated_mlx_memory_limits;
    let model = engine.execution_core().loaded_model()?;
    let mlx_memory_snapshot_after_adjustment = model.memory_snapshot_for_telemetry()?;
    let (expert_residency_telemetry, mlx_memory_telemetry) =
        idle_mlx_memory_telemetry_for_snapshot(
            engine,
            model,
            &mlx_memory_snapshot_after_adjustment,
        )?;
    let expert_memory_mode = model.expert_memory_mode();
    log_applied_mlx_memory_ceiling_change(&AppliedMlxMemoryCeilingChange {
        old_mlx_memory_ceiling_bytes,
        new_mlx_memory_ceiling_bytes: requested_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
        adjustment_started_at,
        mlx_memory_snapshot_before_adjustment,
        mlx_memory_snapshot_after_adjustment,
        expert_memory_mode_before_adjustment,
        expert_memory_mode,
    });
    let mlx_buffer_cache_limit_bytes =
        u64::try_from(updated_mlx_memory_limits.mlx_buffer_cache_limit_bytes())
            .map_err(mlx_buffer_cache_limit_exceeds_u64_range_error)?;
    Ok(MlxMemoryLimitAdjustment::new(
        requested_mlx_memory_ceiling_bytes,
        mlx_buffer_cache_limit_bytes,
        minimum_mlx_memory_ceiling_bytes,
        expert_memory_mode,
        Some(mlx_memory_telemetry),
    )
    .with_expert_residency_telemetry(expert_residency_telemetry))
}

/// What one applied change logs: the ceilings, how long the change took,
/// and the memory and expert placement on either side of it.
struct AppliedMlxMemoryCeilingChange {
    old_mlx_memory_ceiling_bytes: u64,
    new_mlx_memory_ceiling_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
    adjustment_started_at: Instant,
    mlx_memory_snapshot_before_adjustment: MlxMemorySnapshot,
    mlx_memory_snapshot_after_adjustment: MlxMemorySnapshot,
    expert_memory_mode_before_adjustment: ExpertMemoryMode,
    expert_memory_mode: ExpertMemoryMode,
}

fn log_applied_mlx_memory_ceiling_change(applied_change: &AppliedMlxMemoryCeilingChange) {
    let before = &applied_change.mlx_memory_snapshot_before_adjustment;
    let after = &applied_change.mlx_memory_snapshot_after_adjustment;
    tracing::info!(
        old_mlx_memory_ceiling_bytes = applied_change.old_mlx_memory_ceiling_bytes,
        new_mlx_memory_ceiling_bytes = applied_change.new_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes = applied_change.minimum_mlx_memory_ceiling_bytes,
        adjustment_elapsed_millis = applied_change.adjustment_started_at.elapsed().as_millis(),
        active_mlx_memory_bytes_before = before.active_memory_bytes(),
        active_mlx_memory_bytes_after = after.active_memory_bytes(),
        mlx_buffer_cache_bytes_before = before.mlx_buffer_cache_bytes(),
        mlx_buffer_cache_bytes_after = after.mlx_buffer_cache_bytes(),
        expert_memory_mode = ?applied_change.expert_memory_mode,
        expert_residency_transition_occurred = applied_change.expert_memory_mode
            != applied_change.expert_memory_mode_before_adjustment,
        "applied live MLX memory ceiling adjustment"
    );
}
