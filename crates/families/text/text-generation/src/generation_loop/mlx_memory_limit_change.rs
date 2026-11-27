//! A live MLX memory-ceiling change, for every text engine: the steps all
//! of them take, in the order all of them take them, with the engine's own
//! rule at one point.
//!
//! A change needs a loaded model and is taken between two steps, whether a
//! request is running or not (issue #2681): the engine is told which, so a
//! running request's context and next step are kept room for. Every change is
//! taken in place: the Qwen3.5 text engine, which serves every text model,
//! moves its expert pager, kept layers and memory governor to the new
//! ceiling. The adjustment then reports the memory reading taken after the
//! change.

use std::time::Instant;

use ipc_protocol::ExpertMemoryMode;
use runtime_integration::{MlxMemoryLimits, MlxMemorySnapshot};

use crate::engine_errors::{
    mlx_buffer_cache_limit_exceeds_u64_range_error, mlx_memory_ceiling_change_before_load_error,
    mlx_memory_ceiling_exceeds_u64_range_error,
};
use crate::{InferenceEngineError, MlxMemoryLimitAdjustment};

use super::generation_finalization_steps::{
    TextFinalizationHooks, idle_mlx_memory_telemetry_for_snapshot,
};
use crate::model::text_model::TextModel;

/// The point in a ceiling change where the engines differ.
pub trait TextMlxMemoryLimitHooks: TextFinalizationHooks {
    /// Moves the loaded model to the requested ceiling, returning the
    /// smallest ceiling the model can serve under and the MLX limits now
    /// installed in its runtime. `request_is_running` says a request is
    /// between two of its steps, so what it holds cannot be given back.
    fn change_model_mlx_memory_ceiling(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
        request_is_running: bool,
    ) -> Result<(u64, MlxMemoryLimits), InferenceEngineError>;
}

/// Applies a live MLX memory-ceiling change and reports the memory after it.
pub fn update_mlx_memory_limit<Engine: TextMlxMemoryLimitHooks>(
    engine: &mut Engine,
    requested_mlx_memory_ceiling_bytes: u64,
) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
    let request_is_running = engine.execution_core().active_request.is_some();
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
    let (minimum_mlx_memory_ceiling_bytes, updated_mlx_memory_limits) = engine
        .change_model_mlx_memory_ceiling(requested_mlx_memory_ceiling_bytes, request_is_running)?;
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
