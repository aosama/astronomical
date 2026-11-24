//! The resident engine's own step in a live MLX memory-ceiling change; the
//! shared Qwen3.5 change runs every other step.

use runtime_integration::MlxMemoryLimits;

use crate::InferenceEngineError;
use crate::qwen3_5_core::engine_errors::mlx_memory_ceiling_change_before_load_error;
use crate::qwen3_5_core::shared_execution::mlx_memory_limit_change::Qwen3_5MlxMemoryLimitHooks;

use super::Qwen3_5EngineState;

impl Qwen3_5MlxMemoryLimitHooks for Qwen3_5EngineState {
    /// The resident model reclaims under the old ceiling before lowering it
    /// and refuses a ceiling its weights cannot fit; the memory governor
    /// then follows the installed limit, so the next forward is admitted
    /// against the ceiling the user just chose.
    fn change_model_mlx_memory_ceiling(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(u64, MlxMemoryLimits), InferenceEngineError> {
        let model = self
            .core
            .model
            .as_mut()
            .ok_or_else(mlx_memory_ceiling_change_before_load_error)?;
        let (minimum_mlx_memory_ceiling_bytes, updated_mlx_memory_limits) =
            model.update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)?;
        model.govern_limit_change(updated_mlx_memory_limits.active_memory_limit_bytes() as u64);
        Ok((minimum_mlx_memory_ceiling_bytes, updated_mlx_memory_limits))
    }
}
