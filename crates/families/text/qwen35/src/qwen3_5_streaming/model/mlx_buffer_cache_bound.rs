//! Bounding the `mlx_buffer_cache` to the working buffers the loaded model
//! measured (issue #2533).
//!
//! The worker starts every engine with an `mlx_buffer_cache` limit equal to
//! the active limit, before any model is measured. Once the probe at load
//! knows one prompt step's MLX working buffers, the limit drops to that
//! step, so freed buffers beyond it go back to macOS instead of staying
//! wired beside a full `astro_expert_cache`.

use runtime_integration::{
    MlxMemoryLimits, MlxRuntimeError, measured_mlx_buffer_cache_limit_bytes,
};

use super::Qwen3_5Model;

impl Qwen3_5Model {
    /// Installs the measured `mlx_buffer_cache` limit; nothing changes while
    /// no prompt step has been measured.
    pub fn bound_mlx_buffer_cache_to_measured_working_buffers(
        &mut self,
    ) -> Result<(), MlxRuntimeError> {
        let Some(measured_step_bytes) = self.measured_prompt_mlx_working_buffer_bytes() else {
            return Ok(());
        };
        let active_memory_limit_bytes = self
            .base
            .runtime
            .memory_limits()
            .active_memory_limit_bytes();
        let mlx_buffer_cache_limit_bytes =
            measured_mlx_buffer_cache_limit_bytes(active_memory_limit_bytes, measured_step_bytes);
        tracing::info!(
            active_memory_limit_bytes,
            measured_step_bytes,
            mlx_buffer_cache_limit_bytes,
            "bounded the mlx_buffer_cache to one measured prompt step"
        );
        self.base.runtime.update_memory_limits(MlxMemoryLimits::new(
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
        )?)
    }
}
