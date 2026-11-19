//! Keeps the memory macOS has wired across the whole system under the line
//! where it stops keeping MLX's memory wired.

use super::Qwen3_5Model;
use crate::memory::macos_wired_memory::read_macos_wired_memory;

impl Qwen3_5Model {
    /// Reads the system's wired memory and lets the memory governor move
    /// the kept-weight capacity by how far that total is over or under the
    /// kernel's wire limit. Called after every measured forward, when
    /// what MLX holds is settled.
    pub fn govern_system_wired_memory(&self) {
        let Some(wired_memory_reading) = read_macos_wired_memory() else {
            tracing::debug!("macOS did not report its wired memory; the plan stands");
            return;
        };
        let (reserve_before_bytes, kept_weight_capacity, reserve_after_bytes) = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            let reserve_before_bytes = memory_governor.accounts().system_wired_reserve_bytes;
            let kept_weight_capacity = memory_governor.observe_system_wired(
                wired_memory_reading.system_wired_bytes,
                wired_memory_reading.system_wire_limit_bytes,
            );
            (
                reserve_before_bytes,
                kept_weight_capacity,
                memory_governor.accounts().system_wired_reserve_bytes,
            )
        };
        self.apply_kept_weight_capacity(kept_weight_capacity);
        if reserve_after_bytes > reserve_before_bytes {
            self.release_given_back_experts_from_the_mlx_buffer_cache(
                wired_memory_reading.system_wired_bytes,
                reserve_after_bytes,
            );
        }
    }

    /// Experts the astro expert cache gives back go to the MLX buffer cache,
    /// which keeps them wired: without this release the system's wired total
    /// would not fall and the reserve would grow on every reading. Measured:
    /// the buffer cache reached 6 GB and replies fell to 5 tokens per second.
    fn release_given_back_experts_from_the_mlx_buffer_cache(
        &self,
        system_wired_bytes: u64,
        system_wired_reserve_bytes: u64,
    ) {
        tracing::debug!(
            system_wired_bytes,
            system_wired_reserve_bytes,
            "macOS has more memory wired than it keeps wired; the astro expert cache gives some back"
        );
        if let Err(release_error) = self
            .runtime
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
        {
            tracing::warn!(
                error = %release_error,
                "could not release the MLX buffer cache after giving experts back"
            );
        }
    }
}
