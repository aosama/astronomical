//! Keeps the memory macOS has wired across the whole system under the line
//! where it stops keeping MLX's memory wired.

use super::Qwen3_5Model;
use crate::memory::macos_wired_memory::read_macos_wired_memory;
use memory_policy::ledger::system_wired_ceiling::SystemWiredReading;

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
        let mlx_held_bytes = self.mlx_held_bytes();
        let (reserve_before_bytes, kept_weight_capacity, reserve_after_bytes) = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            let reserve_before_bytes = memory_governor.accounts().system_wired_reserve_bytes;
            let kept_weight_capacity = memory_governor.observe_system_wired(SystemWiredReading {
                system_wired_bytes: wired_memory_reading.system_wired_bytes,
                system_wire_limit_bytes: wired_memory_reading.system_wire_limit_bytes,
                mlx_held_bytes,
            });
            (
                reserve_before_bytes,
                kept_weight_capacity,
                memory_governor.accounts().system_wired_reserve_bytes,
            )
        };
        report_reserve_move(
            (
                wired_memory_reading.system_wired_bytes,
                wired_memory_reading.system_wire_limit_bytes,
                mlx_held_bytes,
            ),
            reserve_before_bytes,
            reserve_after_bytes,
        );
        self.apply_kept_weight_capacity(kept_weight_capacity);
        if reserve_after_bytes > reserve_before_bytes {
            self.release_given_back_experts_from_the_mlx_buffer_cache(
                wired_memory_reading.system_wired_bytes,
                reserve_after_bytes,
            );
        }
    }

    /// MLX's active memory and `mlx_buffer_cache` now; zero when MLX cannot
    /// say, which counts every wired byte as someone else's and so errs
    /// toward a larger reserve.
    fn mlx_held_bytes(&self) -> u64 {
        self.runtime
            .memory_snapshot()
            .map_or(0, |mlx_memory_snapshot| {
                u64::try_from(
                    mlx_memory_snapshot
                        .active_memory_bytes()
                        .saturating_add(mlx_memory_snapshot.mlx_buffer_cache_bytes()),
                )
                .unwrap_or(u64::MAX)
            })
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

/// Says when a reading moved the reserve, with what it read: `(system wired
/// total, kernel wire limit, MLX held)`. The reserve takes its room from the
/// `astro_expert_cache`, so a table dropped and read back soon after is
/// explained by these lines.
fn report_reserve_move(
    (system_wired_bytes, system_wire_limit_bytes, mlx_held_bytes): (u64, u64, u64),
    reserve_before_bytes: u64,
    reserve_after_bytes: u64,
) {
    if reserve_after_bytes != reserve_before_bytes {
        tracing::info!(
            system_wired_bytes,
            system_wire_limit_bytes,
            mlx_held_bytes,
            reserve_before_bytes,
            reserve_after_bytes,
            "the system wired reserve moved"
        );
    }
}
