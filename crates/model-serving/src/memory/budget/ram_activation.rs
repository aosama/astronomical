//! Activation-headroom resolution for the MLX RAM budget.
//!
//! The activation reserve belongs to one planned operation (one forward), so
//! it resolves from the operation's own token count and learned per-bucket
//! evidence — never from the prompt or total context length. Sizing it from
//! the context length multiplied a chunk-shaped observation into a reserve
//! several times the ceiling and rejected every later request (issue #690).

use crate::memory::MemoryPhase;
use crate::memory::budget::ram::{
    CONTEXT_TOKEN_BUCKET_WIDTH, MlxRamBudget, context_token_bucket,
    scale_bytes_proportionally_to_token_count,
};
use crate::memory::reclamation::required_complete_residency_activation_headroom_bytes;

impl MlxRamBudget {
    /// Activation headroom for one planned operation.
    ///
    /// Prefill resolves the learned evidence for the planned operation size
    /// (highest at-or-below bucket, proportionally projected beyond the highest
    /// measured bucket) so one large request's workspace does not size every
    /// later promise (issue #623 follow-up). The static three-layer floor still
    /// bounds every prefill promise. Decode evidence stays a scalar high-water:
    /// one-token writes are activation-cheap and phase-independent.
    ///
    /// The token count is the **operation's own size** (one forward's token
    /// count), never the prompt or total context length. A chunked prefill's
    /// activation is a function of the chunk size; sizing it from the context
    /// length multiplied a chunk-shaped observation into a reserve several
    /// times the ceiling (issue #690).
    #[must_use]
    pub fn activation_headroom_bytes(&self, phase: MemoryPhase, operation_token_count: u64) -> u64 {
        // A reserve above the ceiling can never be admitted, so a projection
        // beyond it is meaningless paper. Cap learned evidence at the ceiling:
        // measured observations keep their pinned dominance over the static
        // floor, but no projection — measured or scaled — may manufacture a
        // reserve the ceiling could never grant (issue #690).
        self.phase_activation_headroom_bytes(phase, operation_token_count)
            .min(self.mlx_active_memory_ceiling_bytes())
    }

    fn phase_activation_headroom_bytes(
        &self,
        phase: MemoryPhase,
        operation_token_count: u64,
    ) -> u64 {
        match phase {
            MemoryPhase::Prefill => {
                // One complete layer is enough to stream a single page. A
                // seated 38.6 GB model under a 40 GB ceiling still needs room
                // for a multi-token prefill working set; three layers is the
                // measured first-chunk overshoot on that shape. Keep the
                // learned high-water when it is larger.
                required_complete_residency_activation_headroom_bytes(
                    self.model_geometry
                        .largest_complete_expert_layer_bytes
                        .saturating_mul(3),
                    self.learned_prefill_activation_headroom_bytes(operation_token_count),
                )
            }
            // GenerationPreparation budgets like decode (see the mapping
            // contract in phase.rs): token writing is activation-cheap.
            MemoryPhase::GenerationPreparation | MemoryPhase::Decode => {
                if self.has_decode_activation_measurement {
                    self.decode_activation_headroom_bytes
                } else if self.has_prefill_activation_measurement {
                    required_complete_residency_activation_headroom_bytes(
                        self.model_geometry.largest_complete_expert_layer_bytes,
                        self.prefill_activation_global_high_water_bytes(),
                    )
                } else {
                    // Decode follows prefill in the user journey. Until one
                    // decode completes, prefill high-water is the only live
                    // evidence preventing warm fill from occupying transient
                    // space that the first token immediately needs back.
                    required_complete_residency_activation_headroom_bytes(
                        self.model_geometry.largest_complete_expert_layer_bytes,
                        0,
                    )
                }
            }
            MemoryPhase::Idle => 0,
        }
    }

    /// Learned prefill activation evidence resolved for the planned operation.
    ///
    /// Within the measured span the highest at-or-bucket observation wins
    /// (activation grows with the attended context, so a smaller observation
    /// must not override a larger known lower bucket). Beyond the highest
    /// measured token count the highest evidence scales proportionally by
    /// token count, mirroring the context-window reserve's projection rule.
    /// The token count is the planned operation's own size, never the prompt
    /// or total context length (issue #690).
    fn learned_prefill_activation_headroom_bytes(&self, operation_token_count: u64) -> u64 {
        let Some(&highest_bucket_high_water_bytes) = self
            .prefill_activation_high_water_by_token_bucket
            .values()
            .last()
        else {
            return 0;
        };
        let highest_measured_bucket = *self
            .prefill_activation_high_water_by_token_bucket
            .keys()
            .next_back()
            .expect("the map is nonempty above");
        let highest_measured_token_count =
            highest_measured_bucket.saturating_add(1) * CONTEXT_TOKEN_BUCKET_WIDTH;
        if operation_token_count > highest_measured_token_count {
            // Proportional projection beyond measured evidence; the ceiling
            // division keeps the projection conservative.
            let scaled_high_water_bytes = scale_bytes_proportionally_to_token_count(
                highest_bucket_high_water_bytes,
                highest_measured_token_count,
                operation_token_count,
            );
            let at_or_below_high_water_bytes = self
                .prefill_activation_high_water_by_token_bucket
                .range(..=context_token_bucket(operation_token_count))
                .map(|(_, measured_bytes)| *measured_bytes)
                .max()
                .unwrap_or(0);
            return at_or_below_high_water_bytes.max(scaled_high_water_bytes);
        }
        self.prefill_activation_high_water_by_token_bucket
            .range(..=context_token_bucket(operation_token_count))
            .map(|(_, measured_bytes)| *measured_bytes)
            .max()
            .unwrap_or(0)
    }

    /// The largest prefill activation observation across every context bucket.
    /// Protects the first decode before decode has its own evidence.
    fn prefill_activation_global_high_water_bytes(&self) -> u64 {
        self.prefill_activation_high_water_by_token_bucket
            .values()
            .copied()
            .max()
            .unwrap_or(0)
    }
}
