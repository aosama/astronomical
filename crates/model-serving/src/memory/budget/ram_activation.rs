//! Activation-headroom resolution for the MLX RAM budget.
//!
//! The activation reserve belongs to one planned operation (one forward), so
//! it resolves from the operation's own token count and learned per-bucket
//! evidence — never from the prompt or total context length. Sizing it from
//! the context length multiplied a chunk-shaped observation into a reserve
//! several times the ceiling and rejected every later request (issue #690).
//!
//! Two rules keep this quantity honest, and both were learned from the same
//! production incident:
//!
//! 1. Evidence is only ever transferred between scopes by a MAXIMUM over
//!     observations already taken at or below the planned scope. Every
//!     observation already embeds the attended-context factor of the forward
//!     that produced it, so multiplying one by a token ratio counts that factor
//!     twice.
//! 2. The reserve is capped by what the loaded model can coexist with, not by
//!     the ceiling. Model core bytes are irrevocable, so a reserve larger than
//!     `ceiling - model_core` can never be granted and only converts a
//!     reclaimable over-reservation into a permanent rejection.
//!
//! The asymmetry behind both rules: UNDER-reserving for a genuinely larger
//! forward is recoverable (typed allocation failure, exact reclamation, retry),
//! while an over-reserve has no recovery path and starves every later request
//! until the model reloads and the learned evidence is forgotten.

use crate::memory::MemoryPhase;
use crate::memory::budget::ram::{MlxRamBudget, context_token_bucket};
use crate::memory::reclamation;

impl MlxRamBudget {
    /// Activation headroom for one planned operation.
    ///
    /// Prefill resolves the learned evidence for the planned operation size
    /// (the highest measured observation at or below the operation's bucket)
    /// so one large request's workspace does not size every later promise
    /// (issue #623 follow-up). The static three-layer floor still bounds every
    /// prefill promise. Decode evidence stays a scalar high-water: one-token
    /// writes are activation-cheap and phase-independent.
    ///
    /// The token count is the **operation's own size** (one forward's token
    /// count), never the prompt or total context length. A chunked prefill's
    /// activation is a function of the chunk size; sizing it from the context
    /// length multiplied a chunk-shaped observation into a reserve several
    /// times the ceiling (issue #690).
    #[must_use]
    pub fn activation_headroom_bytes(&self, phase: MemoryPhase, operation_token_count: u64) -> u64 {
        // A reserve the loaded model can never coexist with is paper by
        // construction: model core bytes cannot be evicted, so activation
        // above ceiling minus model core projects every nonzero live memory
        // above the ceiling and turns admission into a permanent rejection
        // instead of the reclaim path. Measured observations keep their pinned
        // dominance over the static floor within that coexistence bound
        // (issue #690; measured with a ceiling-equal cap, 2026-10-10).
        self.phase_activation_headroom_bytes(phase, operation_token_count)
            .min(
                self.mlx_active_memory_ceiling_bytes()
                    .saturating_sub(self.model_geometry.model_core_payload_bytes),
            )
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
                reclamation::required_complete_residency_activation_headroom_bytes(
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
                    reclamation::required_complete_residency_activation_headroom_bytes(
                        self.model_geometry.largest_complete_expert_layer_bytes,
                        self.prefill_activation_global_high_water_bytes(),
                    )
                } else {
                    // Decode follows prefill in the user journey. Until one
                    // decode completes, prefill high-water is the only live
                    // evidence preventing warm fill from occupying transient
                    // space that the first token immediately needs back.
                    reclamation::required_complete_residency_activation_headroom_bytes(
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
    /// The promise is the highest measured observation at or below the planned
    /// operation's token bucket. Beyond the measured span the static
    /// three-layer floor governs until a forward at that scope completes:
    /// scaling a smaller-scope observation by a token-count ratio
    /// double-counts the attended-context factor every observation already
    /// embeds, and the manufactured paper reserve reached the whole ceiling and
    /// rejected every later request (measured 2026-10-10). Under-reservation
    /// for a genuinely larger single forward is recoverable through the typed
    /// allocation-failure rollback, exact reclamation, and retry; an inflated
    /// reserve has no recovery path.
    ///
    /// The range is INCLUSIVE of the planned bucket on purpose. Activation
    /// grows with attended context, so a smaller observation from a lower
    /// bucket must never override a larger known one from a bucket at or below
    /// the plan — that ordering is the issue #623 lesson.
    fn learned_prefill_activation_headroom_bytes(&self, operation_token_count: u64) -> u64 {
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
