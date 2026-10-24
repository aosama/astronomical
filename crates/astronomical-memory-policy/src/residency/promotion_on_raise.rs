//! Streaming-engine promotion gate for a raised MLX ceiling.
//!
//! A streaming worker is spawned only when its startup ceiling could not
//! admit complete expert residency. When the user later raises the ceiling
//! far enough, the honest transition is a replacement worker: the restart
//! re-runs startup engine selection, which loads the complete-owner
//! resident engine. This module owns the byte arithmetic that predicts that
//! selection from the same startup geometry and headroom the selection
//! itself uses, so a promotion demand can never disagree with the decision
//! the replacement worker actually makes.

use crate::admission::complete_residency::{
    CompleteResidencyDecision, CompleteResidencyRequirements,
};

/// Startup residency selection facts for one loaded streaming model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingStartupResidencyFacts {
    /// Non-expert resident core payload measured from the validated artifact.
    pub model_core_payload_bytes: u64,
    /// Bytes required when every sparse expert is fully resident.
    pub complete_expert_payload_bytes: u64,
    /// Context and activation headroom the startup selection requires above
    /// complete residency (the load-time gate-up fusion transient).
    pub startup_complete_residency_headroom_bytes: u64,
    /// Persistent decoder-state bytes one more context token adds.
    pub sequence_state_bytes_per_token: u64,
}

impl StreamingStartupResidencyFacts {
    /// Decides whether a replacement worker's startup engine selection would
    /// load the complete-owner resident engine under the raised ceiling.
    ///
    /// The input shape mirrors startup selection input-for-input (zero
    /// retained paged bytes, because a replacement worker loads from an
    /// empty process), keeping the prediction exact rather than
    /// approximate.
    #[must_use]
    pub fn raised_ceiling_selects_complete_residency(&self, raised_ceiling_bytes: u64) -> bool {
        matches!(
            CompleteResidencyRequirements {
                current_active_memory_bytes: self.model_core_payload_bytes,
                retained_paged_expert_payload_bytes: 0,
                complete_expert_payload_bytes: self.complete_expert_payload_bytes,
                required_headroom_bytes: self.startup_complete_residency_headroom_bytes,
                active_memory_ceiling_bytes: raised_ceiling_bytes,
            }
            .decide(),
            CompleteResidencyDecision::Admit { .. }
        )
    }

    /// Decides whether complete residency at `effective_ceiling_bytes` also
    /// leaves room for the incoming request's persistent decoder state.
    ///
    /// The static startup check alone admits a promotion whenever the model
    /// fits empty; a request whose own context still exceeds the remaining
    /// bytes would then be demoted again on its first resident forward,
    /// costing two model swaps for no resident token. Charging the request's
    /// context on top of the startup headroom rejects that case up front.
    /// It is a necessary condition, not a guarantee: the resident forward's
    /// learned transient workspace can still demote the request, which the
    /// worker's one-swap-per-direction bound contains.
    #[must_use]
    pub fn complete_residency_admits_request_context(
        &self,
        effective_ceiling_bytes: u64,
        total_context_token_count: usize,
    ) -> bool {
        let total_context_token_count =
            u64::try_from(total_context_token_count).unwrap_or(u64::MAX);
        let request_context_reserve_bytes = self
            .sequence_state_bytes_per_token
            .saturating_mul(total_context_token_count);
        matches!(
            CompleteResidencyRequirements {
                current_active_memory_bytes: self.model_core_payload_bytes,
                retained_paged_expert_payload_bytes: 0,
                complete_expert_payload_bytes: self.complete_expert_payload_bytes,
                required_headroom_bytes: self
                    .startup_complete_residency_headroom_bytes
                    .saturating_add(request_context_reserve_bytes),
                active_memory_ceiling_bytes: effective_ceiling_bytes,
            }
            .decide(),
            CompleteResidencyDecision::Admit { .. }
        )
    }
}
