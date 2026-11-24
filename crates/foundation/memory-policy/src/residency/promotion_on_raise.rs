//! Streaming-engine promotion gate for a raised MLX ceiling.
//!
//! A streaming worker is spawned only when its startup ceiling could not
//! admit complete residency: every expert of a sparse model, or every text
//! weight of a dense model. When the user later raises the ceiling
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
    /// Startup facts for a dense model: the text weights it binds are its
    /// whole core and it has no experts. Dense selection compares those
    /// weights with the limit macOS granted or the user set and nothing
    /// else, so it carries no headroom of its own.
    #[must_use]
    pub const fn for_dense_text_weights(
        dense_text_weight_payload_bytes: u64,
        sequence_state_bytes_per_token: u64,
    ) -> Self {
        Self {
            model_core_payload_bytes: dense_text_weight_payload_bytes,
            complete_expert_payload_bytes: 0,
            startup_complete_residency_headroom_bytes: 0,
            sequence_state_bytes_per_token,
        }
    }

    /// The complete-residency decision startup selection makes under
    /// `active_memory_ceiling_bytes`. Startup selection and the raise check
    /// both call it, so a promotion demand cannot disagree with the engine a
    /// replacement worker would choose.
    #[must_use]
    pub fn startup_complete_residency_decision(
        &self,
        active_memory_ceiling_bytes: u64,
    ) -> CompleteResidencyDecision {
        CompleteResidencyRequirements {
            current_active_memory_bytes: self.model_core_payload_bytes,
            retained_paged_expert_payload_bytes: 0,
            complete_expert_payload_bytes: self.complete_expert_payload_bytes,
            required_headroom_bytes: self.startup_complete_residency_headroom_bytes,
            active_memory_ceiling_bytes,
        }
        .decide()
    }

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
            self.startup_complete_residency_decision(raised_ceiling_bytes),
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
