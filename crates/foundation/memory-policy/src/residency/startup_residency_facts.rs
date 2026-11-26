//! The startup residency arithmetic for a dense text model.
//!
//! Startup works out whether the MLX memory limit holds a whole model, which
//! decides the placement reported to the worker and the prompt chunk size.
//! A dense model has no experts, so its facts are its text weights alone;
//! they go through the same complete-residency decision a sparse model's
//! geometry does, so both kinds are placed by one rule.

use crate::admission::complete_residency::{
    CompleteResidencyDecision, CompleteResidencyRequirements,
};

/// Startup residency selection facts for one model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingStartupResidencyFacts {
    /// Non-expert resident core payload measured from the validated artifact.
    pub model_core_payload_bytes: u64,
    /// Bytes required when every sparse expert is fully resident.
    pub complete_expert_payload_bytes: u64,
    /// Context and activation headroom the startup selection requires above
    /// complete residency (the load-time gate-up fusion transient).
    pub startup_complete_residency_headroom_bytes: u64,
}

impl StreamingStartupResidencyFacts {
    /// Startup facts for a dense model: the text weights it binds are its
    /// whole core and it has no experts. Dense selection compares those
    /// weights with the limit macOS granted or the user set and nothing
    /// else, so it carries no headroom of its own.
    #[must_use]
    pub const fn for_dense_text_weights(dense_text_weight_payload_bytes: u64) -> Self {
        Self {
            model_core_payload_bytes: dense_text_weight_payload_bytes,
            complete_expert_payload_bytes: 0,
            startup_complete_residency_headroom_bytes: 0,
        }
    }

    /// The complete-residency decision startup makes under
    /// `active_memory_ceiling_bytes`.
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
}
