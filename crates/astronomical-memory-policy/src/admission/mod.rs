//! Admission policy: "may this operation begin?"
//!
//! Each module here answers a go/no-go question for one operation class and
//! emits a typed decision. Inputs arrive as exact caller-supplied facts
//! (`*Requirements`) or runtime-sampled facts (`*Observation`); execution
//! owners enact the decision and never re-derive it. See the package
//! documentation for the decision-shape rule. The direct-MLX admission
//! logging owner stays beside the bindings in `astronomical-model-serving`.

pub mod allocation;
pub mod complete_residency;
pub mod context;
pub mod retained_prefix;
pub mod rotating;

#[cfg(test)]
mod context_boundary_tests;
#[cfg(test)]
mod expert_memory_admission_tests;
#[cfg(test)]
mod retained_prefix_policy_tests;
#[cfg(test)]
mod rotating_admission_tests;

pub use allocation::{
    AllocationAdmissionDecision, AllocationAdmissionObservation,
    retained_expert_payload_capacity_bytes,
};
pub use complete_residency::{CompleteResidencyDecision, CompleteResidencyRequirements};
pub use context::{
    ContextAdmissionRequirements, VISIBLE_ANSWER_RESERVE_TOKENS,
    bounded_output_reservation_token_count, combined_persistent_growth_bytes,
    persistent_context_restore_workspace_bytes, request_context_temporary_workspace_bytes,
    safe_minimum_active_memory_ceiling_bytes,
    seated_complete_expert_request_peak_active_memory_bytes,
    seated_complete_expert_request_temporary_workspace_bytes,
};
pub use retained_prefix::{RetainedPrefixRetentionDecision, RetainedPrefixRetentionRequirements};
pub use rotating::{
    RotatingAdmissionError, rotating_committed_token_count, rotating_prefill_transient_token_count,
};
