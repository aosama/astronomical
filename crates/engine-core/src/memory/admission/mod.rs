//! The pure admission decisions (allocation, complete residency, context
//! sizes, retained-prefix, rotating) live in
//! `memory-policy::admission` and are re-exported here so
//! consumer paths stay stable.

pub use memory_policy::admission::{
    AllocationAdmissionDecision, AllocationAdmissionObservation, CompleteResidencyDecision,
    CompleteResidencyRequirements, RetainedPrefixRetentionDecision,
    RetainedPrefixRetentionRequirements, RotatingAdmissionError, VISIBLE_ANSWER_RESERVE_TOKENS,
    bounded_output_reservation_token_count, persistent_context_restore_workspace_bytes,
    rotating_committed_token_count, rotating_prefill_transient_token_count,
    safe_minimum_active_memory_ceiling_bytes,
};
