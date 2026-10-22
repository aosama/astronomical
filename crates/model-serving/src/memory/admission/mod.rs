//! Admission owner in the serving crate: the direct-MLX admission logging
//! stays beside the bindings. The pure admission decisions (allocation,
//! complete residency, context, retained-prefix, rotating) live in
//! `astronomical-memory-policy::admission` and are re-exported here so
//! consumer paths stay stable.

#[cfg(feature = "direct-mlx")]
mod context_logging;

#[cfg(feature = "direct-mlx")]
pub(crate) use context_logging::{
    log_context_admission_projection, log_generation_context_workspace_reservation,
};
pub use astronomical_memory_policy::admission::{
    AllocationAdmissionDecision, AllocationAdmissionObservation,
    CompleteResidencyDecision, CompleteResidencyRequirements, ContextAdmissionRequirements,
    RetainedPrefixRetentionDecision, RetainedPrefixRetentionRequirements, RotatingAdmissionError,
    VISIBLE_ANSWER_RESERVE_TOKENS, bounded_output_reservation_token_count,
    combined_persistent_growth_bytes, persistent_context_restore_workspace_bytes,
    request_context_temporary_workspace_bytes, retained_expert_payload_capacity_bytes,
    rotating_committed_token_count, rotating_prefill_transient_token_count,
    safe_minimum_active_memory_ceiling_bytes,
    seated_complete_expert_request_peak_active_memory_bytes,
    seated_complete_expert_request_temporary_workspace_bytes,
};
