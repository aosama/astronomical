//! Admission policy: "may this operation begin?"
//!
//! Each module here answers a go/no-go question for one operation class and
//! emits a typed decision. Inputs arrive as exact caller-supplied facts
//! (`*Requirements`) or runtime-sampled facts (`*Observation`); execution
//! owners enact the decision and never re-derive it. See the package
//! documentation for the decision-shape rule.

mod allocation;
mod complete_residency;
mod context;
#[cfg(feature = "direct-mlx")]
mod context_logging;
mod mtp;
mod mtp_draft_depth;
mod rotating;
mod speculative_prefill;

pub use allocation::{
    AllocationAdmissionDecision, AllocationAdmissionObservation,
    retained_expert_payload_capacity_bytes,
};
pub use complete_residency::{CompleteResidencyDecision, CompleteResidencyRequirements};
pub use context::{
    ContextAdmissionRequirements, combined_persistent_growth_bytes,
    persistent_context_restore_workspace_bytes, request_context_temporary_workspace_bytes,
    safe_minimum_active_memory_ceiling_bytes,
    seated_complete_expert_request_peak_active_memory_bytes,
    seated_complete_expert_request_temporary_workspace_bytes,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use context_logging::{
    log_context_admission_projection, log_generation_context_workspace_reservation,
};
pub use mtp::{
    MtpAdmission, MtpDepthDowngradeReason, MtpDepthSelection, MtpMemoryCandidate,
    MtpMemoryProjection, MtpMemoryProjectionError,
};
pub use mtp_draft_depth::{MtpDraftDepth, MtpDraftDepthError};
pub use rotating::{
    RotatingAdmissionError, rotating_committed_token_count, rotating_prefill_transient_token_count,
};
pub use speculative_prefill::SpeculativePrefillAdmission;
