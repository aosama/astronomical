//! Isolated Qwen3.5 multi-token prediction ownership.
//!
//! This module is the removable boundary for optional predictor weights,
//! predictor request state, and predictor-specific execution.

#[cfg(feature = "direct-mlx")]
mod accepted_prefix_commit;
mod artifact;
#[cfg(feature = "direct-mlx")]
mod decode;
mod dense_tensor_spec;
mod depth;
#[cfg(feature = "direct-mlx")]
mod forward;
#[cfg(feature = "direct-mlx")]
mod injected_input;
mod memory_admission;
#[cfg(feature = "direct-mlx")]
mod model;
mod moe_tensor_spec;
#[cfg(feature = "direct-mlx")]
mod prefill;
#[cfg(feature = "direct-mlx")]
mod proposal_chain;
mod request_eligibility;
#[cfg(feature = "direct-mlx")]
mod request_state;
#[cfg(feature = "direct-mlx")]
mod runtime;
mod source_selection;
#[cfg(feature = "direct-mlx")]
mod target_verification;
mod tensor_namespace;
mod verification_decision;
#[cfg(feature = "direct-mlx")]
mod verified_emission_queue;

pub use artifact::{Qwen3_5MtpArtifactCapability, Qwen3_5MtpTargetOnlyReason};
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5) use decode::{
    attempt_prediction_proposal_and_verification,
    disable_prediction_after_memory_admission_failure, effective_prediction_depth,
    forward_initial_target_token_with_prediction_state,
    forward_next_target_token_with_prediction_state,
    projected_verification_window_memory_growth_bytes, take_queued_prediction_token,
    verification_boundary_snapshot_bytes, verification_transient_array_bytes,
};
#[cfg(feature = "direct-mlx")]
pub use decode::{
    qwen3_5_depth_one_mtp_window_fits, qwen3_5_mtp_verification_may_cross_thinking_budget,
};
pub use depth::{MtpDraftDepth, MtpDraftDepthError};
#[cfg(feature = "direct-mlx")]
pub use forward::Qwen3_5MtpForwardOutput;
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5) use injected_input::restore_queued_prediction_prefix_before_injection;
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5) use injected_input::{
    disable_prediction_after_optional_injection_failure, projected_injected_prediction_growth_bytes,
};
pub use memory_admission::{
    MtpDepthDowngradeReason, MtpMemoryAdmission, MtpMemoryCandidate, MtpMemoryProjection,
    MtpMemoryProjectionError, qwen3_5_mtp_memory_admission,
    qwen3_5_mtp_verification_transient_array_bytes,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use model::{Qwen3_5MtpWeights, bind_optional_weights, materialize_optional_weights};
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5) use prefill::{
    execute_terminal_optional_history_capture_with_performance_attribution,
    record_prompt_history_initialization_fallback, record_terminal_history_token_count,
};
pub use request_eligibility::{
    Qwen3_5MtpRequestEligibility, Qwen3_5MtpRequestEligibilityInputs,
    qwen3_5_mtp_request_eligibility,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use request_state::{
    MultiTokenPredictionRequestAllocationCheckpoint, Qwen3_5MultiTokenPredictionRequest,
    create_optional_prediction_session,
};
#[cfg(feature = "direct-mlx")]
pub use request_state::{
    Qwen3_5MtpRequestState, Qwen3_5MtpRequestStateAllocationCheckpoint, Qwen3_5MtpUnavailableReason,
};
#[cfg(feature = "direct-mlx")]
pub use runtime::{
    Qwen3_5MtpRuntimeState, qwen3_5_mtp_runtime_configuration_after_load,
    qwen3_5_mtp_runtime_state_after_load,
};
pub use source_selection::{Qwen3_5MtpSourceSelection, Qwen3_5MtpSourceUnavailableReason};
pub use tensor_namespace::{qwen3_5_mtp_tensor_names, qwen3_5_mtp_tensor_profiles};
pub use verification_decision::{
    MtpVerificationDecision, MtpVerificationDecisionError,
    predictor_history_requires_verified_hidden_replay,
    qwen3_5_mtp_effective_depth_and_reason_for_windows, qwen3_5_mtp_effective_depth_for_windows,
    qwen3_5_mtp_verification_decision,
};
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use verified_emission_queue::{VerifiedEmissionQueue, VerifiedTargetFrontier};
