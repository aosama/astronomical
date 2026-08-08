//! Isolated Qwen3.5 multi-token prediction ownership.
//!
//! This module is the removable boundary for optional predictor weights,
//! predictor request state, and predictor-specific execution.

mod artifact;
#[cfg(feature = "direct-mlx")]
mod decode;
mod dense_tensor_spec;
#[cfg(feature = "direct-mlx")]
mod forward;
#[cfg(feature = "direct-mlx")]
mod injected_input;
#[cfg(feature = "direct-mlx")]
mod model;
mod moe_tensor_spec;
#[cfg(feature = "direct-mlx")]
mod prefill;
#[cfg(feature = "direct-mlx")]
mod request_state;
#[cfg(feature = "direct-mlx")]
mod runtime;
#[cfg(feature = "direct-mlx")]
mod target_verification;
mod tensor_namespace;

pub use artifact::Qwen3_5MtpArtifactCapability;
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5) use decode::{
    Qwen3_5PredictionAcceptanceOutcome, attempt_prediction_proposal_and_verification,
    disable_prediction_after_memory_admission_failure,
    forward_initial_target_token_with_prediction_state,
    forward_next_target_token_with_prediction_state, prediction_verification_is_eligible,
    projected_verification_window_memory_growth_bytes, take_queued_prediction_token,
    verification_window_workspace_bytes,
};
#[cfg(feature = "direct-mlx")]
pub use decode::{
    qwen3_5_depth_one_mtp_window_fits, qwen3_5_mtp_verification_may_cross_thinking_budget,
};
#[cfg(feature = "direct-mlx")]
pub use forward::Qwen3_5MtpForwardOutput;
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5) use injected_input::restore_queued_prediction_prefix_before_injection;
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5) use injected_input::{
    disable_prediction_after_optional_injection_failure, forward_final_injected_prediction_token,
    projected_injected_prediction_growth_bytes, reseed_prediction_after_injected_prefix,
    reset_prediction_after_injection,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use model::{Qwen3_5MtpWeights, bind_optional_weights, materialize_optional_weights};
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5) use prefill::{
    execute_terminal_optional_history_capture_with_performance_attribution,
    initialize_prompt_history_from_token_ids_with_performance_attribution,
    record_prompt_history_initialization_fallback, record_terminal_history_token_count,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use request_state::{
    AcceptedMultiTokenPredictionDraftRollback, MultiTokenPredictionRequestAllocationCheckpoint,
    Qwen3_5MultiTokenPredictionRequest, create_optional_prediction_session,
};
#[cfg(feature = "direct-mlx")]
pub use request_state::{
    Qwen3_5MtpRequestState, Qwen3_5MtpRequestStateAllocationCheckpoint, Qwen3_5MtpUnavailableReason,
};
#[cfg(feature = "direct-mlx")]
pub use runtime::{Qwen3_5MtpRuntimeState, qwen3_5_mtp_runtime_state_after_load};
pub use tensor_namespace::{qwen3_5_mtp_tensor_names, qwen3_5_mtp_tensor_profiles};
