//! Bounded model-load logging for the resolved MTP runtime state.

use super::Qwen3_5MtpRuntimeState;

pub(super) fn log_mtp_runtime_state(
    model_id: Option<&str>,
    runtime_state: Qwen3_5MtpRuntimeState,
    unavailable_reason: Option<&str>,
) {
    let model_id = model_id.unwrap_or("unknown");
    match runtime_state {
        Qwen3_5MtpRuntimeState::Disabled => {}
        Qwen3_5MtpRuntimeState::TargetOnly => tracing::info!(
            model_id,
            "MTP is enabled but the selected model has no MTP inventory; serving target-only"
        ),
        Qwen3_5MtpRuntimeState::Active => {
            tracing::info!(model_id, "native MTP is active for this model");
        }
        Qwen3_5MtpRuntimeState::Unavailable => tracing::warn!(
            model_id,
            mtp_unavailable_reason =
                unavailable_reason.unwrap_or("unknown MTP initialization failure"),
            "MTP is enabled but unavailable; serving target-only"
        ),
    }
}
