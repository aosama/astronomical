//! Multi-token prediction runtime-state reporting.

use super::{Qwen3_5MtpArtifactCapability, Qwen3_5MtpUnavailableReason};

/// Runtime execution state of native MTP, distinct from the user preference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5MtpRuntimeState {
    /// The user preference is disabled.
    Disabled,
    /// The preference is enabled but the artifact has no compatible inventory.
    TargetOnly,
    /// The compatible head was loaded and native MTP is available.
    Active,
    /// The artifact advertised MTP but optional initialization failed.
    Unavailable,
}

/// Derives the public MTP state after optional head initialization.
#[doc(hidden)]
#[must_use]
pub fn qwen3_5_mtp_runtime_state_after_load(
    mtp_enabled: bool,
    mtp_artifact_capability: &Qwen3_5MtpArtifactCapability,
    model_has_mtp_weights: bool,
) -> (Qwen3_5MtpRuntimeState, Option<String>) {
    if !mtp_enabled {
        return (Qwen3_5MtpRuntimeState::Disabled, None);
    }
    if model_has_mtp_weights {
        return (Qwen3_5MtpRuntimeState::Active, None);
    }
    match mtp_artifact_capability {
        Qwen3_5MtpArtifactCapability::TargetOnly => (Qwen3_5MtpRuntimeState::TargetOnly, None),
        Qwen3_5MtpArtifactCapability::MtpCapable { .. } => (
            Qwen3_5MtpRuntimeState::Unavailable,
            Some(Qwen3_5MtpUnavailableReason::NoCompatibleHead.to_string()),
        ),
    }
}
