//! Multi-token prediction runtime-state reporting.

use astronomical_ipc_protocol::{MtpDepthResolutionReason, MtpDepthStatus};

use super::{MtpDraftDepth, Qwen3_5MtpArtifactCapability, Qwen3_5MtpUnavailableReason};

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
    let (runtime_state, unavailable_reason, _) = qwen3_5_mtp_runtime_configuration_after_load(
        mtp_enabled,
        None,
        mtp_artifact_capability,
        model_has_mtp_weights,
    );
    (runtime_state, unavailable_reason)
}

/// Resolves configured, artifact, requested, and currently executable MTP depth.
#[must_use]
pub fn qwen3_5_mtp_runtime_configuration_after_load(
    mtp_enabled: bool,
    configured_draft_depth: Option<MtpDraftDepth>,
    mtp_artifact_capability: &Qwen3_5MtpArtifactCapability,
    model_has_mtp_weights: bool,
) -> (Qwen3_5MtpRuntimeState, Option<String>, MtpDepthStatus) {
    let artifact_maximum_draft_depth = mtp_artifact_capability.artifact_maximum_draft_depth();
    let artifact_default_draft_depth = mtp_artifact_capability.artifact_default_draft_depth();
    let mut depth_status = MtpDepthStatus {
        configured_draft_depth: configured_draft_depth.map(MtpDraftDepth::get),
        artifact_maximum_draft_depth: artifact_maximum_draft_depth.map(MtpDraftDepth::get),
        artifact_default_draft_depth: artifact_default_draft_depth.map(MtpDraftDepth::get),
        resolved_requested_draft_depth: None,
        capped_draft_depth: None,
        effective_execution_draft_depth: None,
        resolution_reason: None,
    };
    if !mtp_enabled {
        return (Qwen3_5MtpRuntimeState::Disabled, None, depth_status);
    }
    match mtp_artifact_capability {
        Qwen3_5MtpArtifactCapability::TargetOnly { reason } => (
            Qwen3_5MtpRuntimeState::TargetOnly,
            Some(reason.to_string()),
            depth_status,
        ),
        Qwen3_5MtpArtifactCapability::MtpCapable { .. } => {
            let resolved_draft_depth = configured_draft_depth
                .or(artifact_default_draft_depth)
                // Artifact maximum expresses compatibility, not a measured optimum. Depth one
                // remains the automatic production baseline until metadata explicitly selects a
                // deeper depth that has passed the representative release gate.
                .unwrap_or(MtpDraftDepth::DEPTH_ONE);
            depth_status.resolved_requested_draft_depth = Some(resolved_draft_depth.get());
            let capped_draft_depth = artifact_maximum_draft_depth
                .map_or(resolved_draft_depth, |maximum_depth| {
                    resolved_draft_depth.min(maximum_depth)
                });
            depth_status.capped_draft_depth = Some(capped_draft_depth.get());
            depth_status.resolution_reason = if capped_draft_depth < resolved_draft_depth {
                Some(MtpDepthResolutionReason::ConfiguredDepthClampedToArtifactMaximum)
            } else if configured_draft_depth.is_some()
                && artifact_maximum_draft_depth.is_none()
                && resolved_draft_depth > MtpDraftDepth::DEPTH_ONE
            {
                Some(MtpDepthResolutionReason::ConfiguredDepthExceedsAutomaticGuidance)
            } else {
                None
            };
            if !model_has_mtp_weights {
                return (
                    Qwen3_5MtpRuntimeState::Unavailable,
                    Some(Qwen3_5MtpUnavailableReason::NoCompatibleHead.to_string()),
                    depth_status,
                );
            }
            depth_status.effective_execution_draft_depth = Some(capped_draft_depth.get());
            (Qwen3_5MtpRuntimeState::Active, None, depth_status)
        }
    }
}
