//! Converts validated source selection into one optional standalone MLX attachment.

use crate::qwen3_5::artifacts::Qwen3_5StandaloneMtpBindingParts;
use crate::qwen3_5::model::{attach_standalone_mtp_weights, disable_optional_mtp_weights};
use crate::qwen3_5::multi_token_prediction::Qwen3_5MtpTargetOnlyReason;
use crate::qwen3_5::{Qwen3_5Model, Qwen3_5MtpArtifactCapability, Qwen3_5MtpSourceSelection};
use crate::{InferenceEngineError, PerformanceAttribution, PerformanceOperation};

use super::fatal_engine_error;

pub(super) struct PreparedMtpSource {
    pub(super) artifact_capability: Qwen3_5MtpArtifactCapability,
    pub(super) should_bind_target_local: bool,
    pub(super) standalone_binding_parts: Option<Qwen3_5StandaloneMtpBindingParts>,
}

pub(super) struct MtpDrafterAttribution {
    pub(super) model_id: Option<String>,
    pub(super) model_revision: Option<String>,
    pub(super) storage_fingerprint: Option<String>,
}

pub(super) fn mtp_drafter_attribution(
    selection: &Qwen3_5MtpSourceSelection,
) -> MtpDrafterAttribution {
    match selection {
        Qwen3_5MtpSourceSelection::Standalone { artifact, .. } => MtpDrafterAttribution {
            model_id: Some(artifact.model_id().to_owned()),
            model_revision: Some(artifact.discovered_revision().to_owned()),
            storage_fingerprint: Some(artifact.storage_fingerprint().to_owned()),
        },
        Qwen3_5MtpSourceSelection::TargetOnly {
            drafter_model_id,
            drafter_model_revision,
            drafter_storage_fingerprint,
            ..
        } => MtpDrafterAttribution {
            model_id: Some(drafter_model_id.clone()),
            model_revision: drafter_model_revision.clone(),
            storage_fingerprint: drafter_storage_fingerprint.clone(),
        },
        Qwen3_5MtpSourceSelection::TargetLocal => MtpDrafterAttribution {
            model_id: None,
            model_revision: None,
            storage_fingerprint: None,
        },
    }
}

pub(super) fn prepare_mtp_source(
    selection: Qwen3_5MtpSourceSelection,
    target_local_capability: Qwen3_5MtpArtifactCapability,
) -> Result<PreparedMtpSource, InferenceEngineError> {
    match selection {
        Qwen3_5MtpSourceSelection::TargetLocal => Ok(PreparedMtpSource {
            artifact_capability: target_local_capability,
            should_bind_target_local: true,
            standalone_binding_parts: None,
        }),
        Qwen3_5MtpSourceSelection::Standalone {
            artifact,
            compatibility,
        } => {
            let artifact_capability = Qwen3_5MtpArtifactCapability::from_standalone(
                compatibility.maximum_draft_depth,
                artifact.tensor_profiles().len(),
            )
            .map_err(|_| fatal_engine_error("standalone MTP depth is invalid"))?;
            Ok(PreparedMtpSource {
                artifact_capability,
                should_bind_target_local: false,
                standalone_binding_parts: artifact.into_binding_parts().ok(),
            })
        }
        Qwen3_5MtpSourceSelection::TargetOnly {
            reason: unavailable_reason,
            ..
        } => Ok(PreparedMtpSource {
            artifact_capability: Qwen3_5MtpArtifactCapability::target_only(
                match unavailable_reason {
                    crate::qwen3_5::Qwen3_5MtpSourceUnavailableReason::ConfiguredDrafterNotDiscovered => {
                        Qwen3_5MtpTargetOnlyReason::StandaloneDrafterNotDiscovered
                    }
                    crate::qwen3_5::Qwen3_5MtpSourceUnavailableReason::StandaloneArtifactInvalid => {
                        Qwen3_5MtpTargetOnlyReason::StandaloneDrafterInvalid
                    }
                    crate::qwen3_5::Qwen3_5MtpSourceUnavailableReason::PairingIncompatible => {
                        Qwen3_5MtpTargetOnlyReason::StandalonePairingIncompatible
                    }
                },
            ),
            should_bind_target_local: false,
            standalone_binding_parts: None,
        }),
    }
}

pub(super) fn attach_prepared_standalone_mtp(
    model: &mut Qwen3_5Model,
    standalone_binding_parts: Option<Qwen3_5StandaloneMtpBindingParts>,
    artifact_capability: &mut Qwen3_5MtpArtifactCapability,
    performance_attribution: &mut PerformanceAttribution,
) -> bool {
    let Some(standalone_binding_parts) = standalone_binding_parts else {
        return false;
    };
    performance_attribution.measure_operation(
        PerformanceOperation::StandaloneMtpTensorBinding,
        |_performance_attribution| match attach_standalone_mtp_weights(
            model,
            standalone_binding_parts,
        ) {
            Ok(()) => true,
            Err(standalone_binding_error) => {
                tracing::warn!(
                    error = %standalone_binding_error,
                    "standalone MTP binding failed; serving target-only"
                );
                *artifact_capability = Qwen3_5MtpArtifactCapability::target_only(
                    Qwen3_5MtpTargetOnlyReason::StandaloneBindingFailed,
                );
                false
            }
        },
    )
}

/// Converts optional materialization failure into an explicit target-only runtime.
pub(super) fn handle_optional_mtp_materialization_failure(
    model: &mut Qwen3_5Model,
    artifact_capability: &mut Qwen3_5MtpArtifactCapability,
    materialization_error: &crate::qwen3_5::Qwen3_5ExecutionError,
) {
    tracing::warn!(
        error = %materialization_error,
        "optional MTP weight materialization failed; serving target-only"
    );
    disable_optional_mtp_weights(model);
    *artifact_capability = Qwen3_5MtpArtifactCapability::target_only(
        Qwen3_5MtpTargetOnlyReason::OptionalWeightMaterializationFailed,
    );
    if let Err(mlx_allocator_cleanup_error) = model
        .runtime()
        .synchronize_gpu_stream_and_clear_allocator_cache()
    {
        tracing::warn!(
            error = %mlx_allocator_cleanup_error,
            "failed to reclaim allocator memory after optional MTP initialization failure"
        );
    }
}

pub(super) fn materialize_prepared_mtp(
    model: &mut Qwen3_5Model,
    should_materialize_mtp_weights: bool,
    standalone_mtp_was_attached: bool,
    artifact_capability: &mut Qwen3_5MtpArtifactCapability,
    performance_attribution: &mut PerformanceAttribution,
) {
    if !should_materialize_mtp_weights || !model.mtp_weights() {
        return;
    }
    let materialization_operation = if standalone_mtp_was_attached {
        PerformanceOperation::StandaloneMtpMaterializationSynchronizationWait
    } else {
        PerformanceOperation::ResidentWeightMaterializationSynchronizationWait
    };
    if let Err(materialization_error) = performance_attribution.measure_operation(
        materialization_operation,
        |_performance_attribution| {
            crate::qwen3_5::multi_token_prediction::materialize_optional_weights(model)
        },
    ) {
        handle_optional_mtp_materialization_failure(
            model,
            artifact_capability,
            &materialization_error,
        );
    }
}
