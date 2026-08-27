use std::collections::BTreeSet;
use std::fmt;

use crate::qwen3_5::{Qwen3_5Config, Qwen3_5MtpContract, Qwen3_5ShardIndex};

use super::{MtpDraftDepth, tensor_namespace::qwen3_5_mtp_tensor_names};

/// Bounded artifact reason that keeps optional MTP target-only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen3_5MtpTargetOnlyReason {
    NoTensorInventory,
    UnsupportedStoredLayerCount,
    IncompleteTensorInventory,
    UnexpectedTensorInventory,
    SidecarUnavailable,
    CanonicalTensorCollision,
    TensorValidationFailed,
    /// The declared MTP sidecar could not be validated; the contained diagnostic is the
    /// human-readable cause (missing tensor, dtype/shape mismatch, target collision).
    SidecarValidationFailed(String),
    ContractMalformed,
    ContractRuntimeDocumentTooLarge,
    ContractFieldDisagreement,
    ContractIncompatible,
}

impl fmt::Display for Qwen3_5MtpTargetOnlyReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTensorInventory => formatter.write_str("model has no MTP tensor inventory"),
            Self::UnsupportedStoredLayerCount => {
                formatter.write_str("model does not contain exactly one supported MTP layer")
            }
            Self::IncompleteTensorInventory => {
                formatter.write_str("MTP tensor inventory is incomplete")
            }
            Self::UnexpectedTensorInventory => {
                formatter.write_str("MTP tensor inventory contains unsupported tensors")
            }
            Self::SidecarUnavailable => formatter.write_str("declared MTP sidecar is unavailable"),
            Self::CanonicalTensorCollision => {
                formatter.write_str("MTP tensors have conflicting canonical ownership")
            }
            Self::TensorValidationFailed => formatter.write_str("MTP tensor validation failed"),
            Self::SidecarValidationFailed(diagnostic) => formatter.write_str(diagnostic),
            Self::ContractMalformed => formatter.write_str("optional MTP contract is malformed"),
            Self::ContractRuntimeDocumentTooLarge => {
                formatter.write_str("optional MTP runtime metadata exceeds 64 KB")
            }
            Self::ContractFieldDisagreement => {
                formatter.write_str("duplicated optional MTP contract fields disagree")
            }
            Self::ContractIncompatible => {
                formatter.write_str("optional MTP contract uses unsupported execution semantics")
            }
        }
    }
}

/// Data-driven MTP capability discovered from the validated artifact inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen3_5MtpArtifactCapability {
    TargetOnly {
        reason: Qwen3_5MtpTargetOnlyReason,
    },
    MtpCapable {
        stored_mtp_layer_count: usize,
        artifact_maximum_draft_depth: Option<MtpDraftDepth>,
        artifact_default_draft_depth: Option<MtpDraftDepth>,
        mtp_tensor_count: usize,
    },
}

impl Qwen3_5MtpArtifactCapability {
    const SUPPORTED_STORED_MTP_LAYER_COUNT: usize = 1;

    #[must_use]
    pub const fn target_only(reason: Qwen3_5MtpTargetOnlyReason) -> Self {
        Self::TargetOnly { reason }
    }

    /// Classifies MTP capability from shard-index tensor inventory only.
    #[must_use]
    pub fn from_shard_index(
        qwen3_5_config: &Qwen3_5Config,
        shard_index: &Qwen3_5ShardIndex,
    ) -> Self {
        let actual_mtp_tensor_names = shard_index
            .mtp_tensor_name_to_shard_file_name()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        Self::from_canonical_tensor_names(qwen3_5_config, actual_mtp_tensor_names, None)
    }

    /// Classifies capability from canonical names independent of physical storage.
    #[must_use]
    #[doc(hidden)]
    pub fn from_canonical_tensor_names(
        qwen3_5_config: &Qwen3_5Config,
        actual_mtp_tensor_names: BTreeSet<String>,
        optional_contract: Option<&Qwen3_5MtpContract>,
    ) -> Self {
        if actual_mtp_tensor_names.is_empty() {
            return Self::target_only(Qwen3_5MtpTargetOnlyReason::NoTensorInventory);
        }
        if qwen3_5_config.mtp_layer_count() != Self::SUPPORTED_STORED_MTP_LAYER_COUNT as u32 {
            return Self::target_only(Qwen3_5MtpTargetOnlyReason::UnsupportedStoredLayerCount);
        }
        let expected_mtp_tensor_names = qwen3_5_mtp_tensor_names(qwen3_5_config);
        if actual_mtp_tensor_names
            .difference(&expected_mtp_tensor_names)
            .next()
            .is_some()
        {
            return Self::target_only(Qwen3_5MtpTargetOnlyReason::UnexpectedTensorInventory);
        }
        if expected_mtp_tensor_names
            .difference(&actual_mtp_tensor_names)
            .next()
            .is_some()
        {
            return Self::target_only(Qwen3_5MtpTargetOnlyReason::IncompleteTensorInventory);
        }
        let artifact_maximum_draft_depth =
            optional_contract.and_then(Qwen3_5MtpContract::artifact_maximum_depth);
        let artifact_default_draft_depth = optional_contract
            .and_then(Qwen3_5MtpContract::artifact_default_depth)
            .filter(|default_depth| {
                artifact_maximum_draft_depth
                    .is_none_or(|maximum_depth| *default_depth <= maximum_depth)
            });
        Self::MtpCapable {
            stored_mtp_layer_count: Self::SUPPORTED_STORED_MTP_LAYER_COUNT,
            artifact_maximum_draft_depth,
            artifact_default_draft_depth,
            mtp_tensor_count: actual_mtp_tensor_names.len(),
        }
    }

    #[must_use]
    pub const fn is_mtp_capable(&self) -> bool {
        matches!(self, Self::MtpCapable { .. })
    }

    #[must_use]
    pub fn target_only_reason(&self) -> Option<Qwen3_5MtpTargetOnlyReason> {
        match self {
            Self::TargetOnly { reason } => Some(reason.clone()),
            Self::MtpCapable { .. } => None,
        }
    }

    #[must_use]
    pub const fn artifact_maximum_draft_depth(&self) -> Option<MtpDraftDepth> {
        match self {
            Self::MtpCapable {
                artifact_maximum_draft_depth,
                ..
            } => *artifact_maximum_draft_depth,
            Self::TargetOnly { .. } => None,
        }
    }

    #[must_use]
    pub const fn artifact_default_draft_depth(&self) -> Option<MtpDraftDepth> {
        match self {
            Self::MtpCapable {
                artifact_default_draft_depth,
                ..
            } => *artifact_default_draft_depth,
            Self::TargetOnly { .. } => None,
        }
    }
}
