mod artifact;
mod artifact_helpers;
mod artifact_inventory;
mod mtp_contract;
mod mtp_pairing_compatibility;
mod mtp_storage_fingerprint;
mod shard_index;
mod sidecar_declaration;
mod standalone_mtp_artifact;
mod standalone_mtp_config;
mod standalone_mtp_namespace;
pub(crate) mod tensor_spec;
mod validated_artifact;
mod vision_validation;

pub(crate) use super::configuration::{
    Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture,
};
pub(crate) use super::quantizations;
pub(crate) use super::quantizations::optiq::{OptiQMetadata, OptiQMetadataError};
pub(crate) use super::vision::{Qwen3_5VisionConfig, vision_tensor_spec};
pub use crate::qwen3_5::multi_token_prediction::{
    Qwen3_5MtpArtifactCapability, Qwen3_5MtpTargetOnlyReason,
};
pub use crate::qwen3_5::multi_token_prediction::{
    qwen3_5_mtp_tensor_names, qwen3_5_mtp_tensor_profiles,
};
pub use artifact::{Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator};
pub use mtp_contract::{MAXIMUM_MTPLX_RUNTIME_BYTES, Qwen3_5MtpContract, Qwen3_5MtpContractError};
pub use mtp_pairing_compatibility::{
    Qwen3_5MtpPairingCompatibility, Qwen3_5MtpPairingCompatibilityError,
    compare_qwen3_5_mtp_pairing_contracts,
};
pub use shard_index::{MAXIMUM_INDEX_BYTES, Qwen3_5ArtifactError, Qwen3_5ShardIndex};
pub use sidecar_declaration::{
    Qwen3_5MtpSidecarDeclaration, Qwen3_5MtpSidecarDeclarationError,
    Qwen3_5MtpSidecarValidationOutcome, validate_qwen3_5_mtp_sidecar_for_tests,
};
pub(crate) use standalone_mtp_artifact::Qwen3_5StandaloneMtpBindingParts;
pub use standalone_mtp_artifact::{
    Qwen3_5StandaloneMtpArtifactValidationError, Qwen3_5StandaloneMtpArtifactValidator,
    ValidatedQwen3_5StandaloneMtpArtifact,
};
pub use standalone_mtp_config::{Qwen3_5StandaloneMtpConfig, Qwen3_5StandaloneMtpConfigError};
pub use standalone_mtp_namespace::{
    StandaloneMtpNamespaceError, normalize_qwen3_5_standalone_mtp_tensor_name,
};
pub use tensor_spec::{
    qwen3_5_language_tensor_profiles, qwen3_5_resident_language_tensor_profiles,
};
pub use validated_artifact::ValidatedQwen3_5Artifact;
