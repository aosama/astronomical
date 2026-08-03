mod artifact;
mod mtp_artifact_capability;
mod mtp_tensor_namespace;
mod shard_index;
pub(crate) mod tensor_spec;
mod vision_validation;

pub(crate) use super::configuration::{Qwen3_5MoEConfig, Qwen3_5MoEConfigError};
pub(crate) use super::quantizations;
pub(crate) use super::quantizations::optiq::{OptiQMetadata, OptiQMetadataError};
pub(crate) use super::vision::{Qwen3_5MoEVisionConfig, vision_tensor_spec};

pub use artifact::{
    Qwen3_5MoEArtifactValidationError, Qwen3_5MoEArtifactValidator, ValidatedQwen3_5MoEArtifact,
};
pub use mtp_artifact_capability::Qwen3_5MoEMtpArtifactCapability;
pub use mtp_tensor_namespace::{
    qwen3_5_moe_mtp_tensor_profiles, qwen3_5_moe_quantized_mtp_tensor_names,
};
pub use shard_index::{MAXIMUM_INDEX_BYTES, Qwen3_5MoEArtifactError, Qwen3_5MoEShardIndex};
pub use tensor_spec::{
    qwen3_5_moe_language_tensor_profiles, qwen3_5_moe_resident_language_tensor_profiles,
};
