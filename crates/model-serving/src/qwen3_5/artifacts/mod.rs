mod artifact;
mod mtp_artifact_capability;
pub(crate) mod mtp_tensor_namespace;
mod shard_index;
pub(crate) mod tensor_spec;
mod vision_validation;

pub(crate) use super::configuration::{
    Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture,
};
pub(crate) use super::quantizations;
pub(crate) use super::quantizations::optiq::{OptiQMetadata, OptiQMetadataError};
pub(crate) use super::vision::{Qwen3_5VisionConfig, vision_tensor_spec};

pub use artifact::{
    Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator, ValidatedQwen3_5Artifact,
};
pub use mtp_artifact_capability::Qwen3_5MtpArtifactCapability;
pub use mtp_tensor_namespace::{qwen3_5_mtp_tensor_profiles, qwen3_5_quantized_mtp_tensor_names};
pub use shard_index::{MAXIMUM_INDEX_BYTES, Qwen3_5ArtifactError, Qwen3_5ShardIndex};
pub use tensor_spec::{
    qwen3_5_language_tensor_profiles, qwen3_5_resident_language_tensor_profiles,
};
