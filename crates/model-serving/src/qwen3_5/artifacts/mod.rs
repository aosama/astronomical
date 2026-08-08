mod artifact;
mod artifact_helpers;
mod shard_index;
pub(crate) mod tensor_spec;
mod vision_validation;

pub(crate) use super::configuration::{
    Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture,
};
pub(crate) use super::quantizations;
pub(crate) use super::quantizations::optiq::{OptiQMetadata, OptiQMetadataError};
pub(crate) use super::vision::{Qwen3_5VisionConfig, vision_tensor_spec};

pub use crate::qwen3_5::multi_token_prediction::Qwen3_5MtpArtifactCapability;
pub use crate::qwen3_5::multi_token_prediction::{
    qwen3_5_mtp_tensor_names, qwen3_5_mtp_tensor_profiles,
};
pub use artifact::{
    Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator, ValidatedQwen3_5Artifact,
};
pub use shard_index::{MAXIMUM_INDEX_BYTES, Qwen3_5ArtifactError, Qwen3_5ShardIndex};
pub use tensor_spec::{
    qwen3_5_language_tensor_profiles, qwen3_5_resident_language_tensor_profiles,
};
