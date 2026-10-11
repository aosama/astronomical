mod artifact;
mod artifact_helpers;
mod artifact_inventory;
#[cfg(feature = "direct-mlx")]
pub mod expert_gate_up_fusion_plan;
pub mod quantized_expert_layer_plan;
mod ram_budget_measurements;
mod shard_index;
pub(crate) mod streaming_revision;
pub(crate) mod tensor_spec;
mod validated_artifact;
mod vision_validation;

pub(crate) use super::configuration::{
    Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture,
};
pub(crate) use super::quantizations;
pub(crate) use super::quantizations::optiq::{OptiQMetadata, OptiQMetadataError};
pub(crate) use super::vision::{Qwen3_5VisionConfig, vision_tensor_spec};
pub use artifact::{Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator};
pub use quantized_expert_layer_plan::{
    build_quantized_expert_layer_plan, build_quantized_expert_layer_plans,
};
pub use ram_budget_measurements::{
    Qwen3_5RamBudgetGeometryError, mlx_ram_budget_model_geometry_from_validated_artifact,
};
pub use shard_index::{MAXIMUM_INDEX_BYTES, Qwen3_5ArtifactError, Qwen3_5ShardIndex};
pub use tensor_spec::{
    qwen3_5_language_tensor_profiles, qwen3_5_resident_language_tensor_profiles,
};
pub use validated_artifact::ValidatedQwen3_5Artifact;
