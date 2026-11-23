mod artifact;
mod artifact_helpers;
mod artifact_inventory;
#[cfg(test)]
mod artifact_public_reason_tests;
pub mod atlas_artifact;
pub mod atlas_expert_layer_plans;
#[cfg(feature = "direct-mlx")]
pub mod expert_gate_up_fusion_plan;
mod memory_geometry_measurements;
mod optiq_metadata_validation;
pub mod quantized_expert_layer_plan;
mod shard_index;
pub mod tensor_spec;
mod validated_artifact;
mod vision_validation;

pub use super::configuration::{Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture};
pub use super::quantizations;
pub use super::quantizations::producers::optiq::{OptiQMetadata, OptiQMetadataError};
pub use super::vision::{Qwen3_5VisionConfig, vision_tensor_spec};
pub use artifact::{Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator};
pub use atlas_expert_layer_plans::expert_layer_plans_for_validated_artifact;
#[cfg(feature = "direct-mlx")]
pub use engine_core::expert_paging::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource,
    SafetensorsDtype, build_quantized_expert_page_manifest_from_plan, load_quantized_expert_page,
};
pub use memory_geometry_measurements::{
    Qwen3_5MemoryGeometryError, model_memory_geometry_from_validated_artifact,
};
pub use quantized_expert_layer_plan::{
    build_quantized_expert_layer_plan, build_quantized_expert_layer_plans,
};
pub use shard_index::{MAXIMUM_INDEX_BYTES, Qwen3_5ArtifactError, Qwen3_5ShardIndex};
pub use tensor_spec::qwen3_5_language_tensor_profiles;
#[cfg(feature = "direct-mlx")]
#[cfg(feature = "direct-mlx")]
pub use tensor_spec::qwen3_5_resident_language_tensor_profiles;
pub use validated_artifact::ValidatedQwen3_5Artifact;
