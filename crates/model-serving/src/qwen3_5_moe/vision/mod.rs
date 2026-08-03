mod image_processor;
#[cfg(feature = "direct-mlx")]
mod vision_attention;
mod vision_config;
mod vision_input_plan;
#[cfg(feature = "direct-mlx")]
mod vision_model;
#[cfg(feature = "direct-mlx")]
mod vision_rotary_embedding;
pub(crate) mod vision_tensor_spec;
#[cfg(feature = "direct-mlx")]
mod vision_weights;
#[cfg(feature = "direct-mlx")]
pub(crate) mod visual_embedding_injection;
mod visual_embeddings;

pub use image_processor::{
    Qwen3_5MoEImageDimensions, Qwen3_5MoEImageGrid, Qwen3_5MoEImageProcessingError,
    Qwen3_5MoEImageProcessor, Qwen3_5MoEProcessedImage,
};
pub use vision_config::Qwen3_5MoEVisionConfig;
pub use vision_input_plan::{Qwen3_5MoEVisionInputPlan, Qwen3_5MoEVisionInputPlanError};
#[cfg(feature = "direct-mlx")]
pub use vision_model::Qwen3_5MoEVisionModel;
pub use vision_tensor_spec::qwen3_5_moe_vision_tensor_profiles;
#[cfg(feature = "direct-mlx")]
pub use vision_weights::Qwen3_5MoEVisionWeights;
#[cfg(feature = "direct-mlx")]
pub use visual_embedding_injection::qwen3_5_moe_inject_visual_embeddings;
pub use visual_embeddings::{
    Qwen3_5MoEVisualEmbeddingRequiredImage, Qwen3_5MoEVisualEmbeddingSuffixPlan,
    Qwen3_5MoEVisualEmbeddingSuffixPlanError, plan_qwen3_5_moe_visual_embedding_suffix,
};

#[cfg(feature = "direct-mlx")]
pub(crate) use super::artifacts::ValidatedQwen3_5MoEArtifact;
pub(crate) use super::configuration::Qwen3_5MoEConfigError;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::model::Qwen3_5MoEExecutionError;
