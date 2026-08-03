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
    Qwen3_5ImageDimensions, Qwen3_5ImageGrid, Qwen3_5ImageProcessingError, Qwen3_5ImageProcessor,
    Qwen3_5ProcessedImage,
};
pub use vision_config::Qwen3_5VisionConfig;
pub use vision_input_plan::{Qwen3_5VisionInputPlan, Qwen3_5VisionInputPlanError};
#[cfg(feature = "direct-mlx")]
pub use vision_model::Qwen3_5VisionModel;
pub use vision_tensor_spec::qwen3_5_vision_tensor_profiles;
#[cfg(feature = "direct-mlx")]
pub use vision_weights::Qwen3_5VisionWeights;
#[cfg(feature = "direct-mlx")]
pub use visual_embedding_injection::qwen3_5_inject_visual_embeddings;
pub use visual_embeddings::{
    Qwen3_5VisualEmbeddingRequiredImage, Qwen3_5VisualEmbeddingSuffixPlan,
    Qwen3_5VisualEmbeddingSuffixPlanError, plan_qwen3_5_visual_embedding_suffix,
};

#[cfg(feature = "direct-mlx")]
pub(crate) use super::artifacts::ValidatedQwen3_5Artifact;
pub(crate) use super::configuration::Qwen3_5ConfigError;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::model::Qwen3_5ExecutionError;
