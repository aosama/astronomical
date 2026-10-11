#[cfg(feature = "direct-mlx")]
pub(crate) mod adaptive_ram_growth_logging;
#[cfg(feature = "direct-mlx")]
mod artifact_loading;
#[cfg(feature = "direct-mlx")]
mod decoder_cache_dtype_flow;
#[cfg(feature = "direct-mlx")]
mod decoder_layer_forward;
#[cfg(feature = "direct-mlx")]
mod evaluation;
#[cfg(feature = "direct-mlx")]
mod forward;
#[cfg(feature = "direct-mlx")]
mod forward_attribution;
#[cfg(feature = "direct-mlx")]
mod forward_attribution_generation;
#[cfg(feature = "direct-mlx")]
mod forward_graph;
#[cfg(feature = "direct-mlx")]
mod live_memory_limit;
#[cfg(feature = "direct-mlx")]
pub(crate) mod memory_admission;
#[cfg(feature = "direct-mlx")]
mod memory_breakdown;
#[cfg(feature = "direct-mlx")]
mod resident_execution;
#[cfg(feature = "direct-mlx")]
mod resident_model;

#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::artifacts::{Qwen3_5ShardIndex, ValidatedQwen3_5Artifact};
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::configuration::Qwen3_5FeedForwardArchitecture;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::model::Qwen3_5ModelBase;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::model::full_attention::qwen3_5_full_attention_step;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::model::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::model_math::weights::Qwen3_5Weights;
pub(crate) use crate::qwen3_5_core::model_math::{
    decoder_layer_weights, error, feed_forward_weights, forward_contract,
    gated_delta_boundary_checkpoints, gated_delta_sequence, gated_delta_step,
    gdn_decode_prework_kernel, output_combination, routing, weights,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::vision::{
    Qwen3_5ProcessedImage, Qwen3_5VisualEmbeddingSuffixPlan,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::vision::{Qwen3_5VisionModel, visual_embedding_injection};
#[cfg(feature = "direct-mlx")]
pub use forward_graph::Qwen3_5TargetForwardOutput;
#[cfg(feature = "direct-mlx")]
pub(crate) use resident_model::Qwen3_5ResidentModel;
#[cfg(feature = "direct-mlx")]
pub(crate) use resident_model::Qwen3_5ResidentModel as Qwen3_5Model;
