#[cfg(feature = "direct-mlx")]
pub(crate) mod adaptive_ram_growth_logging;
#[cfg(feature = "direct-mlx")]
mod artifact_loading;
#[cfg(feature = "direct-mlx")]
mod decoder_cache_dtype_flow;
mod decoder_layer_forward;
#[cfg(feature = "direct-mlx")]
pub(crate) mod decoder_layer_weights;
#[cfg(feature = "direct-mlx")]
mod error;
#[cfg(feature = "direct-mlx")]
mod evaluation;
#[cfg(feature = "direct-mlx")]
mod forward_attribution;
mod forward_attribution_generation;
#[cfg(feature = "direct-mlx")]
mod forward_contract;
#[cfg(feature = "direct-mlx")]
mod forward_graph;
#[cfg(feature = "direct-mlx")]
mod full_attention;
#[cfg(feature = "direct-mlx")]
mod gated_delta;
#[cfg(feature = "direct-mlx")]
mod gated_delta_boundary_checkpoints;
#[cfg(feature = "direct-mlx")]
mod gated_delta_pipelined_kernel;
mod gated_delta_sequence;
mod gated_delta_sequence_contract;
#[cfg(feature = "direct-mlx")]
pub(crate) mod gdn_decode_prework_kernel;
#[cfg(feature = "direct-mlx")]
mod live_memory_limit;
#[cfg(feature = "direct-mlx")]
pub(crate) mod memory_admission;
#[cfg(feature = "direct-mlx")]
mod memory_breakdown;
#[cfg(feature = "direct-mlx")]
pub(crate) mod model;
#[cfg(feature = "direct-mlx")]
mod model_chunking_configuration;
#[cfg(feature = "direct-mlx")]
mod tensor_slicing;
#[cfg(feature = "direct-mlx")]
pub(crate) mod weights;
#[cfg(feature = "direct-mlx")]
pub(crate) mod weights_validation;

#[cfg(feature = "direct-mlx")]
pub use error::Qwen3_5ExecutionError;
#[cfg(feature = "direct-mlx")]
pub use forward_graph::Qwen3_5TargetForwardOutput;
#[cfg(feature = "direct-mlx")]
pub use full_attention::qwen3_5_full_attention_step;
#[cfg(feature = "direct-mlx")]
pub use gated_delta::qwen3_5_gated_delta_step;
#[cfg(feature = "direct-mlx")]
pub use gated_delta_boundary_checkpoints::{
    Qwen3_5GatedDeltaBoundaryCheckpointResult, qwen3_5_gated_delta_checkpoint_kernel,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints_ops_fallback,
};
#[cfg(feature = "direct-mlx")]
pub use gated_delta_sequence::{
    qwen3_5_gated_delta_kernel, qwen3_5_gated_delta_sequence,
    qwen3_5_gated_delta_sequence_ops_fallback,
};
#[cfg(feature = "direct-mlx")]
pub use gdn_decode_prework_kernel::{
    is_gdn_decode_prework_eligible, qwen3_5_gdn_decode_prework, qwen3_5_gdn_decode_prework_kernel,
};
#[cfg(feature = "direct-mlx")]
pub use model::Qwen3_5Model;
#[cfg(feature = "direct-mlx")]
pub use model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
#[cfg(feature = "direct-mlx")]
pub use weights::Qwen3_5Weights;

#[cfg(feature = "direct-mlx")]
pub(crate) use super::artifacts::qwen3_5_resident_language_tensor_profiles;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::artifacts::{Qwen3_5ShardIndex, ValidatedQwen3_5Artifact};
#[cfg(feature = "direct-mlx")]
pub(crate) use super::configuration::{Qwen3_5Config, Qwen3_5FeedForwardArchitecture};
#[cfg(feature = "direct-mlx")]
pub(crate) use super::decoder::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::vision::{Qwen3_5VisionModel, visual_embedding_injection};
