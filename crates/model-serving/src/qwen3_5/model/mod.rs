#[cfg(feature = "direct-mlx")]
pub(crate) mod adaptive_ram_growth_logging;
#[cfg(feature = "direct-mlx")]
mod artifact_loading;
#[cfg(feature = "direct-mlx")]
mod decoder_cache_dtype_flow;
#[cfg(feature = "direct-mlx")]
mod decoder_layer_forward;
pub(crate) use crate::qwen3_5_core::model_math::gated_delta_sequence;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::model_math::{
    decoder_layer_weights, error, forward_contract, gated_delta_boundary_checkpoints,
    gated_delta_step, gdn_decode_prework_kernel, weights,
};
// Issue #1132 migration step 3: the shared model base moved to
// `qwen3_5_core::model`. This alias keeps every existing `Qwen3_5Model`
// reference compiling while the engine fork replaces the single model type
// with one model per engine; the alias is deleted with the fork.
pub(crate) use crate::qwen3_5_core::model::Qwen3_5ModelBase;
pub use crate::qwen3_5_streaming::model::streaming_model::Qwen3_5StreamingModel as Qwen3_5Model;
#[cfg(feature = "direct-mlx")]
mod evaluation;
#[cfg(feature = "direct-mlx")]
mod forward_attribution;
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
pub use crate::qwen3_5_core::model::full_attention::qwen3_5_full_attention_step;
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::model::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
#[cfg(feature = "direct-mlx")]
pub use error::Qwen3_5ExecutionError;
#[cfg(feature = "direct-mlx")]
pub use forward_graph::Qwen3_5TargetForwardOutput;
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
pub use gated_delta_step::qwen3_5_gated_delta_step;
#[cfg(feature = "direct-mlx")]
pub use gdn_decode_prework_kernel::{
    is_gdn_decode_prework_eligible, qwen3_5_gdn_decode_prework, qwen3_5_gdn_decode_prework_kernel,
};
#[cfg(feature = "direct-mlx")]
pub use weights::Qwen3_5Weights;

#[cfg(feature = "direct-mlx")]
pub(crate) use super::artifacts::{Qwen3_5ShardIndex, ValidatedQwen3_5Artifact};
#[cfg(feature = "direct-mlx")]
pub(crate) use super::configuration::Qwen3_5FeedForwardArchitecture;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::decoder::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::vision::{Qwen3_5VisionModel, visual_embedding_injection};
