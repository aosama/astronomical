#[cfg(feature = "direct-mlx")]
pub(crate) mod adaptive_ram_growth_logging;
#[cfg(feature = "direct-mlx")]
mod artifact_loading;
#[cfg(feature = "direct-mlx")]
mod attention_execution;
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
mod gated_delta_sequence;
mod gated_delta_sequence_contract;
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
mod speculative_prefill;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_attention_capture;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_draft_forward;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_selection;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_sparse_target;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_visual_forward;
#[cfg(feature = "direct-mlx")]
mod target_verification_four_row_quantized_linear;
#[cfg(feature = "direct-mlx")]
mod target_verification_quantized_linear;
#[cfg(feature = "direct-mlx")]
mod tensor_slicing;
#[cfg(feature = "direct-mlx")]
pub(crate) mod weights;
#[cfg(feature = "direct-mlx")]
pub(crate) mod weights_validation;

#[cfg(feature = "direct-mlx")]
pub use super::multi_token_prediction::Qwen3_5MtpForwardOutput;
#[cfg(feature = "direct-mlx")]
pub use error::Qwen3_5ExecutionError;
#[cfg(feature = "direct-mlx")]
pub(crate) use forward_contract::{forward_state_arrays, validate_forward_input};
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
pub use model::Qwen3_5Model;
#[cfg(feature = "direct-mlx")]
pub use model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
#[cfg(feature = "direct-mlx")]
pub(crate) use speculative_prefill::{
    Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlock,
    Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlockConsumer,
    Qwen3_5SpeculativePrefillDraftScoringOutcome,
};
#[cfg(feature = "direct-mlx")]
pub use speculative_prefill_attention_capture::qwen3_5_aggregate_speculative_prefill_attention_weights;
#[cfg(feature = "direct-mlx")]
pub use speculative_prefill_selection::qwen3_5_select_speculative_prefill_token_positions_on_gpu;
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use target_verification_four_row_quantized_linear::four_row_split_k_quantized_linear_kernel;
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use target_verification_quantized_linear::{
    Qwen3_5TargetVerificationProjection, Qwen3_5TargetVerificationProjectionDispatch,
    qwen3_5_target_verification_quantized_linear, target_verification_quantized_linear_kernel,
};
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
#[cfg(feature = "direct-mlx")]
pub(crate) use speculative_prefill_attention_capture::Qwen3_5AttentionCapture;
