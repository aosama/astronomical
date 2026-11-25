#[cfg(feature = "direct-mlx")]
mod artifact_loading;
#[cfg(feature = "direct-mlx")]
mod decision_readout;
pub use decision_readout::DecoderLayerRecomputation;
#[cfg(feature = "direct-mlx")]
mod decoder_layer_forward;
#[cfg(feature = "direct-mlx")]
mod decoder_stack_suffix_forward;
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
pub mod forward_mlx_working_buffer_probe;
#[cfg(feature = "direct-mlx")]
mod live_memory_limit;
#[cfg(feature = "direct-mlx")]
pub mod request_errors;
#[cfg(feature = "direct-mlx")]
mod resident_execution;
#[cfg(feature = "direct-mlx")]
mod resident_model;
#[cfg(feature = "direct-mlx")]
mod sorted_expert_shared_combination_switch;
#[cfg(feature = "direct-mlx")]
mod tree_verification_forward;
mod verification_forward;

#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::artifacts::{Qwen3_5ShardIndex, ValidatedQwen3_5Artifact};
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::configuration::Qwen3_5FeedForwardArchitecture;
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::model::Qwen3_5ModelBase;
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::model::target_forward_graph::Qwen3_5TargetForwardOutput;
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::model_math::weights::Qwen3_5Weights;
pub use crate::qwen3_5_core::model_math::{
    feed_forward_weights, forward_contract, output_combination, routing,
};
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::vision::Qwen3_5VisionModel;
#[cfg(feature = "direct-mlx")]
pub use decoder_stack_suffix_forward::RecordedDecoderLayerInput;
#[cfg(feature = "direct-mlx")]
pub use resident_model::Qwen3_5ResidentModel;
#[cfg(feature = "direct-mlx")]
pub use text_generation::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
pub use text_generation::execution_error::TextExecutionError;
#[cfg(feature = "direct-mlx")]
pub use text_generation::model::model_chunking_configuration::TextModelChunkingConfiguration;
#[cfg(feature = "direct-mlx")]
pub use tree_verification_forward::Qwen3_5DraftTreeVerification;
pub use verification_forward::{
    Qwen3_5DraftedRowsVerification, Qwen3_5SpeculativeVerificationOutput, VerificationLogitsMode,
};
