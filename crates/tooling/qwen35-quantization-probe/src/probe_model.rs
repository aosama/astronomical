//! The in-memory Qwen3.5 model the sensitivity probe swaps weights in:
//! the shared model base from the Qwen 3.5 crate, plus every routed expert
//! held whole, so one projection can be re-encoded in place and put back.
//!
//! Serving runs the Qwen 3.5 crate's streaming model, which keeps routed
//! experts in the `astro_expert_cache` and reads the rest from the SSD; a
//! swapped copy of one projection for every expert of a layer has no place
//! to go there. The probe therefore owns this model, built from the same
//! shared pieces (binding, attention, dense feed-forward, routing, expert
//! combination), so the sweep still measures the kernels serving runs.

mod artifact_loading;
mod decoder_layer_forward;
mod decoder_stack_suffix_forward;
mod evaluation;
// Crate-visible so the probe can swap one routed-expert projection for a
// re-encoded copy and put the original back.
mod expert_execution;
pub mod experts;
mod forward;
mod forward_graph;
mod model;
mod prefill_forward;

pub use decoder_stack_suffix_forward::RecordedDecoderLayerInput;
pub use model::Qwen3_5ProbeModel;
pub use qwen35::qwen3_5_core::artifacts::{Qwen3_5ShardIndex, ValidatedQwen3_5Artifact};
pub use qwen35::qwen3_5_core::configuration::Qwen3_5FeedForwardArchitecture;
pub use qwen35::qwen3_5_core::model::Qwen3_5ModelBase;
pub use qwen35::qwen3_5_core::model_math::weights::Qwen3_5Weights;
pub use qwen35::qwen3_5_core::model_math::{
    feed_forward_weights, forward_contract, output_combination, routing,
};
pub use qwen35::qwen3_5_core::vision::Qwen3_5VisionModel;
pub use text_generation::RequestDecoderStateStack;
pub use text_generation::execution_error::TextExecutionError;
pub use text_generation::model::model_chunking_configuration::TextModelChunkingConfiguration;
