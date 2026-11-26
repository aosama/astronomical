//! The in-memory Qwen3.5 model the quantization sensitivity probe
//! (`qwen35-quantization-probe`) swaps weights in. Serving runs the
//! streaming model; see `model/resident_model.rs` for why the probe keeps
//! its own.

// Crate-visible so the sensitivity probe can swap one routed-expert
// projection for a re-encoded copy and put the original back.
pub mod experts;
mod model;

pub use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
pub use experts::maximum_resident_gate_up_fusion_transient_payload_bytes;
pub use experts::{
    Qwen3_5ResidentExpertLayerWeights, Qwen3_5ResidentExpertWeights, Qwen3_5ResidentGateUpWeights,
};
pub use experts::{
    ResidentLayerArraysForTests, ResidentProjectionArraysForTests, resident_layer_arrays_for_tests,
};
pub use model::Qwen3_5ResidentModel;
#[cfg(feature = "direct-mlx")]
pub use model::RecordedDecoderLayerInput;
pub use model::TextExecutionError;
