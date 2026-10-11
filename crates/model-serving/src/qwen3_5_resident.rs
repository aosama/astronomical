#![allow(dead_code)]
#![allow(unused_imports)]

mod experts;
mod inference_execution;
mod model;

pub(crate) use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
pub(crate) use experts::{
    Qwen3_5ResidentExpertLayerWeights, Qwen3_5ResidentExpertWeights, Qwen3_5ResidentGateUpWeights,
};
pub use experts::{
    ResidentLayerArraysForTests, ResidentProjectionArraysForTests,
    maximum_resident_gate_up_fusion_transient_payload_bytes, resident_layer_arrays_for_tests,
};
pub(crate) use inference_execution::{Qwen3_5Engine, Qwen3_5InferenceExecution};
pub use inference_execution::{
    Qwen3_5Engine as Qwen3_5ResidentEngine,
    Qwen3_5PromptProcessingChunkSizer as Qwen3_5ResidentPromptProcessingChunkSizer,
    Qwen3_5PromptProcessingChunkSizerError as Qwen3_5ResidentPromptProcessingChunkSizerError,
};
pub(crate) use model::{Qwen3_5ExecutionError, Qwen3_5ResidentModel};
