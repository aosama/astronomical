#![allow(dead_code)]
#![allow(unused_imports)]

pub mod dflash_draft;
// Crate-visible so the sensitivity probe can swap one routed-expert
// projection for a re-encoded copy and put the original back.
pub(crate) mod experts;
mod host_description;
mod inference_execution;
mod model;
mod neural_engine_chunk_comparison;
mod neural_engine_idle_watch;
mod neural_engine_mlp_prefill;
mod neural_engine_output_gate_prefill;
mod neural_engine_prefill_loading;
mod neural_engine_program_cache;
mod neural_engine_program_measurement;
mod neural_engine_program_memory_record;
mod neural_engine_program_placement_record;
mod neural_engine_program_preparation;
mod neural_engine_program_settlement;
mod neural_engine_program_speed_record;

pub(crate) use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
pub(crate) use experts::maximum_resident_gate_up_fusion_transient_payload_bytes;
pub(crate) use experts::{
    Qwen3_5ResidentExpertLayerWeights, Qwen3_5ResidentExpertWeights, Qwen3_5ResidentGateUpWeights,
};
pub use experts::{
    ResidentLayerArraysForTests, ResidentProjectionArraysForTests, resident_layer_arrays_for_tests,
};
pub(crate) use inference_execution::Qwen3_5InferenceExecution;
pub use inference_execution::speculative_decode_lifecycle::{
    SpeculationRoundOutcome, SpeculationRoundRequest, VerificationTokenChoice,
    run_speculative_round,
};
pub use inference_execution::{
    Qwen3_5ResidentEngine, Qwen3_5ResidentPromptProcessingChunkSizer,
    Qwen3_5ResidentPromptProcessingChunkSizerError,
};
pub(crate) use model::Qwen3_5ExecutionError;
pub use model::Qwen3_5ResidentModel;
#[cfg(feature = "direct-mlx")]
pub(crate) use model::RecordedDecoderLayerInput;
pub use model::VerificationLogitsMode;
