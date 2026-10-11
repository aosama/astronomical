mod resident_expert_layer_weights;
mod resident_expert_loading;
mod resident_expert_weights;
mod resident_gate_up_fusion;

pub(crate) use resident_expert_layer_weights::Qwen3_5ResidentExpertLayerWeights;
pub(crate) use resident_expert_weights::Qwen3_5ResidentExpertWeights;
pub(crate) use resident_gate_up_fusion::Qwen3_5ResidentGateUpWeights;
pub use resident_gate_up_fusion::maximum_resident_gate_up_fusion_transient_payload_bytes;
pub use resident_gate_up_fusion::{
    ResidentLayerArraysForTests, ResidentProjectionArraysForTests, resident_layer_arrays_for_tests,
};
