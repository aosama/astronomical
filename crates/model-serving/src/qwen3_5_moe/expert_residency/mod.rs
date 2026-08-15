//! Complete sparse-expert ownership for the resident execution mode.
//!
//! The owner is deliberately all-or-nothing: it contains every target decoder
//! layer and the optional multi-token-prediction layer, or it is not published.
//! Native demand paging remains available through the separately retained pager;
//! there is no mixed state in which some layers are resident and others paged.

mod resident_expert_layer_weights;
mod resident_expert_loading;
mod resident_expert_weights;
mod resident_gate_up_fusion;

pub(crate) use resident_expert_layer_weights::Qwen3_5ResidentExpertLayerWeights;
pub(crate) use resident_expert_weights::Qwen3_5ResidentExpertWeights;
pub(crate) use resident_gate_up_fusion::Qwen3_5ResidentGateUpWeights;
pub use resident_gate_up_fusion::maximum_resident_gate_up_fusion_transient_payload_bytes;
