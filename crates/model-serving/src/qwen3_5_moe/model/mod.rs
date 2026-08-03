#[cfg(feature = "direct-mlx")]
mod adaptive_ram_growth_logging;
#[cfg(feature = "direct-mlx")]
mod artifact_loading;
#[cfg(feature = "direct-mlx")]
pub(crate) mod decoder_layer_weights;
#[cfg(feature = "direct-mlx")]
mod dense_mlp;
#[cfg(feature = "direct-mlx")]
mod error;
#[cfg(feature = "direct-mlx")]
mod evaluation;
#[cfg(feature = "direct-mlx")]
mod expert_memory_mode;
#[cfg(feature = "direct-mlx")]
mod expert_retention_memory_pressure;
#[cfg(feature = "direct-mlx")]
mod expert_storage_format;
#[cfg(feature = "direct-mlx")]
mod forward_attribution;
#[cfg(feature = "direct-mlx")]
mod forward_contract;
#[cfg(feature = "direct-mlx")]
mod forward_graph;
#[cfg(feature = "direct-mlx")]
mod full_attention;
#[cfg(feature = "direct-mlx")]
mod fused_gate_up_swiglu;
#[cfg(feature = "direct-mlx")]
mod gated_delta;
#[cfg(feature = "direct-mlx")]
mod gated_delta_sequence;
#[cfg(feature = "direct-mlx")]
mod live_memory_limit;
#[cfg(feature = "direct-mlx")]
pub(crate) mod memory_admission;
#[cfg(feature = "direct-mlx")]
mod memory_breakdown;
#[cfg(feature = "direct-mlx")]
mod model;
#[cfg(feature = "direct-mlx")]
mod moe;
#[cfg(feature = "direct-mlx")]
mod moe_combination;
#[cfg(feature = "direct-mlx")]
mod mtp;
#[cfg(feature = "direct-mlx")]
mod mtp_forward;
#[cfg(feature = "direct-mlx")]
mod paged_moe_execution;
#[cfg(feature = "direct-mlx")]
mod paged_moe_forward;
#[cfg(feature = "direct-mlx")]
mod paged_prefill_execution_mode;
#[cfg(feature = "direct-mlx")]
mod tensor_slicing;
#[cfg(feature = "direct-mlx")]
mod weights;
#[cfg(feature = "direct-mlx")]
mod weights_validation;

#[cfg(feature = "direct-mlx")]
pub use error::Qwen3_5MoEExecutionError;
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5_moe) use expert_retention_memory_pressure::reclaim_retained_experts_for_request_memory_pressure;
#[cfg(feature = "direct-mlx")]
pub use forward_graph::Qwen3_5MoETargetForwardOutput;
#[cfg(feature = "direct-mlx")]
pub use full_attention::qwen3_5_moe_full_attention_step;
#[cfg(feature = "direct-mlx")]
pub use fused_gate_up_swiglu::{
    qwen3_5_moe_fused_four_bit_affine_gate_up_swiglu,
    qwen3_5_moe_fused_four_bit_affine_gate_up_swiglu_kernel,
};
#[cfg(feature = "direct-mlx")]
pub use gated_delta::qwen3_5_moe_gated_delta_step;
#[cfg(feature = "direct-mlx")]
pub use gated_delta_sequence::{qwen3_5_moe_gated_delta_kernel, qwen3_5_moe_gated_delta_sequence};
#[cfg(feature = "direct-mlx")]
pub use memory_admission::{
    combined_target_and_mtp_persistent_growth_bytes,
    context_memory_admission_projected_active_memory_bytes,
    persistent_prompt_cache_restore_temporary_workspace_bytes,
};
#[cfg(feature = "direct-mlx")]
pub use model::Qwen3_5MoEModel;
#[cfg(feature = "direct-mlx")]
pub use moe::{
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
#[cfg(feature = "direct-mlx")]
pub use moe_combination::qwen3_5_moe_combine_experts;
#[cfg(feature = "direct-mlx")]
pub use mtp_forward::Qwen3_5MoEMtpForwardOutput;
#[cfg(feature = "direct-mlx")]
pub use paged_moe_execution::qwen3_5_moe_remap_expert_page_slots;
#[cfg(feature = "direct-mlx")]
pub use paged_prefill_execution_mode::Qwen3_5MoEPagedPrefillExecutionMode;
#[cfg(feature = "direct-mlx")]
pub use weights::Qwen3_5MoEWeights;

#[cfg(feature = "direct-mlx")]
pub(crate) use super::artifacts::qwen3_5_moe_resident_language_tensor_profiles;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::artifacts::{Qwen3_5MoEShardIndex, ValidatedQwen3_5MoEArtifact};
#[cfg(feature = "direct-mlx")]
pub(crate) use super::configuration::Qwen3_5MoEConfig;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::decoder::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
#[cfg(feature = "direct-mlx")]
pub(crate) use super::expert_paging;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::expert_paging::ExpertWeightMemoryCache;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::vision::{Qwen3_5MoEVisionModel, visual_embedding_injection};
