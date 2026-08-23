pub(crate) mod artifacts;
#[cfg(feature = "direct-mlx")]
pub(crate) mod expert_paging;
#[cfg(feature = "direct-mlx")]
pub(crate) mod expert_residency;
#[cfg(feature = "direct-mlx")]
pub(crate) mod model;

#[cfg(feature = "direct-mlx")]
pub use crate::expert_paging::build_source_manifests;
#[cfg(feature = "direct-mlx")]
pub use crate::expert_paging::contiguous_selected_runs;
#[cfg(feature = "direct-mlx")]
pub(crate) use expert_paging::Qwen3_5RetainedExpertLayer;
#[cfg(feature = "direct-mlx")]
pub use expert_paging::quantized_expert_layer_plan::build_quantized_expert_layer_plan;
#[cfg(feature = "direct-mlx")]
pub use expert_paging::{ExpertPagingError, Qwen3_5ExpertPager};
#[cfg(feature = "direct-mlx")]
pub(crate) use expert_residency::Qwen3_5ResidentExpertWeights;
#[cfg(feature = "direct-mlx")]
pub use expert_residency::maximum_resident_gate_up_fusion_transient_payload_bytes;
#[cfg(feature = "direct-mlx")]
pub(crate) use model::feed_forward_weights::bind_qwen3_5_moe_feed_forward_weights;
#[cfg(feature = "direct-mlx")]
pub(crate) use model::{
    PagedForwardMissingRouteCollector, PagedRouteValidationOutcome,
    Qwen3_5ExpertResidencyTransitionReason, reclaim_retained_experts_for_request_memory_pressure,
};
#[cfg(feature = "direct-mlx")]
pub use model::{
    Qwen3_5MoEPagedPrefillExecutionMode, Qwen3_5MoESplitPageRoute, qwen3_5_moe_combine_experts,
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};

/// Model identity constants retained for sparse-artifact test fixtures.
pub const ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
pub const ORNITH_1_0_35B_OPTIQ_4BIT_REVISION: &str = "ce62c23d34b91d84f838e0b292d517dbe4b9b60f";
