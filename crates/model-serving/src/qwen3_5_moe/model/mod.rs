/// Demand-selected page fill after prefill. Does not restore complete residency.
#[cfg(feature = "direct-mlx")]
mod decode_warm_expert_layers;
#[cfg(feature = "direct-mlx")]
mod diagnostic_paging;
#[cfg(feature = "direct-mlx")]
mod expert_memory_mode;
/// Atomic complete-owner promote/demote. No sticky "stay paged" flag.
#[cfg(feature = "direct-mlx")]
mod expert_residency_transition;
/// Temporary retained-page freeze while the remaining prompt still needs RAM.
#[cfg(feature = "direct-mlx")]
mod expert_retention_memory_pressure;
#[cfg(feature = "direct-mlx")]
pub(crate) mod feed_forward_weights;
#[cfg(feature = "direct-mlx")]
mod forward;
#[cfg(feature = "direct-mlx")]
mod output_combination;
#[cfg(feature = "direct-mlx")]
mod paged_execution;
#[cfg(feature = "direct-mlx")]
mod paged_route_resolution;
#[cfg(feature = "direct-mlx")]
mod prefill_execution_mode;
#[cfg(feature = "direct-mlx")]
mod resident_execution;
#[cfg(feature = "direct-mlx")]
mod route_id_materialization;
#[cfg(feature = "direct-mlx")]
mod routing;
#[cfg(feature = "direct-mlx")]
mod split_page_route;

#[cfg(feature = "direct-mlx")]
pub(crate) use expert_residency_transition::{
    Qwen3_5ExpertResidencyPromotionOutcome, Qwen3_5ExpertResidencyTransitionReason,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use expert_retention_memory_pressure::reclaim_retained_experts_for_request_memory_pressure;
#[cfg(feature = "direct-mlx")]
pub use output_combination::qwen3_5_moe_combine_experts;
#[cfg(feature = "direct-mlx")]
pub(crate) use paged_route_resolution::{
    PagedForwardMissingRouteCollector, PagedRouteValidationOutcome,
};
#[cfg(feature = "direct-mlx")]
pub use prefill_execution_mode::Qwen3_5MoEPagedPrefillExecutionMode;
#[cfg(feature = "direct-mlx")]
pub use routing::{
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
#[cfg(feature = "direct-mlx")]
pub use split_page_route::Qwen3_5MoESplitPageRoute;
