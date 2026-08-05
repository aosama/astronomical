#[cfg(feature = "direct-mlx")]
mod expert_memory_mode;
#[cfg(feature = "direct-mlx")]
mod expert_retention_memory_pressure;
#[cfg(feature = "direct-mlx")]
pub(crate) mod feed_forward_weights;
#[cfg(feature = "direct-mlx")]
mod output_combination;
#[cfg(feature = "direct-mlx")]
mod paged_execution;
#[cfg(feature = "direct-mlx")]
mod paged_forward;
#[cfg(feature = "direct-mlx")]
mod prefill_execution_mode;
#[cfg(feature = "direct-mlx")]
mod routing;

#[cfg(feature = "direct-mlx")]
pub(crate) use expert_retention_memory_pressure::reclaim_retained_experts_for_request_memory_pressure;
#[cfg(feature = "direct-mlx")]
pub use output_combination::qwen3_5_moe_combine_experts;
#[cfg(feature = "direct-mlx")]
pub use paged_execution::qwen3_5_moe_remap_expert_page_slots;
#[cfg(feature = "direct-mlx")]
pub use prefill_execution_mode::Qwen3_5MoEPagedPrefillExecutionMode;
#[cfg(feature = "direct-mlx")]
pub use routing::{
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
