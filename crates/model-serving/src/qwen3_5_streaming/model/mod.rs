#[cfg(feature = "direct-mlx")]
mod cached_plus_streamed_page_route;
#[cfg(feature = "direct-mlx")]
mod diagnostic_paging;
#[cfg(feature = "direct-mlx")]
mod expert_memory_mode;
/// Temporary retained-page freeze while the remaining prompt still needs RAM.
#[cfg(feature = "direct-mlx")]
mod expert_retention_memory_pressure;
#[cfg(feature = "direct-mlx")]
pub(crate) mod expert_reuse;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::model_math::{
    feed_forward_weights, output_combination, routing,
};
#[cfg(feature = "direct-mlx")]
mod forward;
#[cfg(feature = "direct-mlx")]
mod mixed_decode_execution;
mod paged_execution;
#[cfg(feature = "direct-mlx")]
mod paged_route_resolution;
#[cfg(feature = "direct-mlx")]
mod phase_aware_expert_residency;
#[cfg(feature = "direct-mlx")]
mod prefill_execution_mode;
#[cfg(feature = "direct-mlx")]
mod read_through_residency;
#[cfg(feature = "direct-mlx")]
mod route_id_materialization;
#[cfg(feature = "direct-mlx")]
pub(crate) mod route_observation;
#[cfg(feature = "direct-mlx")]
mod seat_planned_complete_layers;
#[cfg(feature = "direct-mlx")]
pub(crate) mod streaming_model;

pub(crate) use crate::qwen3_5_core::model::Qwen3_5ModelBase;
pub(crate) use crate::qwen3_5_core::model_math::gated_delta_sequence;
pub(crate) use crate::qwen3_5_core::model_math::{
    decoder_layer_weights, error, forward_contract, gated_delta_boundary_checkpoints,
    gdn_decode_prework_kernel, weights,
};
pub use crate::qwen3_5_streaming::model::streaming_model::Qwen3_5StreamingModel as Qwen3_5Model;

#[cfg(feature = "direct-mlx")]
pub(crate) mod adaptive_ram_growth_logging;
#[cfg(feature = "direct-mlx")]
mod artifact_loading;
#[cfg(feature = "direct-mlx")]
mod decoder_cache_dtype_flow;
#[cfg(feature = "direct-mlx")]
mod decoder_layer_forward;
#[cfg(feature = "direct-mlx")]
mod evaluation;
#[cfg(feature = "direct-mlx")]
mod forward_attribution;
#[cfg(feature = "direct-mlx")]
mod forward_attribution_generation;
#[cfg(feature = "direct-mlx")]
mod forward_graph;
#[cfg(feature = "direct-mlx")]
mod live_memory_limit;
#[cfg(feature = "direct-mlx")]
pub(crate) mod memory_admission;
#[cfg(feature = "direct-mlx")]
mod memory_breakdown;

#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::model::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
#[cfg(feature = "direct-mlx")]
pub use error::Qwen3_5ExecutionError;
#[cfg(feature = "direct-mlx")]
pub use forward_graph::Qwen3_5TargetForwardOutput;
#[cfg(feature = "direct-mlx")]
pub use weights::Qwen3_5Weights;

#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::artifacts::{Qwen3_5ShardIndex, ValidatedQwen3_5Artifact};
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::configuration::Qwen3_5FeedForwardArchitecture;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::vision::{Qwen3_5VisionModel, visual_embedding_injection};

#[cfg(feature = "direct-mlx")]
pub use cached_plus_streamed_page_route::Qwen3_5MoECachedPlusStreamedPageRoute;
#[cfg(feature = "direct-mlx")]
pub(crate) use expert_retention_memory_pressure::reclaim_retained_experts_for_request_memory_pressure;
pub use mixed_decode_execution::qwen3_5_moe_combine_partial_route_outputs_for_tests;
#[cfg(feature = "direct-mlx")]
pub use output_combination::qwen3_5_moe_combine_experts;
#[cfg(feature = "direct-mlx")]
pub(crate) use paged_route_resolution::{
    PagedForwardMissingRouteCollector, PagedRouteValidationOutcome,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use phase_aware_expert_residency::record_expert_reclamation_attribution;
#[cfg(feature = "direct-mlx")]
pub use prefill_execution_mode::Qwen3_5MoEPagedPrefillExecutionMode;
#[cfg(feature = "direct-mlx")]
pub use routing::{
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel, qwen3_5_moe_unsorted_expert_weighted_sum,
};
