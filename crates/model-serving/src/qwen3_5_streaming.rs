pub(crate) mod expert_paging;
#[cfg(feature = "direct-mlx")]
pub(crate) mod inference_execution;
#[cfg(feature = "direct-mlx")]
pub(crate) mod model;

#[cfg(feature = "direct-mlx")]
pub use crate::expert_paging::build_source_manifests;
#[cfg(feature = "direct-mlx")]
pub use crate::expert_paging::contiguous_selected_runs;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
#[cfg(feature = "direct-mlx")]
pub(crate) use expert_paging::RetainedExpertCache;
pub use expert_paging::route_observation::{
    RouteObservationRecord, RouteObservationRing, sorted_unique_layer_routed_expert_ids,
};
#[cfg(feature = "direct-mlx")]
pub use expert_paging::{ExpertPagingError, Qwen3_5ExpertPager};
#[cfg(feature = "direct-mlx")]
pub use inference_execution::{
    Qwen3_5PrefillExecutionContext as Qwen3_5StreamingPrefillExecutionContext,
    Qwen3_5StreamingEngine, Qwen3_5StreamingPromptProcessingChunkSizer,
    Qwen3_5StreamingPromptProcessingChunkSizerError,
    persistent_prompt_cache_publication_advances_parent_chain,
    safe_minimum_mlx_memory_ceiling_bytes,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use model::{
    PagedForwardMissingRouteCollector, PagedRouteValidationOutcome,
    reclaim_retained_experts_for_request_memory_pressure,
};
#[cfg(feature = "direct-mlx")]
pub use model::{
    Qwen3_5MoECachedPlusStreamedPageRoute, Qwen3_5MoEPagedPrefillExecutionMode,
    qwen3_5_moe_combine_experts, qwen3_5_moe_combine_partial_route_outputs_for_tests,
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel, qwen3_5_moe_unsorted_expert_weighted_sum,
};
#[cfg(feature = "direct-mlx")]
pub use model::{
    Qwen3_5Model as Qwen3_5StreamingModel,
    Qwen3_5TargetForwardOutput as Qwen3_5StreamingTargetForwardOutput,
};

/// Model identity constants retained for sparse-artifact test fixtures.
pub const ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
pub const ORNITH_1_0_35B_OPTIQ_4BIT_REVISION: &str = "ce62c23d34b91d84f838e0b292d517dbe4b9b60f";
