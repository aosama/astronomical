//! Expert-cache owner in the serving crate: the page-materialization byte
//! arithmetic is compiled only with the direct-MLX feature (plus the unit
//! artifact-contract tests) and stays beside the bindings.

// Pure checked byte arithmetic over validated shapes with no MLX (Metal
// Performance Shaders graph engine) dependency. The direct-MLX admission
// callers and the unit artifact-contract tests (cfg(test)) are its only
// consumers, so the plain default-feature lib builds without it.
#[cfg(any(feature = "direct-mlx", test))]
mod page_materialization;

#[cfg(test)]
mod page_materialization_tests;

// Session counters are pure arithmetic. Direct-MLX streaming owns them at
// runtime; the unit tests own them without the graphics-processor runtime.
// Leaving the module ungated makes a default build fail on unused items,
// because warnings are denied.
#[cfg(any(feature = "direct-mlx", test))]
mod streaming_stats;
#[cfg(test)]
mod streaming_stats_tests;
#[cfg(feature = "direct-mlx")]
pub(crate) use crate::performance_attribution::StreamedPageSourcePlan;
#[cfg(any(feature = "direct-mlx", test))]
pub use page_materialization::expert_page_peak_allocation_bytes;
#[cfg(feature = "direct-mlx")]
pub use page_materialization::source_expert_page_payload_bytes;
#[cfg(any(feature = "direct-mlx", test))]
pub(crate) use streaming_stats::{
    ExpertStreamingSessionCounters, RouteCoverageClass, expert_layer_topology_from_residency,
    no_cache_route_source_bytes, record_streamed_page_source_plan_in_both_ledgers,
};
