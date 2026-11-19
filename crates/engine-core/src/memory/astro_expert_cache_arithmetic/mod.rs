//! Byte arithmetic for the `astro_expert_cache` owner in the engine core:
//! page materialization peaks and the expert-streaming session counters,
//! kept beside the MLX bindings that act on them.

// Pure checked byte arithmetic over validated shapes with no MLX (Metal
// Performance Shaders graph engine) dependency. It is compiled in every
// build: the serving crate's artifact-contract unit tests call it from
// another crate, where this crate's own `cfg(test)` does not apply.
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
pub use astronomical_model_primitives::performance_attribution::StreamedPageSourcePlan;
pub use page_materialization::expert_page_peak_allocation_bytes;
#[cfg(feature = "direct-mlx")]
pub use page_materialization::source_expert_page_payload_bytes;
#[cfg(any(feature = "direct-mlx", test))]
pub use streaming_stats::{
    ExpertStreamingSessionCounters, RouteCoverageClass, expert_layer_topology_from_residency,
    no_astro_expert_cache_route_source_bytes, record_streamed_page_source_plan_in_both_ledgers,
};
