//! Expert-cache owner in the serving crate: the page-materialization byte
//! arithmetic is compiled only with the direct-MLX feature (plus the offline
//! artifact-contract tests) and stays beside the bindings. The family-neutral
//! decode cache lives in `astronomical-memory-policy::expert_cache::decode`
//! and is re-exported here so consumer paths stay stable.

// Pure checked byte arithmetic over validated shapes with no MLX (Metal
// Performance Shaders graph engine) dependency. The direct-MLX admission
// callers and the offline artifact-contract tests (cfg(test)) are its only
// consumers, so the plain default-feature lib builds without it.
#[cfg(any(feature = "direct-mlx", test))]
mod page_materialization;

#[cfg(test)]
mod page_materialization_tests;

pub use astronomical_memory_policy::expert_cache::decode::{
    DecodeExpertCache, PreviousTokenPrefetchCandidate, PreviousTokenPrefetchLayerCapacity,
    PreviousTokenPrefetchPlan, ResidentExpertWeight, plan_previous_token_prefetch,
};
#[cfg(any(feature = "direct-mlx", test))]
pub use page_materialization::expert_page_peak_allocation_bytes;
#[cfg(feature = "direct-mlx")]
pub use page_materialization::source_expert_page_payload_bytes;
