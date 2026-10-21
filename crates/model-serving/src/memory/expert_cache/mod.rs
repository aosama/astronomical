//! Expert-cache policy owned by the memory package.
//!
//! Prefill pinning stays in `expert_paging::RetainedExpertPageCache`. Decode
//! residency lives in `decode` and never names a layer pin.

pub mod decode;
// Pure checked byte arithmetic over validated shapes with no MLX (Metal
// Performance Shaders graph engine) dependency. The direct-MLX admission
// callers and the offline artifact-contract tests (cfg(test)) are its only
// consumers, so the plain default-feature lib builds without it.
#[cfg(any(feature = "direct-mlx", test))]
mod page_materialization;

#[cfg(test)]
mod page_materialization_tests;

pub use decode::{
    DecodeExpertCache, PreviousTokenPrefetchCandidate, PreviousTokenPrefetchLayerCapacity,
    PreviousTokenPrefetchPlan, ResidentExpertWeight, plan_previous_token_prefetch,
};
#[cfg(any(feature = "direct-mlx", test))]
pub use page_materialization::expert_page_peak_allocation_bytes;
#[cfg(feature = "direct-mlx")]
pub use page_materialization::{
    retained_expert_page_payload_bytes, source_expert_page_payload_bytes,
};
