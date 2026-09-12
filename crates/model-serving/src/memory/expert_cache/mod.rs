//! Expert-cache policy owned by the memory package.
//!
//! Prefill pinning stays in `expert_paging::RetainedExpertPageCache`. Decode
//! residency lives in `decode` and never names a layer pin.

pub mod decode;

pub use decode::{
    DecodeExpertCache, PreviousTokenPrefetchCandidate, PreviousTokenPrefetchLayerCapacity,
    PreviousTokenPrefetchPlan, ResidentExpertWeight, merge_predicted_experts_into_protected_set,
    plan_previous_token_prefetch, select_top_expert_ids_from_logits,
};
