//! Family-neutral decode expert cache.
//!
//! The irreducible unit is one expert id in one sparse-layer slot. Bytes are
//! accounting. This module does not import a model family, MLX, or a page type.

mod cache;
mod demand_ledger;
mod eviction;
mod predictor_retention;
mod previous_token_prefetch;
mod resident_set;
mod weight;

pub use cache::DecodeExpertCache;
pub use predictor_retention::{
    merge_predicted_experts_into_protected_set, select_top_expert_ids_from_logits,
};
pub use previous_token_prefetch::{
    PreviousTokenPrefetchCandidate, PreviousTokenPrefetchLayerCapacity, PreviousTokenPrefetchPlan,
    plan_previous_token_prefetch,
};
pub use weight::ResidentExpertWeight;
