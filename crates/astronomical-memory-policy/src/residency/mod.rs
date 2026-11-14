//! Family-neutral expert residency policy. While a prompt is read every
//! layer may be kept whole and the cache's capacity is the only refusal; a
//! plan is made once, when the reply begins, and says for each layer where
//! it should end up: kept whole, hot, or absent.

mod complete_headroom;
mod decode_seating;
mod foundation_overlay;
mod observed_mode;
mod ownership_mode;
mod page_commit;
mod planner;
mod promotion_on_raise;
mod seated_layers;
mod types;
mod validation;

#[cfg(test)]
mod complete_residency_headroom_boundary_tests;
#[cfg(test)]
mod expert_residency_policy_tests;
#[cfg(test)]
mod foundation_overlay_accounting_tests;
#[cfg(test)]
mod generation_full_fit_tests;
#[cfg(test)]
mod observed_mode_tests;
#[cfg(test)]
mod page_commit_boundary_tests;
#[cfg(test)]
mod paged_prefill_seat_oversubscription_tests;
#[cfg(test)]
mod phase_aware_expert_residency_tests;
#[cfg(test)]
mod planner_boundary_tests;
#[cfg(test)]
mod promotion_on_raise_tests;
#[cfg(test)]
mod validation_boundary_tests;

pub use complete_headroom::CompleteResidencyHeadroomBoundary;
pub use decode_seating::kept_layer_indexes_to_seat_before_reply;
pub use observed_mode::observed_expert_memory_mode;
pub use ownership_mode::classify_expert_memory_mode;
pub use page_commit::{
    decode_warm_floor_eviction_indexes, hot_expert_warm_slot_count, minimum_decode_warm_slot_count,
    should_commit_mandatory_complete_layer, should_commit_mandatory_routed_page,
    should_commit_multi_token_union_as_complete_layer,
    should_complete_partial_layer_on_prefill_read, should_seed_partial_layer_on_prefill_read,
};
pub use planner::plan_expert_residency;
pub use promotion_on_raise::StreamingStartupResidencyFacts;
pub use seated_layers::{
    retained_resident_ceiling_after_budget_refresh, seated_complete_layer_count,
};
pub use types::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerState, ExpertResidencyPlan,
    ExpertResidencyPlanError, RetainedExpertPageClass,
};
