//! Family-neutral expert residency policy. While a prompt is read every
//! layer may be kept whole and the `astro_expert_cache` capacity is the only refusal; a
//! plan is made once, when the reply begins, and says for each layer where
//! it should end up: kept whole, hot, or absent.

mod complete_headroom;
mod decode_seating;
mod foundation_overlay;
mod observed_mode;
mod ownership_mode;
mod page_commit;
mod planner;
mod seated_layers;
mod startup_residency_facts;
mod types;
mod validation;

#[cfg(test)]
mod complete_residency_headroom_boundary_tests;
#[cfg(test)]
mod expert_residency_policy_tests;
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
mod startup_residency_facts_tests;
#[cfg(test)]
mod validation_boundary_tests;

pub use complete_headroom::CompleteResidencyHeadroomBoundary;
pub use decode_seating::kept_layer_indexes_to_seat_before_reply;
pub use observed_mode::observed_expert_memory_mode;
pub use ownership_mode::classify_expert_memory_mode;
pub use page_commit::{
    DECODE_TABLE_TARGET_SLOT_COUNT, complete_layers_beyond_keep_count,
    decode_table_target_slot_count, decode_warm_floor_eviction_indexes,
    decode_whole_layer_keep_count, hot_expert_warm_slot_count, minimum_decode_warm_slot_count,
    should_commit_mandatory_complete_layer, should_commit_mandatory_routed_page,
    should_commit_multi_token_union_as_complete_layer,
    should_complete_partial_layer_on_prefill_read, should_seed_partial_layer_on_prefill_read,
};
pub use planner::plan_expert_residency;
pub use seated_layers::seated_complete_layer_count;
pub use startup_residency_facts::StartupResidencyFacts;
pub use types::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerState, ExpertResidencyPlan,
    ExpertResidencyPlanError, RetainedExpertPageClass,
};
