//! Family-neutral expert residency policy. Prefill keeps a request-stable
//! contract; generation completes partial pages whenever the complete payload fits.

mod complete_headroom;
mod decode_seating;
mod foundation_overlay;
mod observed_mode;
mod ownership_mode;
mod page_commit;
mod planner;
mod promotion_on_raise;
mod release;
mod request_plan;
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
mod release_boundary_tests;
#[cfg(test)]
mod request_plan_boundary_tests;
#[cfg(test)]
mod request_plan_seated_charge_boundary_tests;
#[cfg(test)]
mod validation_boundary_tests;

pub use complete_headroom::CompleteResidencyHeadroomBoundary;
pub use decode_seating::complete_layer_indexes_required_before_decode;
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
pub use release::should_enact_planned_expert_release;
pub use request_plan::{
    RequestExpertLayerRole, RequestExpertResidency, publish_request_stable_residency_plan,
    retained_complete_layer_ceiling_after_prefill_budget_refresh,
    retained_resident_ceiling_after_budget_refresh, seated_complete_layer_count,
};
pub use types::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPlan, ExpertResidencyPlanError, RetainedExpertPageClass,
};
