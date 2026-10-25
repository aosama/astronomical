//! Hunt tests for foundation-overlay demotion, release ordering, and plan
//! byte self-consistency.

use crate::phase::MemoryPhase;
use crate::residency::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    RetainedExpertPageClass, plan_expert_residency, should_enact_planned_expert_release,
};

/// Uniform geometry: routed floor 20 bytes per layer, complete payload 40.
fn uniform_geometry(layer_count: usize) -> Vec<ExpertLayerGeometry> {
    (0..layer_count)
        .map(|layer_index| ExpertLayerGeometry {
            layer_index,
            complete_layer_payload_bytes: 40,
            expert_payload_bytes: 10,
            expert_capacity: 4,
            experts_per_token: 2,
        })
        .collect()
}

fn complete_layer_residency(layer_index: usize) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index,
        class: RetainedExpertPageClass::StableCompleteLayer,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes: 40,
        covered_weighted_demand: 5,
    }
}

fn partial_residency(
    layer_index: usize,
    retained_expert_count: usize,
    covered_weighted_demand: u64,
) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index,
        class: RetainedExpertPageClass::ElasticRoutedExperts,
        retained_expert_ids: (0..retained_expert_count).collect(),
        payload_bytes: 10 * u64::try_from(retained_expert_count).unwrap_or(u64::MAX),
        covered_weighted_demand,
    }
}

#[test]
fn should_preserve_a_seated_complete_layer_at_the_exact_incremental_boundary() {
    // Three uniform layers: routed floors total 60 and the seated complete
    // layer needs exactly the last 20 bytes, so an 80-byte ceiling keeps it.
    let plan = plan_expert_residency(
        MemoryPhase::Prefill,
        80,
        &uniform_geometry(3),
        &[complete_layer_residency(0)],
    )
    .expect("the seated complete layer fits at the boundary");

    assert_eq!(plan.complete_layer_targets, vec![0]);
    assert_eq!(
        plan.layer_targets[0],
        ExpertLayerResidencyTarget::PreserveComplete
    );
    assert_eq!(plan.expected_preserved_bytes, 40);
    assert_eq!(plan.reserved_routed_overlay_bytes, 40);
}

#[test]
fn should_release_a_seated_complete_layer_one_byte_below_the_incremental_boundary() {
    let plan = plan_expert_residency(
        MemoryPhase::Prefill,
        79,
        &uniform_geometry(3),
        &[complete_layer_residency(0)],
    )
    .expect("the routed floors still fit");

    assert!(
        plan.complete_layer_targets.is_empty(),
        "one byte short of the incremental, no complete layer may stay seated"
    );
    assert_eq!(
        plan.layer_targets[0],
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
    );
}

#[test]
fn should_count_the_retained_remainder_of_a_complete_layer_released_for_exact_deficit() {
    // A 70-byte ceiling funds the 60-byte routed-floor foundation but no
    // complete-layer incremental, so the seated complete layer yields — yet
    // only by its 20-byte deficit over the routed floor the plan itself
    // reserves for that layer. `ReleaseCompleteForExactDeficit` documents
    // that the rest of the layer stays retained, and the plan reserves the
    // layer's 20-byte floor in `reserved_routed_overlay_bytes`, so those
    // already-resident floor bytes must also appear as preserved bytes;
    // counting them as new reads orders a storage re-read of bytes the
    // table already holds.
    let plan = plan_expert_residency(
        MemoryPhase::Prefill,
        70,
        &uniform_geometry(3),
        &[complete_layer_residency(0)],
    )
    .expect("the routed floors fit");

    assert_eq!(
        plan.layer_targets[0],
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
    );
    assert_eq!(plan.reserved_routed_overlay_bytes, 60);
    assert_eq!(
        plan.expected_preserved_bytes, 20,
        "the released layer's resident floor must be preserved, not re-read from storage"
    );
    assert_eq!(plan.maximum_new_retained_bytes, 40);
}

#[test]
fn should_not_reduce_the_new_retention_budget_by_preserved_bytes_above_the_routed_floor() {
    // Two uniform layers under a 50-byte ceiling: both routed floors (40)
    // fit, no complete-layer incremental (20 each) fits, and preserving
    // layer 0's 30-byte elastic page consumes exactly the 10 overlay bytes
    // above its floor. Layer 1 holds nothing, so its 20-byte floor can only
    // become resident through newly retained bytes. Preserved bytes above a
    // preserved layer's routed floor are not available to layer 1's floor
    // and must not shrink its new-retention budget.
    let plan = plan_expert_residency(
        MemoryPhase::Prefill,
        50,
        &uniform_geometry(2),
        &[partial_residency(0, 3, 1_000)],
    )
    .expect("the routed floors fit");

    assert_eq!(
        plan.layer_targets[0],
        ExpertLayerResidencyTarget::PreservePartial
    );
    assert_eq!(
        plan.layer_targets[1],
        ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
    );
    assert_eq!(
        plan.maximum_new_retained_bytes, 20,
        "layer 1's admission floor must fit inside the new-retention budget exactly"
    );
}

#[test]
fn should_charge_only_the_missing_bytes_when_promoting_over_a_smaller_elastic_page() {
    // Both layers take complete targets under an 80-byte ceiling; layer 0
    // already holds a 20-byte elastic page below its 40-byte complete
    // payload, so promotion must charge only the 20 missing bytes while the
    // existing page counts as preserved.
    let plan = plan_expert_residency(
        MemoryPhase::Prefill,
        80,
        &uniform_geometry(2),
        &[partial_residency(0, 2, 1_000)],
    )
    .expect("both complete targets fit");

    assert_eq!(
        plan.layer_targets[0],
        ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
    );
    assert_eq!(plan.expected_preserved_bytes, 20);
    assert_eq!(plan.maximum_new_retained_bytes, 60);
}

#[test]
fn should_order_whole_table_release_after_every_elastic_page_and_coldest_demand_first() {
    // Deterministic release order is the eviction contract: elastic pages
    // yield before every pinned complete table, and inside the elastic
    // class the coldest covered demand per retained byte yields first.
    let plan = plan_expert_residency(
        MemoryPhase::Decode,
        90,
        &uniform_geometry(3),
        &[
            partial_residency(0, 3, 1),
            partial_residency(1, 2, 100),
            complete_layer_residency(2),
        ],
    )
    .expect("current residency fits the planning ceiling");

    assert_eq!(plan.deterministic_release_order, vec![0, 1, 2]);
}

#[test]
fn should_defer_the_whole_table_release_of_a_demoted_complete_layer_to_idle() {
    let demoted_target = ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit;
    assert!(!should_enact_planned_expert_release(
        MemoryPhase::Prefill,
        demoted_target,
    ));
    assert!(!should_enact_planned_expert_release(
        MemoryPhase::GenerationPreparation,
        demoted_target,
    ));
    assert!(!should_enact_planned_expert_release(
        MemoryPhase::Decode,
        demoted_target,
    ));
    assert!(should_enact_planned_expert_release(
        MemoryPhase::Idle,
        demoted_target,
    ));
}
