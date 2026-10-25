use crate::budget::adaptive_growth::AdaptiveRamGrowthExecutionProfile;
use crate::phase::MemoryPhase;
use crate::recovery::{ForwardRecoveryDecision, ForwardRecoveryRequirements};
use crate::residency::ExpertLayerGeometry;
use crate::residency::{
    CurrentExpertLayerResidency, ExpertLayerResidencyTarget, ExpertResidencyPlanError,
    RequestExpertLayerRole, RequestExpertResidency, RetainedExpertPageClass,
    decode_warm_floor_eviction_indexes, hot_expert_warm_slot_count, minimum_decode_warm_slot_count,
    plan_expert_residency, publish_request_stable_residency_plan,
    retained_complete_layer_ceiling_after_prefill_budget_refresh,
    retained_resident_ceiling_after_budget_refresh, should_commit_mandatory_complete_layer,
    should_commit_mandatory_routed_page, should_enact_planned_expert_release,
};

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

#[test]
fn should_keep_complete_layers_during_prefill_even_when_the_plan_names_a_release() {
    let geometries = (0..3)
        .map(|layer_index| ExpertLayerGeometry {
            layer_index,
            complete_layer_payload_bytes: 40,
            expert_payload_bytes: 10,
            expert_capacity: 4,
            experts_per_token: 2,
        })
        .collect::<Vec<_>>();
    let plan = plan_expert_residency(
        MemoryPhase::Prefill,
        60,
        &geometries,
        &[CurrentExpertLayerResidency {
            layer_index: 2,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        }],
    )
    .expect("routed floors should consume the full ceiling");

    assert_eq!(
        plan.layer_targets[2],
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
    );
    assert!(!should_enact_planned_expert_release(
        MemoryPhase::Prefill,
        plan.layer_targets[2],
    ));
    assert!(!should_enact_planned_expert_release(
        MemoryPhase::GenerationPreparation,
        plan.layer_targets[2],
    ));
    assert!(should_enact_planned_expert_release(
        MemoryPhase::Idle,
        plan.layer_targets[2],
    ));
}

#[test]
fn should_seat_complete_layers_on_mandatory_prefill_reads() {
    assert!(should_commit_mandatory_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
    ));
    assert!(should_commit_mandatory_complete_layer(
        1,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
    ));
    // Streamed-role layers admit too: the mandatory read already paid the
    // complete-layer cost, and keeping it is reuse rather than pinning.
    assert!(should_commit_mandatory_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead),
    ));
    assert!(!should_commit_mandatory_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::StreamOperationLocal),
    ));
    // The active plan is the phase's will for the layer: a layer the plan
    // streams operation-local keeps streaming, and a layer the plan is
    // releasing is not refilled behind the plan's back.
    assert!(!should_commit_mandatory_routed_page(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::StreamOperationLocal),
        false,
    ));
    assert!(!should_commit_mandatory_routed_page(
        1,
        true,
        Some(ExpertLayerResidencyTarget::ReleasePartial),
        false,
    ));
    assert!(!should_commit_mandatory_routed_page(
        1,
        true,
        Some(ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit),
        false,
    ));
    assert!(should_commit_mandatory_routed_page(
        1,
        true,
        Some(ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead),
        true,
    ));
    assert!(should_commit_mandatory_routed_page(
        1,
        true,
        Some(ExpertLayerResidencyTarget::PreservePartial),
        false,
    ));
    assert!(should_commit_mandatory_routed_page(1, true, None, false));
    assert!(!should_commit_mandatory_routed_page(
        2_048,
        false,
        Some(ExpertLayerResidencyTarget::StreamOperationLocal),
        true,
    ));
    assert!(!should_commit_mandatory_routed_page(1, false, None, false));
}

#[test]
fn should_cap_hot_expert_warm_tables_at_the_layer_expert_count() {
    // Issue #514: leftover decode entitlement is the economic cap, applied
    // by the caller. This function is only the structural cap.
    assert_eq!(hot_expert_warm_slot_count(512), 512);
    assert_eq!(hot_expert_warm_slot_count(4), 4);
    assert_eq!(hot_expert_warm_slot_count(6), 6);
}

#[test]
fn should_floor_decode_warm_slots_at_one_token_routing_per_layer() {
    // The floor survives complete-layer seating: two tokens' routing per
    // sparse layer is the minimum at which consecutive tokens' overlapping
    // routes stay resident and mixed serving has a table to land in. Zero
    // only when the caller passes zero, which no Qwen3.5 configuration does.
    assert_eq!(minimum_decode_warm_slot_count(8), 16);
    assert_eq!(minimum_decode_warm_slot_count(2), 4);
    assert_eq!(minimum_decode_warm_slot_count(1), 2);
    assert_eq!(minimum_decode_warm_slot_count(0), 1);
}

fn complete_layer_residency(
    layer_index: usize,
    payload_bytes: u64,
    covered_weighted_demand: u64,
) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index,
        class: RetainedExpertPageClass::StableCompleteLayer,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes,
        covered_weighted_demand,
    }
}

#[test]
fn should_evict_nothing_when_the_warm_budget_already_funds_the_floor() {
    // Five sparse layers, one seated complete layer, and a warm budget that
    // already covers four unseated layers at two slots each: nothing moves.
    let layer_residencies = vec![complete_layer_residency(0, 40, 7)];
    assert!(
        decode_warm_floor_eviction_indexes(&layer_residencies, 4_u64 * 10 * 2, 10, 2, 5).is_empty()
    );
}

#[test]
fn should_evict_nothing_for_a_fully_seated_model() {
    // Every sparse layer holds a complete table: there is no unseated layer
    // to warm, so the floor is zero and the safety property for models whose
    // residency is complete-layer-based holds — nothing is ever evicted.
    let layer_residencies: Vec<_> = (0..5_u64)
        .map(|layer_index| {
            complete_layer_residency(
                usize::try_from(layer_index).expect("the layer index fits usize"),
                40,
                layer_index,
            )
        })
        .collect();
    assert!(decode_warm_floor_eviction_indexes(&layer_residencies, 0, 10, 2, 5).is_empty());
}

#[test]
fn should_evict_coldest_complete_layers_until_the_recomputed_floor_fits() {
    // Two complete layers (layer 3 colder than layer 1), five sparse layers
    // total, per-expert payload 10, two-slot floor. With zero budget the
    // floor for four unseated layers is 4 × 10 × 2 = 80; evicting layer 3
    // (40 bytes) raises the floor to 5 unseated × 10 × 2 = 100 while freeing
    // 40 — still short — so layer 1 follows, freeing another 40 against a
    // floor that can no longer grow (no complete layers remain).
    let layer_residencies = vec![
        complete_layer_residency(1, 40, 100),
        complete_layer_residency(3, 40, 2),
    ];
    assert_eq!(
        decode_warm_floor_eviction_indexes(&layer_residencies, 0, 10, 2, 5),
        vec![3, 1]
    );
    // A budget that the single coldest eviction covers stops there: floor 80,
    // budget 45 + freed 40 = 85 ≥ 80 after evicting layer 3 only.
    assert_eq!(
        decode_warm_floor_eviction_indexes(&layer_residencies, 45, 10, 2, 5),
        vec![3]
    );
}

#[test]
fn should_authorize_one_same_chunk_retry_for_paged_execution_after_reclamation() {
    let resident_retry = ForwardRecoveryRequirements {
        stable_active_memory_bytes: 900,
        active_memory_bytes_at_failure: 950,
        attempted_allocation_bytes: 100,
        observed_transient_high_water_bytes: 25,
        retained_expert_payload_bytes_before_reclamation: 200,
        retained_expert_payload_bytes_after_reclamation: 100,
        active_memory_ceiling_bytes: 1_000,
        has_already_retried_after_reclamation: false,
        execution_profile: AdaptiveRamGrowthExecutionProfile::Resident,
    };
    let paged_retry = ForwardRecoveryRequirements {
        execution_profile: AdaptiveRamGrowthExecutionProfile::Paged,
        ..resident_retry
    };

    assert!(matches!(
        resident_retry.decide(),
        ForwardRecoveryDecision::Retry { .. }
    ));
    assert!(matches!(
        paged_retry.decide(),
        ForwardRecoveryDecision::Retry { .. }
    ));
    // The retry is bounded exactly as it is for resident execution: a second
    // capacity failure on the retried chunk must not loop.
    let paged_second_retry = ForwardRecoveryRequirements {
        has_already_retried_after_reclamation: true,
        ..paged_retry
    };
    assert!(matches!(
        paged_second_retry.decide(),
        ForwardRecoveryDecision::Reject { .. }
    ));
}

#[test]
fn should_keep_already_seated_complete_layers_when_leftover_budget_tightens() {
    let seated_complete_layer_payload_bytes = 80;
    let tighter_leftover_expert_budget_bytes = 70;
    let richer_leftover_expert_budget_bytes = 90;
    assert_eq!(
        retained_complete_layer_ceiling_after_prefill_budget_refresh(
            tighter_leftover_expert_budget_bytes,
            seated_complete_layer_payload_bytes,
        ),
        seated_complete_layer_payload_bytes,
        "learned context reserve must not evict a complete layer this request already seated"
    );
    assert_eq!(
        retained_complete_layer_ceiling_after_prefill_budget_refresh(
            richer_leftover_expert_budget_bytes,
            seated_complete_layer_payload_bytes,
        ),
        richer_leftover_expert_budget_bytes,
        "a richer leftover budget must still be able to seat additional complete layers"
    );
    assert_eq!(
        retained_complete_layer_ceiling_after_prefill_budget_refresh(0, 0),
        0
    );
}

#[test]
fn should_keep_all_resident_expert_pages_when_learned_budget_tightens_between_requests() {
    let resident_complete_and_warm_table_payload_bytes = 100;
    let tighter_leftover_expert_budget_bytes = 80;
    let richer_leftover_expert_budget_bytes = 120;
    assert_eq!(
        retained_resident_ceiling_after_budget_refresh(
            tighter_leftover_expert_budget_bytes,
            resident_complete_and_warm_table_payload_bytes,
        ),
        resident_complete_and_warm_table_payload_bytes,
        "learned context-reserve growth must not evict warm tables a previous \
         request already paid to read (issue #955)"
    );
    assert_eq!(
        retained_resident_ceiling_after_budget_refresh(
            richer_leftover_expert_budget_bytes,
            resident_complete_and_warm_table_payload_bytes,
        ),
        richer_leftover_expert_budget_bytes,
        "a richer leftover budget must still admit additional expert pages"
    );
    assert_eq!(retained_resident_ceiling_after_budget_refresh(0, 0), 0);
}

#[test]
fn should_plan_prefill_with_the_floored_ceiling_when_seated_layers_exceed_leftover() {
    let geometries = uniform_geometry(3);
    let seated_complete_layers = [
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
    ];
    let leftover_expert_budget_bytes = 70;
    let seated_complete_layer_payload_bytes = 80;
    assert!(
        matches!(
            plan_expert_residency(
                MemoryPhase::Prefill,
                leftover_expert_budget_bytes,
                &geometries,
                &seated_complete_layers,
            ),
            Err(ExpertResidencyPlanError::CurrentResidencyExceedsCeiling),
        ),
        "planning against the tighter leftover number must not silently drop seated layers"
    );
    let floored_ceiling_bytes = retained_complete_layer_ceiling_after_prefill_budget_refresh(
        leftover_expert_budget_bytes,
        seated_complete_layer_payload_bytes,
    );
    plan_expert_residency(
        MemoryPhase::Prefill,
        floored_ceiling_bytes,
        &geometries,
        &seated_complete_layers,
    )
    .expect("the floored Prefill ceiling must accept already-seated complete layers");
    plan_expert_residency(
        MemoryPhase::Decode,
        floored_ceiling_bytes,
        &geometries,
        &seated_complete_layers,
    )
    .expect("the floored Decode ceiling must accept already-seated complete layers");
    plan_expert_residency(
        MemoryPhase::GenerationPreparation,
        floored_ceiling_bytes,
        &geometries,
        &seated_complete_layers,
    )
    .expect("the floored GenerationPreparation ceiling must accept already-seated complete layers");
}

#[test]
fn should_keep_the_opening_prefill_pin_set_when_later_leftover_wants_more_layers() {
    let geometries = uniform_geometry(3);
    let opening_candidate = plan_expert_residency(MemoryPhase::Prefill, 100, &geometries, &[])
        .expect("two complete layers should fit the opening leftover");
    let (opened_residency, opened_plan) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        None,
        opening_candidate,
        &[],
        0,
        &geometries,
    );
    let opened_residency = opened_residency.expect("Prefill must open a request contract");
    assert_eq!(
        opened_residency.layer_role(0),
        Some(RequestExpertLayerRole::PinnedComplete)
    );
    assert_eq!(
        opened_residency.layer_role(2),
        Some(RequestExpertLayerRole::Streamed)
    );
    assert_eq!(
        opened_plan.layer_targets[2],
        // The pinned claims (two unseated complete layers, 80 bytes) leave
        // 20 bytes of the composed leftover; a 40-byte complete page can
        // never seat there, so the layer must resolve to operation-local
        // streaming instead of an authorized offer the budget refuses.
        ExpertLayerResidencyTarget::StreamOperationLocal
    );

    let richer_candidate = plan_expert_residency(MemoryPhase::Prefill, 120, &geometries, &[])
        .expect("the full model should fit a richer leftover");
    let (_continued_residency, continued_plan) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        Some(&opened_residency),
        richer_candidate,
        &[],
        0,
        &geometries,
    );

    assert_eq!(continued_plan.complete_layer_targets, vec![0, 1]);
    assert_eq!(
        continued_plan.layer_targets[2],
        // The richer leftover holds the pinned claims plus one complete
        // page (80 + 40 = 120), so the streamed layer keeps admit-on-read
        // exactly while the composed budget can seat it.
        ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead
    );
}

#[test]
fn should_keep_opening_prefill_pins_when_later_leftover_is_tighter_without_capacity_failure() {
    let geometries = uniform_geometry(3);
    let opening_candidate = plan_expert_residency(MemoryPhase::Prefill, 100, &geometries, &[])
        .expect("two complete layers should fit the opening leftover");
    let opened_residency = RequestExpertResidency::open_prefill(&opening_candidate);
    assert_eq!(opened_residency.pinned_complete_layer_indexes(), vec![0, 1]);

    let tighter_candidate = plan_expert_residency(MemoryPhase::Prefill, 70, &geometries, &[])
        .expect("one complete layer should still fit a tighter leftover");
    let (continued_residency, continued_plan) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        Some(&opened_residency),
        tighter_candidate,
        &[],
        0,
        &geometries,
    );
    let continued_residency = continued_residency.expect("Prefill must keep the request contract");

    assert_eq!(
        continued_residency.pinned_complete_layer_indexes(),
        vec![0, 1],
        "learned leftover tightening is not a capacity failure and must not \
         unpin layers at the contract level"
    );
    // The plan-level claims respect the tighter 70-byte ceiling: pin 0 funds
    // its 40-byte claim, pin 1 does not (40 more would claim 80 of 70), so
    // it falls through to streamed handling instead of promising a
    // promotion the budget cannot honor — the over-claim shape that left
    // 26.95 GB unused on the 2026-10-23 Ornith flight.
    assert_eq!(continued_plan.complete_layer_targets, vec![0]);
    assert_eq!(
        continued_plan.layer_targets[0],
        ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
    );
    assert_eq!(
        continued_plan.layer_targets[1],
        ExpertLayerResidencyTarget::StreamOperationLocal
    );
}

#[test]
fn should_not_re_pin_a_layer_after_a_prefill_capacity_failure() {
    let geometries = uniform_geometry(3);
    let opening_candidate = plan_expert_residency(MemoryPhase::Prefill, 100, &geometries, &[])
        .expect("two complete layers should fit");
    let opened_residency = RequestExpertResidency::open_prefill(&opening_candidate);
    let shrunk_residency = opened_residency.shrink_after_capacity_failure(40, &geometries);
    assert_eq!(
        shrunk_residency.layer_role(1),
        Some(RequestExpertLayerRole::DemotedByCapacityFailure)
    );

    let richer_candidate = plan_expert_residency(MemoryPhase::Prefill, 120, &geometries, &[])
        .expect("leftover after failure still wants every layer");
    let (_published_residency, published_plan) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        Some(&shrunk_residency),
        richer_candidate,
        &[],
        0,
        &geometries,
    );

    assert_eq!(published_plan.complete_layer_targets, vec![0]);
    assert_eq!(
        published_plan.layer_targets[1],
        // A capacity-evicted layer never re-seats within the request: the
        // bytes a recovery reclaimed return to the exact pressure that
        // evicted them the moment a retry re-admits the layer (issue
        // #1278), so the layer streams operation-local until the request
        // ends.
        ExpertLayerResidencyTarget::StreamOperationLocal
    );
}

#[test]
fn should_promote_every_partial_prefill_page_when_generation_payload_fits() {
    let geometries = uniform_geometry(3);
    let prefill_routed_pages = [
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 1],
            payload_bytes: 20,
            covered_weighted_demand: 8,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 2],
            payload_bytes: 20,
            covered_weighted_demand: 6,
        },
        CurrentExpertLayerResidency {
            layer_index: 2,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![1, 3],
            payload_bytes: 20,
            covered_weighted_demand: 4,
        },
    ];
    let generation_candidate = plan_expert_residency(
        MemoryPhase::GenerationPreparation,
        120,
        &geometries,
        &prefill_routed_pages,
    )
    .expect("generation leftover should keep every prefill routed page");
    let (generation_residency, generation_plan) = publish_request_stable_residency_plan(
        MemoryPhase::GenerationPreparation,
        None,
        generation_candidate,
        &prefill_routed_pages,
        0,
        &geometries,
    );

    assert!(generation_residency.is_none());
    assert_eq!(generation_plan.complete_layer_targets, vec![0, 1, 2]);
    assert_eq!(generation_plan.expected_preserved_bytes, 60);
    assert_eq!(generation_plan.maximum_new_retained_bytes, 60);
    assert!(
        generation_plan.layer_targets.iter().all(|target| {
            *target == ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
        })
    );
}
