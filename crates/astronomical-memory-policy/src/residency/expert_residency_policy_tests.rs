use crate::phase::MemoryPhase;
use crate::residency::ExpertLayerGeometry;
use crate::residency::{
    CurrentExpertLayerResidency, ExpertLayerState, RetainedExpertPageClass,
    decode_warm_floor_eviction_indexes, hot_expert_warm_slot_count, minimum_decode_warm_slot_count,
    plan_expert_residency, should_commit_mandatory_complete_layer,
    should_commit_mandatory_routed_page,
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
fn should_keep_a_whole_layer_that_was_just_read_only_when_it_should_be_kept() {
    assert!(should_commit_mandatory_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
    ));
    assert!(should_commit_mandatory_complete_layer(
        1,
        true,
        ExpertLayerState::Kept,
    ));
    assert!(!should_commit_mandatory_complete_layer(
        2_048,
        true,
        ExpertLayerState::Hot,
    ));
    assert!(!should_commit_mandatory_complete_layer(
        2_048,
        true,
        ExpertLayerState::Absent,
    ));
    assert!(!should_commit_mandatory_complete_layer(
        2_048,
        false,
        ExpertLayerState::Kept,
    ));
}

#[test]
fn should_keep_routed_experts_unless_the_layer_should_be_absent() {
    assert!(should_commit_mandatory_routed_page(
        1,
        true,
        ExpertLayerState::Hot,
        false,
    ));
    assert!(should_commit_mandatory_routed_page(
        1,
        true,
        ExpertLayerState::Kept,
        true,
    ));
    // A layer that should be absent is not refilled behind the plan's back.
    assert!(!should_commit_mandatory_routed_page(
        1,
        true,
        ExpertLayerState::Absent,
        false,
    ));
    assert!(!should_commit_mandatory_routed_page(
        1,
        false,
        ExpertLayerState::Hot,
        false,
    ));
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
    let generation_plan = plan_expert_residency(
        MemoryPhase::GenerationPreparation,
        120,
        &geometries,
        &prefill_routed_pages,
    )
    .expect("generation leftover should keep every prefill routed page");

    assert_eq!(generation_plan.complete_layer_targets, vec![0, 1, 2]);
    assert_eq!(generation_plan.expected_preserved_bytes, 60);
    assert_eq!(generation_plan.maximum_new_retained_bytes, 60);
    assert!(
        generation_plan
            .layer_states
            .iter()
            .all(|target| { *target == ExpertLayerState::Kept })
    );
}
