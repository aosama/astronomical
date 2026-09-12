//! Previous-token prefetch admission: leftover slots only, never eviction.

use astronomical_model_serving::{
    PreviousTokenPrefetchCandidate, PreviousTokenPrefetchLayerCapacity,
    plan_previous_token_prefetch,
};

fn candidate(
    layer_index: usize,
    expert_id: usize,
    is_already_resident: bool,
) -> PreviousTokenPrefetchCandidate {
    PreviousTokenPrefetchCandidate {
        layer_index,
        expert_id,
        payload_bytes: 8,
        is_already_resident,
    }
}

fn layer_capacity(
    layer_index: usize,
    free_slot_count: usize,
) -> PreviousTokenPrefetchLayerCapacity {
    PreviousTokenPrefetchLayerCapacity {
        layer_index,
        free_slot_count,
    }
}

#[test]
fn drops_every_miss_when_the_layer_has_no_free_slot() {
    let plan = plan_previous_token_prefetch(
        &[candidate(0, 1, false), candidate(0, 2, false)],
        &[layer_capacity(0, 0)],
    );
    assert!(
        plan.experts_to_retain.is_empty(),
        "a full layer must not displace a demanded page to keep a previous-token expert"
    );
    assert_eq!(plan.dropped_for_capacity_count, 2);
    assert_eq!(plan.skipped_already_resident_count, 0);
}

#[test]
fn fills_free_slots_then_drops_the_overflow_instead_of_evicting() {
    let plan = plan_previous_token_prefetch(
        &[
            candidate(0, 1, false),
            candidate(0, 2, false),
            candidate(0, 3, false),
        ],
        &[layer_capacity(0, 2)],
    );
    assert_eq!(plan.experts_to_retain, vec![(0, 1), (0, 2)]);
    assert_eq!(plan.dropped_for_capacity_count, 1);
}

#[test]
fn skips_experts_the_warm_table_already_holds() {
    let plan = plan_previous_token_prefetch(
        &[candidate(1, 4, true), candidate(1, 5, false)],
        &[layer_capacity(1, 1)],
    );
    assert_eq!(plan.experts_to_retain, vec![(1, 5)]);
    assert_eq!(plan.skipped_already_resident_count, 1);
    assert_eq!(plan.dropped_for_capacity_count, 0);
}

#[test]
fn does_not_borrow_free_slots_from_another_layer() {
    let plan = plan_previous_token_prefetch(
        &[candidate(0, 1, false), candidate(1, 2, false)],
        &[layer_capacity(0, 0), layer_capacity(1, 1)],
    );
    assert_eq!(plan.experts_to_retain, vec![(1, 2)]);
    assert_eq!(plan.dropped_for_capacity_count, 1);
}

#[test]
fn drops_a_zero_payload_expert_rather_than_planning_a_empty_retain() {
    let plan = plan_previous_token_prefetch(
        &[PreviousTokenPrefetchCandidate {
            layer_index: 0,
            expert_id: 9,
            payload_bytes: 0,
            is_already_resident: false,
        }],
        &[layer_capacity(0, 4)],
    );
    assert!(plan.experts_to_retain.is_empty());
    assert_eq!(plan.dropped_for_capacity_count, 1);
}
