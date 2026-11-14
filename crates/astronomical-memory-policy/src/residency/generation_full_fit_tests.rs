use super::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerState, RetainedExpertPageClass,
    kept_layer_indexes_to_seat_before_reply, plan_expert_residency,
    should_commit_mandatory_complete_layer, should_complete_partial_layer_on_prefill_read,
};

fn partial_layer_fixture() -> (Vec<ExpertLayerGeometry>, Vec<CurrentExpertLayerResidency>) {
    let geometries = (0..3)
        .map(|layer_index| ExpertLayerGeometry {
            layer_index,
            complete_layer_payload_bytes: 40,
            expert_payload_bytes: 10,
            expert_capacity: 4,
            experts_per_token: 2,
        })
        .collect();
    let partial_pages = (0..3)
        .map(|layer_index| CurrentExpertLayerResidency {
            layer_index,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 2],
            payload_bytes: 20,
            covered_weighted_demand: 8,
        })
        .collect();
    (geometries, partial_pages)
}

#[test]
fn should_complete_every_partial_page_at_the_exact_generation_boundary() {
    let (geometries, partial_pages) = partial_layer_fixture();
    let published_plan = plan_expert_residency(120, &geometries, &partial_pages)
        .expect("the complete payload fits exactly");
    assert_eq!(
        kept_layer_indexes_to_seat_before_reply(&published_plan, &partial_pages),
        vec![0, 1, 2]
    );
    for layer_index in &published_plan.complete_layer_targets {
        let target = published_plan.layer_states[*layer_index];
        assert!(should_commit_mandatory_complete_layer(1, true, target));
        assert!(should_complete_partial_layer_on_prefill_read(
            true,
            target,
            partial_pages[*layer_index].retained_expert_ids.len(),
            geometries[*layer_index].expert_capacity,
        ));
    }
}

#[test]
fn should_complete_only_the_affordable_partial_pages_one_byte_below_the_generation_boundary() {
    let (geometries, partial_pages) = partial_layer_fixture();
    let plan = plan_expert_residency(119, &geometries, &partial_pages)
        .expect("existing pages fit without funding the complete payload");
    assert_eq!(
        kept_layer_indexes_to_seat_before_reply(&plan, &partial_pages),
        vec![0, 1]
    );
    assert_eq!(plan.layer_states[2], ExpertLayerState::Hot);
}

fn mixed_precision_generation_fixture()
-> (Vec<ExpertLayerGeometry>, Vec<CurrentExpertLayerResidency>) {
    // Mirrors the 21 GB flight shape in miniature: wide layers seated
    // complete, one partial page, and unseated layers whose complete payload
    // is larger than a seated layer's.
    let complete_layer_payload_bytes = [40, 40, 60, 60, 60];
    let geometries = complete_layer_payload_bytes
        .iter()
        .enumerate()
        .map(
            |(layer_index, complete_layer_payload_bytes)| ExpertLayerGeometry {
                layer_index,
                complete_layer_payload_bytes: *complete_layer_payload_bytes,
                expert_payload_bytes: complete_layer_payload_bytes / 4,
                expert_capacity: 4,
                experts_per_token: 2,
            },
        )
        .collect();
    let current_pages = vec![
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 8,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 8,
        },
        CurrentExpertLayerResidency {
            layer_index: 2,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 1, 2],
            payload_bytes: 45,
            covered_weighted_demand: 6,
        },
    ];
    (geometries, current_pages)
}

#[test]
fn should_seat_unretained_complete_layers_from_generation_leftover_below_full_fit() {
    let (geometries, current_pages) = mixed_precision_generation_fixture();
    // Full fit is 260; preserved is 125, so 134 leftover funds the 15-byte
    // partial completion and one 60-byte unretained layer while still
    // reserving the last layer's 30-byte routed floor, not two layers.
    let plan = plan_expert_residency(259, &geometries, &current_pages)
        .expect("generation below full fit preserves and completes");
    assert_eq!(plan.complete_layer_targets, vec![0, 1, 2, 3]);
    assert_eq!(
        kept_layer_indexes_to_seat_before_reply(&plan, &current_pages),
        vec![2, 3]
    );
    assert_eq!(
        plan.layer_states,
        vec![
            ExpertLayerState::Kept,
            ExpertLayerState::Kept,
            ExpertLayerState::Kept,
            ExpertLayerState::Kept,
            ExpertLayerState::Hot,
        ]
    );
}

#[test]
fn should_seat_the_last_unretained_layer_when_the_leftover_equals_its_payload_exactly() {
    let (geometries, current_pages) = mixed_precision_generation_fixture();
    // Preserved 125 + completion 15 + one layer 60 + the 30-byte routed
    // floor still owed to the last unretained layer = 230: the boundary byte
    // count funds exactly two promotions and nothing more.
    let plan =
        plan_expert_residency(230, &geometries, &current_pages).expect("generation boundary plan");
    assert_eq!(
        kept_layer_indexes_to_seat_before_reply(&plan, &current_pages),
        vec![2, 3]
    );
    let one_byte_short = plan_expert_residency(229, &geometries, &current_pages)
        .expect("generation one byte short plan");
    assert_eq!(
        kept_layer_indexes_to_seat_before_reply(&one_byte_short, &current_pages),
        vec![2]
    );
}

#[test]
fn should_not_promote_any_layer_when_preserved_pages_consume_the_generation_ceiling() {
    let (geometries, current_pages) = mixed_precision_generation_fixture();
    let plan = plan_expert_residency(139, &geometries, &current_pages)
        .expect("generation plan with 14 leftover bytes");
    assert!(kept_layer_indexes_to_seat_before_reply(&plan, &current_pages).is_empty());
    assert_eq!(plan.layer_states[2], ExpertLayerState::Hot);
    assert!(
        plan.layer_states[3..]
            .iter()
            .all(|target| { *target == ExpertLayerState::Hot })
    );
}
