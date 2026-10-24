use super::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    RetainedExpertPageClass, complete_layer_indexes_required_before_decode, plan_expert_residency,
    publish_request_stable_residency_plan, should_commit_mandatory_complete_layer,
    should_complete_partial_layer_on_prefill_read,
};
use crate::phase::MemoryPhase;

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
    for phase in [MemoryPhase::GenerationPreparation, MemoryPhase::Decode] {
        let candidate = plan_expert_residency(phase, 120, &geometries, &partial_pages)
            .expect("the complete payload fits exactly");
        let (prefill_contract, published_plan) = publish_request_stable_residency_plan(
            phase,
            None,
            candidate,
            &partial_pages,
            0,
            &geometries,
        );
        assert!(prefill_contract.is_none());
        assert_eq!(
            complete_layer_indexes_required_before_decode(&published_plan),
            vec![0, 1, 2]
        );
        for layer_index in &published_plan.complete_layer_targets {
            let target = Some(published_plan.layer_targets[*layer_index]);
            assert!(should_commit_mandatory_complete_layer(1, true, target));
            assert!(should_complete_partial_layer_on_prefill_read(
                true,
                target,
                partial_pages[*layer_index].retained_expert_ids.len(),
                geometries[*layer_index].expert_capacity,
            ));
        }
        assert_eq!(published_plan.expected_preserved_bytes, 60);
        assert_eq!(published_plan.maximum_new_retained_bytes, 60);
        assert_eq!(
            published_plan.expected_preserved_bytes + published_plan.maximum_new_retained_bytes,
            120
        );
    }
}

#[test]
fn should_preserve_only_affordable_pages_one_byte_below_the_generation_boundary() {
    let (geometries, partial_pages) = partial_layer_fixture();
    for phase in [MemoryPhase::GenerationPreparation, MemoryPhase::Decode] {
        let plan = plan_expert_residency(phase, 119, &geometries, &partial_pages)
            .expect("existing pages fit without funding the complete payload");
        assert!(complete_layer_indexes_required_before_decode(&plan).is_empty());
        assert_eq!(plan.expected_preserved_bytes, 60);
        assert!(
            plan.layer_targets
                .iter()
                .all(|target| { *target == ExpertLayerResidencyTarget::PreservePartial })
        );
        assert_eq!(
            plan.expected_preserved_bytes + plan.maximum_new_retained_bytes,
            119
        );
    }
}

#[test]
fn should_not_read_already_complete_layers_again_at_the_generation_boundary() {
    let (geometries, mut partial_pages) = partial_layer_fixture();
    partial_pages[1].class = RetainedExpertPageClass::StableCompleteLayer;
    partial_pages[1].retained_expert_ids = vec![0, 1, 2, 3];
    partial_pages[1].payload_bytes = 40;
    let plan = plan_expert_residency(
        MemoryPhase::GenerationPreparation,
        120,
        &geometries,
        &partial_pages,
    )
    .expect("mixed complete and partial residency fits");
    assert_eq!(
        complete_layer_indexes_required_before_decode(&plan),
        vec![0, 2]
    );
    assert_eq!(
        plan.layer_targets[1],
        ExpertLayerResidencyTarget::PreserveComplete
    );
    assert_eq!(plan.expected_preserved_bytes, 80);
    assert_eq!(plan.maximum_new_retained_bytes, 40);
}

#[test]
fn should_fund_complete_decode_residency_for_the_observed_sparse_model_geometry() {
    const EXPERT_PAYLOAD_BYTES: u64 = 3_342_336;
    const EXPERT_CAPACITY: usize = 256;
    const SPARSE_LAYER_COUNT: usize = 40;
    let geometries = (0..SPARSE_LAYER_COUNT)
        .map(|layer_index| ExpertLayerGeometry {
            layer_index,
            complete_layer_payload_bytes: EXPERT_PAYLOAD_BYTES * EXPERT_CAPACITY as u64,
            expert_payload_bytes: EXPERT_PAYLOAD_BYTES,
            expert_capacity: EXPERT_CAPACITY,
            experts_per_token: 8,
        })
        .collect::<Vec<_>>();
    let partial_pages = (0..37)
        .map(|layer_index| {
            let retained_expert_count = if layer_index < 26 { 179 } else { 178 };
            CurrentExpertLayerResidency {
                layer_index,
                class: RetainedExpertPageClass::ElasticRoutedExperts,
                retained_expert_ids: (0..retained_expert_count).collect(),
                payload_bytes: retained_expert_count as u64 * EXPERT_PAYLOAD_BYTES,
                covered_weighted_demand: 1,
            }
        })
        .collect::<Vec<_>>();
    let complete_payload_bytes = geometries
        .iter()
        .map(|geometry| geometry.complete_layer_payload_bytes)
        .sum::<u64>();
    let plan = plan_expert_residency(
        MemoryPhase::GenerationPreparation,
        35_024_537_344,
        &geometries,
        &partial_pages,
    )
    .expect("the observed model geometry fits the decode entitlement");
    assert_eq!(
        complete_layer_indexes_required_before_decode(&plan),
        (0..SPARSE_LAYER_COUNT).collect::<Vec<_>>(),
        "existing partial pages must not strand the remaining decode entitlement"
    );
    assert_eq!(
        plan.expected_preserved_bytes + plan.maximum_new_retained_bytes,
        complete_payload_bytes
    );
    assert_eq!(plan.expected_preserved_bytes, 22_099_525_632);
    assert_eq!(plan.maximum_new_retained_bytes, 12_125_995_008);
}
