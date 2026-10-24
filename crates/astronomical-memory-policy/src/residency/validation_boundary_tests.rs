//! Error-arm battery for residency plan input validation: every structural
//! defect the fail-closed validator names, driven through the planner entry
//! point exactly as production callers reach them.

use crate::phase::MemoryPhase;
use crate::residency::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertResidencyPlanError,
    RetainedExpertPageClass, plan_expert_residency,
};

fn valid_geometry(layer_count: usize) -> Vec<ExpertLayerGeometry> {
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

fn complete_residency(layer_index: usize) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index,
        class: RetainedExpertPageClass::StableCompleteLayer,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes: 40,
        covered_weighted_demand: 0,
    }
}

fn plan_with(
    geometries: &[ExpertLayerGeometry],
    current_residencies: &[CurrentExpertLayerResidency],
) -> Result<crate::residency::ExpertResidencyPlan, ExpertResidencyPlanError> {
    plan_expert_residency(MemoryPhase::Prefill, 200, geometries, current_residencies)
}

#[test]
fn should_reject_a_non_contiguous_layer_index() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut geometries = valid_geometry(2);
    geometries[0].layer_index = 1;

    assert!(matches!(
        plan_with(&geometries, &[]),
        Err(ExpertResidencyPlanError::NonContiguousLayerIndex {
            expected_layer_index: 0,
            actual_layer_index: 1,
        })
    ));
}

#[test]
fn should_reject_every_zero_geometry_field_independently() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    for zeroed_field in [
        (0_u64, 10_usize, 40_u64, 2_usize),
        (10, 0, 40, 2),
        (10, 4, 0, 2),
        (10, 4, 40, 0),
    ] {
        let geometries = vec![ExpertLayerGeometry {
            layer_index: 0,
            // (expert payload, expert capacity, complete payload, experts per token)
            expert_payload_bytes: zeroed_field.0,
            expert_capacity: zeroed_field.1,
            complete_layer_payload_bytes: zeroed_field.2,
            experts_per_token: zeroed_field.3,
        }];
        assert!(
            matches!(
                plan_with(&geometries, &[]),
                Err(ExpertResidencyPlanError::ZeroGeometry { layer_index: 0 })
            ),
            "zeroed field {:?} must fail closed",
            zeroed_field
        );
    }
}

#[test]
fn should_reject_a_complete_payload_that_disagrees_with_the_expert_geometry() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut geometries = valid_geometry(1);
    geometries[0].complete_layer_payload_bytes = 41;

    assert!(matches!(
        plan_with(&geometries, &[]),
        Err(ExpertResidencyPlanError::InconsistentCompletePayload { layer_index: 0 })
    ));
}

#[test]
fn should_reject_duplicate_or_descending_current_layers() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let geometries = valid_geometry(3);
    assert!(matches!(
        plan_with(&geometries, &[complete_residency(1), complete_residency(1)]),
        Err(ExpertResidencyPlanError::DuplicateOrUnorderedCurrentLayer { layer_index: 1 })
    ));
    assert!(matches!(
        plan_with(&geometries, &[complete_residency(2), complete_residency(1)]),
        Err(ExpertResidencyPlanError::DuplicateOrUnorderedCurrentLayer { layer_index: 1 })
    ));
}

#[test]
fn should_reject_a_current_layer_beyond_the_geometry() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert!(matches!(
        plan_with(&valid_geometry(1), &[complete_residency(3)]),
        Err(ExpertResidencyPlanError::CurrentLayerOutOfRange { layer_index: 3 })
    ));
}

#[test]
fn should_reject_every_invalid_retained_expert_identifier_shape() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let geometries = valid_geometry(1);
    for invalid_ids in [vec![], vec![1, 0], vec![0, 0], vec![0, 4]] {
        let retained_count = invalid_ids.len();
        let residency = CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            payload_bytes: 10 * retained_count as u64,
            retained_expert_ids: invalid_ids,
            covered_weighted_demand: 0,
        };
        assert!(
            matches!(
                plan_with(&geometries, &[residency]),
                Err(ExpertResidencyPlanError::InvalidRetainedExpertIds { layer_index: 0 })
            ),
            "identifier shape with {retained_count} ids must fail closed"
        );
    }
}

#[test]
fn should_reject_a_current_payload_or_class_that_disagrees_with_retention() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let geometries = valid_geometry(1);

    // Payload byte count disagrees with the retained identifier count.
    let wrong_payload = CurrentExpertLayerResidency {
        payload_bytes: 50,
        ..complete_residency(0)
    };
    assert!(matches!(
        plan_with(&geometries, &[wrong_payload]),
        Err(ExpertResidencyPlanError::InconsistentCurrentPayload { layer_index: 0, .. })
    ));

    // A complete-class layer must retain the layer's whole capacity.
    let shrunken_complete = CurrentExpertLayerResidency {
        retained_expert_ids: vec![0, 1, 2],
        payload_bytes: 30,
        ..complete_residency(0)
    };
    assert!(matches!(
        plan_with(&geometries, &[shrunken_complete]),
        Err(ExpertResidencyPlanError::InconsistentCurrentPayload { layer_index: 0, .. })
    ));

    // An elastic-class layer at full capacity is complete, not elastic.
    let full_elastic = CurrentExpertLayerResidency {
        class: RetainedExpertPageClass::ElasticRoutedExperts,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes: 40,
        covered_weighted_demand: 0,
        layer_index: 0,
    };
    assert!(matches!(
        plan_with(&geometries, &[full_elastic]),
        Err(ExpertResidencyPlanError::InconsistentCurrentPayload { layer_index: 0, .. })
    ));
}

#[test]
fn should_reject_current_ownership_one_byte_over_the_planning_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let plan_error = plan_expert_residency(
        MemoryPhase::Prefill,
        39,
        &valid_geometry(1),
        &[complete_residency(0)],
    )
    .expect_err("ownership above the ceiling must fail closed");

    assert_eq!(
        plan_error,
        ExpertResidencyPlanError::CurrentResidencyExceedsCeiling
    );
}

#[test]
fn should_reject_layer_facts_whose_current_payload_sum_overflows() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // Two layers each holding one expert of u64::MAX bytes: the sum
    // overflows before any plan arithmetic can begin.
    let geometries = vec![
        ExpertLayerGeometry {
            layer_index: 0,
            complete_layer_payload_bytes: u64::MAX,
            expert_payload_bytes: u64::MAX,
            expert_capacity: 1,
            experts_per_token: 1,
        },
        ExpertLayerGeometry {
            layer_index: 1,
            complete_layer_payload_bytes: u64::MAX,
            expert_payload_bytes: u64::MAX,
            expert_capacity: 1,
            experts_per_token: 1,
        },
    ];
    let residencies = [
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0],
            payload_bytes: u64::MAX,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0],
            payload_bytes: u64::MAX,
            covered_weighted_demand: 0,
        },
    ];

    assert!(matches!(
        plan_with(&geometries, &residencies),
        Err(ExpertResidencyPlanError::ByteCountOverflow)
    ));
}
