//! Occupancy counts: whole layers are counted, partly held ones are not.

use super::seated_complete_layer_count;
use crate::residency::{CurrentExpertLayerResidency, RetainedExpertPageClass};

fn residency(layer_index: usize, class: RetainedExpertPageClass) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index,
        class,
        retained_expert_ids: vec![0],
        payload_bytes: 100,
        covered_weighted_demand: 0,
    }
}

#[test]
fn should_count_only_layers_held_whole() {
    let current_residencies = [
        residency(0, RetainedExpertPageClass::StableCompleteLayer),
        residency(1, RetainedExpertPageClass::ElasticRoutedExperts),
        residency(2, RetainedExpertPageClass::StableCompleteLayer),
    ];
    assert_eq!(seated_complete_layer_count(&current_residencies), 2);
}

#[test]
fn should_count_nothing_when_nothing_is_held() {
    assert_eq!(seated_complete_layer_count(&[]), 0);
}
