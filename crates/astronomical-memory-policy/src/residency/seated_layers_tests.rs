//! Occupancy counts: whole layers are counted, partly held ones are not.

use super::{holds_every_layer_whole, seated_complete_layer_count};
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

#[test]
fn should_report_every_layer_whole_when_each_is_a_complete_layer() {
    let current_residencies = [
        residency(0, RetainedExpertPageClass::StableCompleteLayer),
        residency(1, RetainedExpertPageClass::StableCompleteLayer),
    ];
    assert!(holds_every_layer_whole(&current_residencies, 2));
}

#[test]
fn should_not_report_every_layer_whole_when_one_is_partly_held() {
    let current_residencies = [
        residency(0, RetainedExpertPageClass::StableCompleteLayer),
        residency(1, RetainedExpertPageClass::ElasticRoutedExperts),
    ];
    assert!(!holds_every_layer_whole(&current_residencies, 2));
}

#[test]
fn should_not_report_every_layer_whole_when_a_layer_holds_nothing() {
    let current_residencies = [residency(0, RetainedExpertPageClass::StableCompleteLayer)];
    assert!(!holds_every_layer_whole(&current_residencies, 2));
}

#[test]
fn should_not_report_every_layer_whole_for_a_model_without_expert_layers() {
    assert!(!holds_every_layer_whole(&[], 0));
}
