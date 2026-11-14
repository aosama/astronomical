//! Beside-source coverage for the pure count the reply plan's telemetry
//! reports. Planning itself runs under the serial engine lane against real
//! pagers.

use super::count_layers_in_state;
use crate::{ExpertLayerState, ExpertResidencyPlan, MemoryPhase};

fn plan_with(layer_states: Vec<ExpertLayerState>) -> ExpertResidencyPlan {
    ExpertResidencyPlan {
        phase: MemoryPhase::GenerationPreparation,
        retained_expert_ceiling_bytes: 0,
        complete_layer_targets: Vec::new(),
        layer_states,
        reserved_routed_overlay_bytes: 0,
        expected_preserved_bytes: 0,
        maximum_new_retained_bytes: 0,
        deterministic_release_order: Vec::new(),
        is_low_budget_partial_mode: false,
    }
}

#[test]
fn should_count_the_layers_in_each_state() {
    let residency_plan = plan_with(vec![
        ExpertLayerState::Kept,
        ExpertLayerState::Hot,
        ExpertLayerState::Hot,
        ExpertLayerState::Absent,
    ]);

    assert_eq!(
        count_layers_in_state(&residency_plan, ExpertLayerState::Kept),
        1
    );
    assert_eq!(
        count_layers_in_state(&residency_plan, ExpertLayerState::Hot),
        2
    );
    assert_eq!(
        count_layers_in_state(&residency_plan, ExpertLayerState::Absent),
        1
    );
}

#[test]
fn should_count_no_layer_in_an_empty_plan() {
    let residency_plan = plan_with(Vec::new());

    assert_eq!(
        count_layers_in_state(&residency_plan, ExpertLayerState::Kept),
        0
    );
}

#[test]
fn should_count_no_layer_in_a_state_the_plan_does_not_use() {
    let residency_plan = plan_with(vec![ExpertLayerState::Kept, ExpertLayerState::Kept]);

    assert_eq!(
        count_layers_in_state(&residency_plan, ExpertLayerState::Absent),
        0
    );
}
