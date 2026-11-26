use crate::admission::complete_residency::{
    CompleteResidencyDecision, CompleteResidencyRequirements,
};
use crate::placement::engine_placement::EnginePlacement;

fn startup_complete_residency_decision_for_ceiling(
    active_memory_ceiling_bytes: u64,
) -> CompleteResidencyDecision {
    CompleteResidencyRequirements {
        current_active_memory_bytes: 1_000,
        retained_paged_expert_payload_bytes: 0,
        complete_expert_payload_bytes: 2_000,
        required_headroom_bytes: 500,
        active_memory_ceiling_bytes,
    }
    .decide()
}

#[test]
fn should_map_admitted_startup_decision_to_complete_residency_fits() {
    let decision = startup_complete_residency_decision_for_ceiling(4_000);
    assert!(matches!(decision, CompleteResidencyDecision::Admit { .. }));
    assert_eq!(
        EnginePlacement::from_startup_complete_residency_decision(decision),
        Ok(EnginePlacement::CompleteResidencyFits)
    );
}

#[test]
fn should_map_startup_shortfall_to_complete_residency_does_not_fit() {
    let decision = startup_complete_residency_decision_for_ceiling(3_000);
    let CompleteResidencyDecision::DoesNotFit {
        shortfall_bytes, ..
    } = decision
    else {
        panic!("a 3,000-byte ceiling below the 3,500-byte requirement must not fit");
    };
    assert_eq!(shortfall_bytes, 500);
    assert_eq!(
        EnginePlacement::from_startup_complete_residency_decision(decision),
        Ok(EnginePlacement::CompleteResidencyDoesNotFit {
            shortfall_bytes: 500
        })
    );
}

#[test]
fn should_return_the_observation_error_for_invalid_observation_startup_decision() {
    let invalid_observation_decision = CompleteResidencyRequirements {
        // Retained paged payload exceeding live active memory is an
        // inconsistent observation, the one input decide() rejects outright.
        current_active_memory_bytes: 0,
        retained_paged_expert_payload_bytes: 1,
        complete_expert_payload_bytes: 1_000,
        required_headroom_bytes: 1,
        active_memory_ceiling_bytes: 10_000,
    }
    .decide();
    let CompleteResidencyDecision::RejectInvalidObservation {
        error: observation_error,
    } = invalid_observation_decision
    else {
        panic!("retained paged payload above live active memory must be rejected");
    };
    assert_eq!(
        EnginePlacement::from_startup_complete_residency_decision(invalid_observation_decision),
        Err(observation_error)
    );
}

#[test]
fn should_log_a_distinct_label_for_each_placement() {
    assert_ne!(
        EnginePlacement::CompleteResidencyFits.log_label(),
        EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 1 }.log_label()
    );
}

#[test]
fn should_hold_the_whole_model_only_when_complete_residency_fits() {
    assert!(EnginePlacement::CompleteResidencyFits.holds_whole_model());
    assert!(
        !EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 1 }.holds_whole_model()
    );
}

#[test]
fn should_not_hold_the_whole_model_at_a_zero_byte_shortfall() {
    assert!(
        !EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 0 }.holds_whole_model()
    );
}
