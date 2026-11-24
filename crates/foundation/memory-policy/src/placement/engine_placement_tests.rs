use crate::admission::complete_residency::{
    CompleteResidencyDecision, CompleteResidencyRequirements,
};
use crate::forward_admission::StreamingEngineRequirement;
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
fn should_report_demotion_facts_only_for_request_demotion_placement() {
    let demotion_cause = StreamingEngineRequirement::StableShortfall {
        shortfall_bytes: 128,
    };
    let demoted_placement = EnginePlacement::DemotedForRequest {
        cause: demotion_cause,
        demoted_total_context_token_count: 9_000,
    };
    assert!(demoted_placement.is_demoted_for_request());
    assert_eq!(demoted_placement.demotion_cause(), Some(demotion_cause));
    assert_eq!(
        demoted_placement.demoted_total_context_token_count(),
        Some(9_000)
    );

    for other_placement in [
        EnginePlacement::CompleteResidencyFits,
        EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 1 },
        EnginePlacement::PromotedAfterPressureResolved,
    ] {
        assert!(!other_placement.is_demoted_for_request());
        assert_eq!(other_placement.demotion_cause(), None);
        assert_eq!(other_placement.demoted_total_context_token_count(), None);
    }
}

#[test]
fn should_log_a_distinct_label_for_every_placement_variant() {
    let labels = [
        EnginePlacement::CompleteResidencyFits.log_label(),
        EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 1 }.log_label(),
        EnginePlacement::DemotedForRequest {
            cause: StreamingEngineRequirement::StableShortfall { shortfall_bytes: 1 },
            demoted_total_context_token_count: 1,
        }
        .log_label(),
        EnginePlacement::PromotedAfterPressureResolved.log_label(),
    ];
    for (label_index, label) in labels.iter().enumerate() {
        for (other_index, other_label) in labels.iter().enumerate() {
            if label_index != other_index {
                assert_ne!(label, other_label);
            }
        }
    }
}

#[test]
fn should_run_the_resident_engine_only_for_complete_residency_placements() {
    assert!(EnginePlacement::CompleteResidencyFits.runs_resident_engine());
    assert!(EnginePlacement::PromotedAfterPressureResolved.runs_resident_engine());
    assert!(
        !EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 1 }.runs_resident_engine()
    );
    assert!(
        !EnginePlacement::DemotedForRequest {
            cause: StreamingEngineRequirement::StableShortfall { shortfall_bytes: 1 },
            demoted_total_context_token_count: 1,
        }
        .runs_resident_engine()
    );
}
