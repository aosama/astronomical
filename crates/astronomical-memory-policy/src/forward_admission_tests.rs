use crate::forward_admission::{
    ResidentForwardAdmissionDecision, ResidentForwardAdmissionRejection, StreamingEngineRequirement,
};

#[test]
fn should_carry_no_rejection_for_an_admitted_resident_forward() {
    assert_eq!(ResidentForwardAdmissionDecision::Admit.rejection(), None);
}

#[test]
fn should_carry_the_typed_streaming_cause_into_the_resident_rejection() {
    let stable_shortfall_cause = StreamingEngineRequirement::StableShortfall { shortfall_bytes: 1 };
    assert_eq!(
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: stable_shortfall_cause,
        }
        .rejection(),
        Some(ResidentForwardAdmissionRejection::RequireStreamingEngine {
            cause: stable_shortfall_cause,
        })
    );
}

#[test]
fn should_admit_a_forward_the_ledger_leaves_nothing_short_for() {
    assert_eq!(
        ResidentForwardAdmissionDecision::for_measured_shortfalls(0, 0, Some(2_048)),
        ResidentForwardAdmissionDecision::Admit
    );
}

#[test]
fn should_name_the_stored_state_when_it_alone_is_over_the_limit() {
    assert_eq!(
        ResidentForwardAdmissionDecision::for_measured_shortfalls(7, 90, Some(2_048)),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::StableShortfall { shortfall_bytes: 7 },
        }
    );
}

#[test]
fn should_name_the_prompt_chunk_when_only_its_mlx_working_buffers_are_over() {
    assert_eq!(
        ResidentForwardAdmissionDecision::for_measured_shortfalls(0, 1, Some(2_048)),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                shortfall_bytes: 1,
                prefill_chunk_token_count: 2_048,
            },
        }
    );
}

#[test]
fn should_name_the_reply_step_when_its_mlx_working_buffers_are_over() {
    assert_eq!(
        ResidentForwardAdmissionDecision::for_measured_shortfalls(0, 5, None),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallWithoutPrefillChunk {
                shortfall_bytes: 5,
            },
        }
    );
}
