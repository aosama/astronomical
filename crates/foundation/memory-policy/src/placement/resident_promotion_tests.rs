use crate::forward_admission::StreamingEngineRequirement;
use crate::placement::engine_placement::EnginePlacement;
use crate::placement::resident_promotion::{
    ResidentPromotionBlocker, ResidentPromotionDecision,
    should_promote_demoted_streaming_engine_to_resident,
};
use crate::residency::StreamingStartupResidencyFacts;

/// Complete residency (core 1,000 + experts 2,000 + headroom 500) fits a
/// 3,500-byte empty ceiling and misses a smaller one; each context token
/// adds one more byte of decoder state.
const STARTUP_RESIDENCY_FACTS: StreamingStartupResidencyFacts = StreamingStartupResidencyFacts {
    model_core_payload_bytes: 1_000,
    complete_expert_payload_bytes: 2_000,
    startup_complete_residency_headroom_bytes: 500,
    sequence_state_bytes_per_token: 1,
};
const EMPTY_COMPLETE_RESIDENCY_BYTES: u64 = 3_500;
/// Leaves room for up to 10,000 context tokens on top of complete residency,
/// so every request below the 9,000-token demotion context fits.
const FITTING_EFFECTIVE_CEILING_BYTES: u64 = EMPTY_COMPLETE_RESIDENCY_BYTES + 10_000;
const DEMOTED_PLACEMENT: EnginePlacement = EnginePlacement::DemotedForRequest {
    cause: StreamingEngineRequirement::StableShortfall {
        shortfall_bytes: 128,
    },
    demoted_total_context_token_count: 9_000,
};

#[test]
fn should_promote_when_every_gate_passes() {
    assert_eq!(
        should_promote_demoted_streaming_engine_to_resident(
            DEMOTED_PLACEMENT,
            true,
            &STARTUP_RESIDENCY_FACTS,
            FITTING_EFFECTIVE_CEILING_BYTES,
            1_000,
        ),
        ResidentPromotionDecision::PromoteToResident
    );
}

#[test]
fn should_stay_streaming_when_placement_is_not_a_request_demotion() {
    for placement in [
        EnginePlacement::CompleteResidencyFits,
        EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 1 },
        EnginePlacement::PromotedAfterPressureResolved,
    ] {
        assert_eq!(
            should_promote_demoted_streaming_engine_to_resident(
                placement,
                true,
                &STARTUP_RESIDENCY_FACTS,
                FITTING_EFFECTIVE_CEILING_BYTES,
                1,
            ),
            ResidentPromotionDecision::StayStreaming {
                cause: ResidentPromotionBlocker::NotDemotedForRequest
            }
        );
    }
}

#[test]
fn should_stay_streaming_when_factory_cannot_build_a_resident_engine() {
    assert_eq!(
        should_promote_demoted_streaming_engine_to_resident(
            DEMOTED_PLACEMENT,
            false,
            &STARTUP_RESIDENCY_FACTS,
            FITTING_EFFECTIVE_CEILING_BYTES,
            1,
        ),
        ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::ResidentPromotionUnsupportedByFactory
        }
    );
}

#[test]
fn should_stay_streaming_when_complete_residency_does_not_fit_at_effective_ceiling() {
    assert_eq!(
        should_promote_demoted_streaming_engine_to_resident(
            DEMOTED_PLACEMENT,
            true,
            &STARTUP_RESIDENCY_FACTS,
            EMPTY_COMPLETE_RESIDENCY_BYTES - 1,
            1,
        ),
        ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::CompleteResidencyDoesNotFitAtEffectiveCeiling
        }
    );
}

#[test]
fn should_stay_streaming_when_incoming_context_equals_the_demotion_context() {
    // The hysteresis edge: the demoting request itself re-entering admission
    // compares equal, not strictly below, so it can never re-trigger the
    // promotion it caused and the placement cannot loop.
    assert_eq!(
        should_promote_demoted_streaming_engine_to_resident(
            DEMOTED_PLACEMENT,
            true,
            &STARTUP_RESIDENCY_FACTS,
            FITTING_EFFECTIVE_CEILING_BYTES,
            9_000,
        ),
        ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::IncomingContextNotBelowDemotionContext {
                incoming_total_context_token_count: 9_000,
                demoted_total_context_token_count: 9_000,
            }
        }
    );
}

#[test]
fn should_stay_streaming_when_incoming_context_exceeds_the_demotion_context() {
    assert_eq!(
        should_promote_demoted_streaming_engine_to_resident(
            DEMOTED_PLACEMENT,
            true,
            &STARTUP_RESIDENCY_FACTS,
            FITTING_EFFECTIVE_CEILING_BYTES,
            9_001,
        ),
        ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::IncomingContextNotBelowDemotionContext {
                incoming_total_context_token_count: 9_001,
                demoted_total_context_token_count: 9_000,
            }
        }
    );
}

#[test]
fn should_promote_at_the_strictly_smallest_context_below_the_demotion_context() {
    assert_eq!(
        should_promote_demoted_streaming_engine_to_resident(
            DEMOTED_PLACEMENT,
            true,
            &STARTUP_RESIDENCY_FACTS,
            FITTING_EFFECTIVE_CEILING_BYTES,
            8_999,
        ),
        ResidentPromotionDecision::PromoteToResident
    );
}

#[test]
fn should_stay_streaming_when_a_smaller_request_still_does_not_fit_complete_residency() {
    // Complete residency fits empty and the request is below the demotion
    // context, but its 5,000 tokens of decoder state exceed the 4,000 bytes
    // left above complete residency: promoting would only demote it again.
    assert_eq!(
        should_promote_demoted_streaming_engine_to_resident(
            DEMOTED_PLACEMENT,
            true,
            &STARTUP_RESIDENCY_FACTS,
            EMPTY_COMPLETE_RESIDENCY_BYTES + 4_000,
            5_000,
        ),
        ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::IncomingRequestContextDoesNotFitCompleteResidency {
                incoming_total_context_token_count: 5_000,
            }
        }
    );
}

#[test]
fn should_promote_when_the_request_context_exactly_fills_the_room_above_complete_residency() {
    assert_eq!(
        should_promote_demoted_streaming_engine_to_resident(
            DEMOTED_PLACEMENT,
            true,
            &STARTUP_RESIDENCY_FACTS,
            EMPTY_COMPLETE_RESIDENCY_BYTES + 4_000,
            4_000,
        ),
        ResidentPromotionDecision::PromoteToResident
    );
}
