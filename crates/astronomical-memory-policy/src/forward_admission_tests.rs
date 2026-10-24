use crate::budget::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
};
use crate::forward_admission::{
    MINIMUM_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT, ResidentForwardAdmissionDecision,
    ResidentForwardAdmissionRejection, StreamingEngineRequirement,
    next_smaller_resident_prefill_chunk_token_count,
};

const ACTIVE_MEMORY_CEILING_BYTES: usize = 10_000;

fn resident_prefill_projection(
    current_active_memory_bytes: usize,
    transient_workspace_bytes: usize,
) -> crate::budget::AdaptiveRamGrowthProjection {
    AdaptiveRamGrowthGuard::new(ACTIVE_MEMORY_CEILING_BYTES)
        .expect("valid ceiling")
        .project_growth_for_context(
            AdaptiveRamGrowthContext::prefill(
                2_048,
                0,
                false,
                AdaptiveRamGrowthExecutionProfile::Resident,
            ),
            current_active_memory_bytes,
            0,
            0,
            0,
            transient_workspace_bytes,
        )
        .expect("the projection should fit the platform range")
}

#[test]
fn should_admit_a_resident_forward_when_stable_and_peak_limits_fit() {
    let forward_projection = resident_prefill_projection(1_000, 100);

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, Some(2_048)),
        ResidentForwardAdmissionDecision::Admit
    );
}

#[test]
fn should_shrink_a_transient_only_miss_while_the_prefill_chunk_can_shrink() {
    let forward_projection = resident_prefill_projection(ACTIVE_MEMORY_CEILING_BYTES, 101);

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, Some(1_024)),
        ResidentForwardAdmissionDecision::ShrinkPrefillChunk {
            maximum_token_count: 512,
        }
    );
}

#[test]
fn should_never_route_a_shrinkable_transient_peak_miss_to_streaming() {
    let forward_projection = resident_prefill_projection(ACTIVE_MEMORY_CEILING_BYTES, 101);
    let forward_admission_decision =
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, Some(512));

    assert!(matches!(
        forward_admission_decision,
        ResidentForwardAdmissionDecision::ShrinkPrefillChunk { .. }
    ));
    assert!(!matches!(
        forward_admission_decision,
        ResidentForwardAdmissionDecision::RequireStreamingEngine { .. }
    ));
}

#[test]
fn should_require_streaming_only_for_a_transient_peak_at_the_minimum_chunk() {
    let forward_projection = resident_prefill_projection(ACTIVE_MEMORY_CEILING_BYTES, 101);

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(
            &forward_projection,
            Some(MINIMUM_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT),
        ),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallAtMinimumPrefillChunk {
                shortfall_bytes: 1,
                prefill_chunk_token_count: MINIMUM_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT,
            },
        }
    );
}

#[test]
fn should_attribute_a_stable_shortfall_only_to_the_stable_cause() {
    let forward_projection = resident_prefill_projection(ACTIVE_MEMORY_CEILING_BYTES + 1, 0);

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, Some(2_048)),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::StableShortfall { shortfall_bytes: 1 },
        }
    );
}

#[test]
fn should_use_a_decode_cause_when_no_prefill_chunk_can_shrink() {
    let forward_projection =
        crate::budget::AdaptiveRamGrowthGuard::new(ACTIVE_MEMORY_CEILING_BYTES)
            .expect("valid ceiling")
            .project_growth_for_context(
                AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Resident),
                ACTIVE_MEMORY_CEILING_BYTES,
                0,
                0,
                0,
                101,
            )
            .expect("the projection should fit the platform range");

    assert!(matches!(
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, None),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallWithoutPrefillChunk {
                shortfall_bytes: 1
            }
        }
    ));
}

#[test]
fn should_halve_chunks_down_to_but_never_below_the_minimum() {
    assert_eq!(
        next_smaller_resident_prefill_chunk_token_count(2_048),
        Some(1_024)
    );
    assert_eq!(
        next_smaller_resident_prefill_chunk_token_count(400),
        Some(MINIMUM_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT)
    );
    assert_eq!(
        next_smaller_resident_prefill_chunk_token_count(
            MINIMUM_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT + 1
        ),
        Some(MINIMUM_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT)
    );
    assert_eq!(
        next_smaller_resident_prefill_chunk_token_count(MINIMUM_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT),
        None
    );
    assert_eq!(next_smaller_resident_prefill_chunk_token_count(0), None);
}

#[test]
fn should_carry_no_rejection_for_an_admitted_resident_forward() {
    assert_eq!(ResidentForwardAdmissionDecision::Admit.rejection(), None);
}

#[test]
fn should_carry_the_shrink_bound_into_the_resident_rejection() {
    assert_eq!(
        ResidentForwardAdmissionDecision::ShrinkPrefillChunk {
            maximum_token_count: MINIMUM_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT,
        }
        .rejection(),
        Some(ResidentForwardAdmissionRejection::ShrinkPrefillChunk {
            maximum_token_count: MINIMUM_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT,
        })
    );
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
