use crate::budget::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
};
use crate::forward_admission::{
    ResidentForwardAdmissionDecision, ResidentForwardAdmissionRejection, StreamingEngineRequirement,
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
fn should_require_streaming_for_a_transient_peak_miss_at_the_configured_chunk() {
    let forward_projection = resident_prefill_projection(ACTIVE_MEMORY_CEILING_BYTES, 101);

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, Some(2_048)),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                shortfall_bytes: 1,
                prefill_chunk_token_count: 2_048,
            },
        }
    );
}

#[test]
fn should_require_streaming_for_a_transient_peak_miss_at_a_single_token_chunk() {
    let forward_projection = resident_prefill_projection(ACTIVE_MEMORY_CEILING_BYTES, 101);

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, Some(1)),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                shortfall_bytes: 1,
                prefill_chunk_token_count: 1,
            },
        }
    );
}

#[test]
fn should_admit_a_transient_peak_exactly_at_the_ceiling_without_rejecting() {
    let forward_projection = resident_prefill_projection(ACTIVE_MEMORY_CEILING_BYTES - 100, 100);

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, Some(2_048)),
        ResidentForwardAdmissionDecision::Admit
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
