use super::AdaptiveRamGrowthMemoryAdmissionError;
use super::resident_forward_admission_reason;
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
    InferenceEngineError, ResidentForwardAdmissionDecision, StreamingEngineRequirement,
};

const TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES: usize = 10_000_000_000;
const ONE_PERCENT_TRANSIENT_ALLOWANCE_BYTES: usize =
    TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES / 100;
const CONFIGURED_PREFILL_CHUNK_TOKEN_COUNT: usize = 2_048;

fn resident_prefill_projection(
    current_active_memory_bytes: usize,
    unobserved_scope_transient_floor_bytes: usize,
) -> crate::AdaptiveRamGrowthProjection {
    AdaptiveRamGrowthGuard::new(TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES)
        .expect("a ten-billion-byte ceiling should create a guard")
        .project_growth_for_context(
            AdaptiveRamGrowthContext::prefill(
                CONFIGURED_PREFILL_CHUNK_TOKEN_COUNT,
                0,
                false,
                AdaptiveRamGrowthExecutionProfile::Resident,
            ),
            current_active_memory_bytes,
            0,
            0,
            0,
            unobserved_scope_transient_floor_bytes,
        )
        .expect("an in-range resident prefill should project")
}

#[test]
fn should_admit_a_resident_forward_whose_stable_and_peak_projections_fit() {
    let forward_projection = resident_prefill_projection(1_000_000_000, 1_000_000_000);

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(
            &forward_projection,
            Some(CONFIGURED_PREFILL_CHUNK_TOKEN_COUNT),
        ),
        ResidentForwardAdmissionDecision::Admit
    );
}

#[test]
fn should_admit_a_resident_forward_whose_peak_lands_on_the_transient_allowance() {
    let forward_projection = resident_prefill_projection(
        TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES,
        ONE_PERCENT_TRANSIENT_ALLOWANCE_BYTES,
    );

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(
            &forward_projection,
            Some(CONFIGURED_PREFILL_CHUNK_TOKEN_COUNT),
        ),
        ResidentForwardAdmissionDecision::Admit
    );
}

#[test]
fn should_require_streaming_for_a_peak_one_byte_over_the_allowance_at_the_configured_chunk() {
    let forward_projection = resident_prefill_projection(
        TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES,
        ONE_PERCENT_TRANSIENT_ALLOWANCE_BYTES + 1,
    );
    let forward_admission_decision = ResidentForwardAdmissionDecision::for_resident_forward(
        &forward_projection,
        Some(CONFIGURED_PREFILL_CHUNK_TOKEN_COUNT),
    );

    assert_eq!(
        forward_admission_decision,
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                shortfall_bytes: 1,
                prefill_chunk_token_count: CONFIGURED_PREFILL_CHUNK_TOKEN_COUNT,
            }
        }
    );
    assert!(
        resident_forward_admission_reason(
            &forward_projection,
            forward_admission_decision
                .rejection()
                .expect("a missed forward is a rejection"),
        )
        .contains("transient peak")
    );
}

#[test]
fn should_classify_a_stable_footprint_over_the_ceiling_as_a_stable_shortfall() {
    let forward_projection =
        resident_prefill_projection(TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES + 1, 0);
    let forward_admission_decision = ResidentForwardAdmissionDecision::for_resident_forward(
        &forward_projection,
        Some(CONFIGURED_PREFILL_CHUNK_TOKEN_COUNT),
    );

    assert_eq!(
        forward_admission_decision,
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::StableShortfall { shortfall_bytes: 1 },
        }
    );
    assert!(
        resident_forward_admission_reason(
            &forward_projection,
            forward_admission_decision
                .rejection()
                .expect("a missed forward is a rejection"),
        )
        .contains("exceeds the 10000000000 byte ceiling by 1 bytes")
    );
}

#[test]
fn should_keep_a_decode_transient_peak_miss_as_an_invalid_request() {
    // A decode forward has no chunk to shrink and no retry the caller can enact,
    // so converting its typed decision through InferenceEngineError must keep
    // the pre-slice-A invalid-request rejection. Promoting it to
    // ResidentForkRequired would make decode paths request an engine fork that
    // only a replayable prefill attempt can justify.
    let forward_projection =
        AdaptiveRamGrowthGuard::new(TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES)
            .expect("a ten-billion-byte ceiling should create a guard")
            .project_growth_for_context(
                AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Resident),
                TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES,
                0,
                0,
                0,
                ONE_PERCENT_TRANSIENT_ALLOWANCE_BYTES + 1,
            )
            .expect("an in-range decode should project");
    let forward_admission_decision =
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, None);

    assert!(matches!(
        forward_admission_decision,
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallWithoutPrefillChunk { .. },
        }
    ));
    let forward_admission_rejection = forward_admission_decision
        .rejection()
        .expect("a transient-peak miss is a rejection");
    let converted_inference_engine_error =
        InferenceEngineError::from(AdaptiveRamGrowthMemoryAdmissionError::ForwardAdmission {
            rejection: forward_admission_rejection,
            reason: resident_forward_admission_reason(
                &forward_projection,
                forward_admission_rejection,
            ),
        });

    assert!(
        matches!(
            converted_inference_engine_error,
            InferenceEngineError::InvalidRequest { .. }
        ),
        "a decode transient-peak miss must stay an invalid request, got {converted_inference_engine_error:?}"
    );
    assert!(!matches!(
        converted_inference_engine_error,
        InferenceEngineError::ResidentForkRequired { .. }
    ));
}
