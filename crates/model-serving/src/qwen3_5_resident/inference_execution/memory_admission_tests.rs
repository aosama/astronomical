use super::{AdaptiveRamGrowthMemoryAdmissionError, resident_forward_admission_rejection};
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
    AdaptiveRamGrowthProjection,
};

const TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES: usize = 10_000_000_000;
const ONE_PERCENT_TRANSIENT_ALLOWANCE_BYTES: usize =
    TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES / 100;
const CONFIGURED_PREFILL_CHUNK_TOKEN_COUNT: usize = 2_048;

/// Projects one resident 2,048-token prefill on an evidence-free guard, so the
/// analytic floor argument alone sets the transient reserve.
fn resident_prefill_projection(
    current_active_memory_bytes: usize,
    unobserved_scope_transient_floor_bytes: usize,
) -> AdaptiveRamGrowthProjection {
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

    assert!(resident_forward_admission_rejection(&forward_projection).is_none());
}

#[test]
fn should_admit_a_resident_forward_whose_peak_lands_exactly_on_the_transient_allowance() {
    let forward_projection = resident_prefill_projection(
        TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES,
        ONE_PERCENT_TRANSIENT_ALLOWANCE_BYTES,
    );

    assert!(resident_forward_admission_rejection(&forward_projection).is_none());
}

#[test]
fn should_classify_a_peak_one_byte_over_the_allowance_as_a_transient_peak_miss() {
    let forward_projection = resident_prefill_projection(
        TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES,
        ONE_PERCENT_TRANSIENT_ALLOWANCE_BYTES + 1,
    );

    assert!(matches!(
        resident_forward_admission_rejection(&forward_projection),
        Some(AdaptiveRamGrowthMemoryAdmissionError::TransientPeakExceeded { .. })
    ));
}

#[test]
fn should_classify_a_fitting_stable_footprint_with_a_large_transient_as_a_transient_peak_miss() {
    let forward_projection = resident_prefill_projection(9_500_000_000, 2_000_000_000);

    assert!(matches!(
        resident_forward_admission_rejection(&forward_projection),
        Some(AdaptiveRamGrowthMemoryAdmissionError::TransientPeakExceeded { .. })
    ));
}

#[test]
fn should_classify_a_stable_footprint_over_the_ceiling_as_insufficient_capacity() {
    let forward_projection =
        resident_prefill_projection(TEN_BILLION_BYTE_ACTIVE_MEMORY_CEILING_BYTES + 1, 0);

    let admission_rejection = resident_forward_admission_rejection(&forward_projection);

    match admission_rejection {
        Some(AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity { reason }) => {
            assert!(
                reason.contains("allowed ceiling is 10000000000 bytes"),
                "the rejection should name the ceiling: {reason}"
            );
        }
        _ => panic!("a stable-footprint miss must request the streaming engine"),
    }
}
