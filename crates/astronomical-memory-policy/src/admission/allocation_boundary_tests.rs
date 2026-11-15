//! Boundary tests for allocation admission: exact ceiling arrivals and the
//! overflow arms.

use crate::admission::allocation::{AllocationAdmissionDecision, AllocationAdmissionObservation};
use crate::vocabulary::MemoryBoundary;

#[test]
fn should_admit_when_the_projection_lands_exactly_on_the_ceiling() {
    let decision = AllocationAdmissionObservation::new(90, 0, 10, 100).decide();

    assert_eq!(decision, AllocationAdmissionDecision::Admit);
}

#[test]
fn should_reject_one_byte_over_the_ceiling_with_an_exact_shortfall() {
    let decision = AllocationAdmissionObservation::new(91, 0, 10, 100).decide();

    assert_eq!(
        decision,
        AllocationAdmissionDecision::Reject {
            boundary: MemoryBoundary::AllocationProjection,
            shortfall_bytes: 1,
        }
    );
}

#[test]
fn should_reject_with_the_maximum_shortfall_when_the_projection_overflows() {
    let decision = AllocationAdmissionObservation::new(u64::MAX, 0, 1, u64::MAX).decide();

    assert_eq!(
        decision,
        AllocationAdmissionDecision::Reject {
            boundary: MemoryBoundary::AllocationProjection,
            shortfall_bytes: u64::MAX,
        },
        "an overflowing projection cannot fit any finite ceiling"
    );
}

#[test]
fn should_admit_with_mlx_buffer_cache_when_the_total_lands_exactly_on_the_ceiling() {
    // Reclaimable cache is excluded from active-limit enforcement, so a
    // total arriving exactly at the ceiling stays a plain Admit.
    let decision = AllocationAdmissionObservation::new(90, 10, 0, 100).decide();

    assert_eq!(decision, AllocationAdmissionDecision::Admit);
}

#[test]
fn should_clear_the_mlx_buffer_cache_one_byte_of_total_pressure_past_the_ceiling() {
    let decision = AllocationAdmissionObservation::new(90, 11, 10, 100).decide();

    assert_eq!(
        decision,
        AllocationAdmissionDecision::ClearMlxBufferCacheThenAdmit
    );
}

#[test]
fn should_clear_the_mlx_buffer_cache_when_the_total_projection_overflows() {
    let decision = AllocationAdmissionObservation::new(10, u64::MAX, 0, u64::MAX).decide();

    assert_eq!(
        decision,
        AllocationAdmissionDecision::ClearMlxBufferCacheThenAdmit,
        "an overflowing total still fits once the reclaimable cache is cleared"
    );
}
