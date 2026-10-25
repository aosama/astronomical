//! Boundary tests for allocation admission: exact ceiling arrivals, the
//! overflow arms, and the retained-expert capacity arithmetic under
//! memory pressure.

use crate::admission::allocation::{
    AllocationAdmissionDecision, AllocationAdmissionObservation,
    retained_expert_payload_capacity_bytes,
};
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
fn should_admit_with_allocator_cache_when_the_total_lands_exactly_on_the_ceiling() {
    // Reclaimable cache is excluded from active-limit enforcement, so a
    // total arriving exactly at the ceiling stays a plain Admit.
    let decision = AllocationAdmissionObservation::new(90, 10, 0, 100).decide();

    assert_eq!(decision, AllocationAdmissionDecision::Admit);
}

#[test]
fn should_clear_the_allocator_cache_one_byte_of_total_pressure_past_the_ceiling() {
    let decision = AllocationAdmissionObservation::new(90, 11, 10, 100).decide();

    assert_eq!(
        decision,
        AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit
    );
}

#[test]
fn should_clear_the_allocator_cache_when_the_total_projection_overflows() {
    let decision = AllocationAdmissionObservation::new(10, u64::MAX, 0, u64::MAX).decide();

    assert_eq!(
        decision,
        AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit,
        "an overflowing total still fits once the reclaimable cache is cleared"
    );
}

#[test]
fn should_report_no_capacity_when_the_pending_allocation_plus_page_reserve_overflows() {
    // With a pending retained payload the reserve charges the allocation and
    // the largest page together; their overflow must fail closed to zero
    // rather than wrap into a huge capacity.
    let capacity_bytes =
        retained_expert_payload_capacity_bytes(1_000, 2_000, 1, u64::MAX, 0, 100, 1);

    assert_eq!(
        capacity_bytes, 0,
        "an overflowing reserve leaves no honest retained capacity"
    );
}

#[test]
fn should_subtract_exactly_the_shortfall_from_retained_capacity_under_pressure() {
    // Active 1,000 plus the 100-byte page reserve stands 200 bytes over the
    // 900-byte ceiling, so the 500 post-load retained payload must shrink by
    // exactly 200.
    let capacity_bytes = retained_expert_payload_capacity_bytes(1_000, 900, 100, 0, 0, 300, 200);

    assert_eq!(capacity_bytes, 300);
}

#[test]
fn should_add_the_remaining_headroom_to_retained_capacity_when_the_reserve_fits() {
    // Active 800 plus the 100-byte reserve leaves 100 bytes under the
    // 1,000-byte ceiling on top of the 500 post-load retained payload.
    let capacity_bytes = retained_expert_payload_capacity_bytes(800, 1_000, 100, 0, 0, 300, 200);

    assert_eq!(capacity_bytes, 600);
}

#[test]
fn should_floor_capacity_at_zero_when_the_pressure_shortfall_exceeds_the_post_load_payload() {
    // A shortfall of 400 against a post-load payload of 100 saturates at
    // zero; negative capacity must never wrap.
    let capacity_bytes = retained_expert_payload_capacity_bytes(1_200, 900, 100, 0, 0, 100, 0);

    assert_eq!(capacity_bytes, 0);
}

#[test]
fn should_reserve_the_pending_allocation_itself_when_it_exceeds_the_largest_page() {
    // With no pending retained payload the reserve is the larger of the
    // largest page and the pending allocation; a 150-byte allocation
    // outweighs the 100-byte page, leaving 50 bytes of headroom under the
    // 1,000-byte ceiling on top of the 300 resident bytes.
    let capacity_bytes = retained_expert_payload_capacity_bytes(800, 1_000, 100, 150, 0, 300, 0);

    assert_eq!(capacity_bytes, 350);
}
