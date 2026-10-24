//! Unit contracts for predicting startup engine selection under a raised
//! ceiling, mirroring the numbers the streaming round-trip journeys use.

use super::promotion_on_raise::StreamingStartupResidencyFacts;

/// Core 2.6 GB + complete experts 24.8 GB + startup headroom 0.9 GB: the
/// large sparse mixture-of-experts shape the expert-paging journeys load.
const LARGE_SPARSE_MOE_FACTS: StreamingStartupResidencyFacts = StreamingStartupResidencyFacts {
    model_core_payload_bytes: 2_600_000_000,
    complete_expert_payload_bytes: 24_800_000_000,
    startup_complete_residency_headroom_bytes: 900_000_000,
    sequence_state_bytes_per_token: 100_000,
};

#[test]
fn should_select_complete_residency_when_the_raised_ceiling_covers_payload_and_headroom() {
    // 38 marketed GB (2^30 convention) comfortably covers 28.3 GB of
    // projected payload plus headroom.
    assert!(LARGE_SPARSE_MOE_FACTS.raised_ceiling_selects_complete_residency(38 * 1_073_741_824));
}

#[test]
fn should_keep_streaming_when_the_raised_ceiling_still_misses_the_headroom() {
    // 28.3 GB exactly equals payload + headroom, so the ceiling must be at
    // least that value; one byte less keeps the streaming engine.
    let payload_with_headroom_bytes = LARGE_SPARSE_MOE_FACTS.model_core_payload_bytes
        + LARGE_SPARSE_MOE_FACTS.complete_expert_payload_bytes
        + LARGE_SPARSE_MOE_FACTS.startup_complete_residency_headroom_bytes;
    assert!(
        LARGE_SPARSE_MOE_FACTS
            .raised_ceiling_selects_complete_residency(payload_with_headroom_bytes)
    );
    assert!(
        !LARGE_SPARSE_MOE_FACTS
            .raised_ceiling_selects_complete_residency(payload_with_headroom_bytes - 1)
    );
}

#[test]
fn should_keep_streaming_for_a_ceiling_that_only_fits_the_payload() {
    // 27.4 GB covers the projected payload but not the activation headroom,
    // which is the exact squeeze the startup selection must reject.
    assert!(!LARGE_SPARSE_MOE_FACTS.raised_ceiling_selects_complete_residency(27_400_000_000));
}

#[test]
fn should_admit_request_context_that_fits_above_complete_residency() {
    let empty_complete_residency_bytes = 28_300_000_000;
    // 10,000 tokens x 100 KB = 1 GB of decoder state.
    assert!(
        LARGE_SPARSE_MOE_FACTS.complete_residency_admits_request_context(
            empty_complete_residency_bytes + 1_000_000_000,
            10_000,
        )
    );
}

#[test]
fn should_reject_request_context_one_token_beyond_the_room_above_complete_residency() {
    let empty_complete_residency_bytes = 28_300_000_000;
    assert!(
        !LARGE_SPARSE_MOE_FACTS.complete_residency_admits_request_context(
            empty_complete_residency_bytes + 1_000_000_000,
            10_001,
        )
    );
}

#[test]
fn should_reject_a_saturating_request_context_without_overflowing() {
    assert!(
        !LARGE_SPARSE_MOE_FACTS.complete_residency_admits_request_context(u64::MAX, usize::MAX)
    );
}
