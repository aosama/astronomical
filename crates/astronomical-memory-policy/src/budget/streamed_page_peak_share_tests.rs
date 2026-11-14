use super::promoted_expert_page_bytes_within_forward_peak;
use crate::budget::adaptive_growth::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
};

const GIBIBYTE: u64 = 1 << 30;
/// One complete expert layer of the 122B mixed 2-bit model.
const COMPLETE_LAYER_BYTES: u64 = 754_974_720;
const LAYER_COUNT: u64 = 48;

#[test]
fn should_give_back_only_the_page_reservation_when_a_chunk_streams_every_layer() {
    // 36 GB went through one chunk. Only a page of it can be owed to the
    // peak the admission reservation covers.
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(
            LAYER_COUNT * COMPLETE_LAYER_BYTES,
            0,
            COMPLETE_LAYER_BYTES,
        ),
        COMPLETE_LAYER_BYTES
    );
}

#[test]
fn should_give_back_everything_a_forward_streamed_below_its_reservation() {
    // One decode token that read a single small page: all of it was alive.
    let small_page_bytes = 23_592_960;
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(small_page_bytes, 0, COMPLETE_LAYER_BYTES),
        small_page_bytes
    );
}

#[test]
fn should_give_back_exactly_the_reservation_at_the_boundary() {
    for streamed_expert_page_bytes in [COMPLETE_LAYER_BYTES, COMPLETE_LAYER_BYTES + 1] {
        assert_eq!(
            promoted_expert_page_bytes_within_forward_peak(
                streamed_expert_page_bytes,
                0,
                COMPLETE_LAYER_BYTES,
            ),
            COMPLETE_LAYER_BYTES,
            "{streamed_expert_page_bytes}"
        );
    }
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(
            COMPLETE_LAYER_BYTES - 1,
            0,
            COMPLETE_LAYER_BYTES,
        ),
        COMPLETE_LAYER_BYTES - 1
    );
}

#[test]
fn should_always_count_pages_that_stayed_resident() {
    // Ten layers were seated by the chunk and the rest streamed and released.
    let retained_growth_bytes = 10 * COMPLETE_LAYER_BYTES;
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(
            LAYER_COUNT * COMPLETE_LAYER_BYTES,
            retained_growth_bytes,
            COMPLETE_LAYER_BYTES,
        ),
        retained_growth_bytes + COMPLETE_LAYER_BYTES
    );
    // Everything streamed was seated: nothing was released to give back.
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(
            retained_growth_bytes,
            retained_growth_bytes,
            COMPLETE_LAYER_BYTES,
        ),
        retained_growth_bytes
    );
}

#[test]
fn should_give_back_nothing_when_nothing_was_streamed_or_reserved() {
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(0, 0, COMPLETE_LAYER_BYTES),
        0
    );
    // Retained growth with no streamed bytes is not this function's to
    // explain; it must not invent promoted bytes.
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(0, GIBIBYTE, COMPLETE_LAYER_BYTES),
        0
    );
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(LAYER_COUNT * COMPLETE_LAYER_BYTES, 0, 0),
        0
    );
}

#[test]
fn should_not_overflow_at_the_extremes() {
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(u64::MAX, u64::MAX, u64::MAX),
        u64::MAX
    );
    assert_eq!(
        promoted_expert_page_bytes_within_forward_peak(u64::MAX, 0, u64::MAX),
        u64::MAX
    );
}

/// The journey the function exists for, through the guard that learns from
/// it: a fully streamed chunk must teach the same workspace whether the
/// streamed-byte count is the real 36 GB or was never counted, and that
/// workspace plus the page reservation must cover the observed peak.
#[test]
fn should_teach_a_streamed_chunk_a_workspace_that_with_its_reservation_covers_the_peak() {
    let ceiling_bytes = usize::try_from(32 * GIBIBYTE).expect("the ceiling fits usize");
    let active_memory_bytes_before = usize::try_from(3_631_647_118_u64).expect("fits usize");
    let active_memory_bytes_after = usize::try_from(3_735_259_534_u64).expect("fits usize");
    // The peak the chunk reached: the retained baseline plus the 6.42 GB
    // that everyday settings used to learn as workspace.
    let observed_peak_growth_bytes = 6_416_813_132_u64;
    let peak_memory_bytes = active_memory_bytes_after
        + usize::try_from(observed_peak_growth_bytes).expect("fits usize");
    let chunk_context = AdaptiveRamGrowthContext::prefill(
        4_096,
        0,
        false,
        AdaptiveRamGrowthExecutionProfile::Paged,
    );
    let promoted_bytes_within_peak = promoted_expert_page_bytes_within_forward_peak(
        LAYER_COUNT * COMPLETE_LAYER_BYTES,
        0,
        COMPLETE_LAYER_BYTES,
    );
    let mut guard = AdaptiveRamGrowthGuard::new(ceiling_bytes).expect("a valid ceiling");

    guard.record_completed_growth_excluding_expert_page_stream(
        chunk_context,
        true,
        active_memory_bytes_before,
        active_memory_bytes_after,
        peak_memory_bytes,
        0,
        0,
        promoted_bytes_within_peak,
    );

    let learned_workspace_bytes =
        u64::try_from(guard.observed_transient_high_water_bytes_for_context(chunk_context))
            .expect("fits u64");
    assert_eq!(
        learned_workspace_bytes,
        observed_peak_growth_bytes - COMPLETE_LAYER_BYTES,
        "the chunk must teach its real workspace, not zero"
    );
    assert_eq!(
        learned_workspace_bytes + COMPLETE_LAYER_BYTES,
        observed_peak_growth_bytes,
        "workspace plus the page reservation is exactly the peak that was observed"
    );
}
