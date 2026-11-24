//! Boundary contracts for the sizes a request's context needs: the output
//! tokens to hold state for and the smallest limit a loaded model can be
//! lowered to.

use crate::admission::context::{
    VISIBLE_ANSWER_RESERVE_TOKENS, bounded_output_reservation_token_count,
    safe_minimum_active_memory_ceiling_bytes,
};

#[test]
fn should_floor_the_safe_minimum_ceiling_at_zero_when_idle_memory_is_fully_evictable() {
    assert_eq!(
        safe_minimum_active_memory_ceiling_bytes(500, 500, 60),
        60,
        "fully evictable idle memory leaves only the page reserve as the floor"
    );
    assert_eq!(
        safe_minimum_active_memory_ceiling_bytes(200, 500, 60),
        60,
        "more evictable payload than idle memory must saturate at zero, not wrap"
    );
    assert_eq!(
        safe_minimum_active_memory_ceiling_bytes(500, 100, 60),
        460,
        "a partially evictable idle footprint keeps its non-evictable remainder"
    );
}

#[test]
fn should_bound_the_output_reservation_by_the_thinking_allowance_plus_the_answer_reserve() {
    assert_eq!(
        bounded_output_reservation_token_count(65_535, Some(16_384)),
        18_432,
        "the 2026-10-16 agent shape: 65,535 requested with a 16,384 thinking budget reserves thinking plus one answer chunk, not the full budget"
    );
    assert_eq!(VISIBLE_ANSWER_RESERVE_TOKENS, 2_048);
}

#[test]
fn should_never_reserve_more_output_than_the_caller_configured() {
    assert_eq!(
        bounded_output_reservation_token_count(512, Some(16_384)),
        512
    );
    assert_eq!(bounded_output_reservation_token_count(300, None), 300);
}

#[test]
fn should_reserve_the_visible_answer_allowance_when_no_thinking_budget_exists() {
    assert_eq!(
        bounded_output_reservation_token_count(65_535, None),
        2_048,
        "without a thinking budget only the visible-answer reserve is pre-held; longer answers grow under the per-step guard"
    );
}

#[test]
fn should_saturate_when_the_thinking_allowance_nears_the_word_size() {
    assert_eq!(
        bounded_output_reservation_token_count(usize::MAX, Some(usize::MAX)),
        usize::MAX
    );
}

#[test]
fn should_floor_the_safe_minimum_ceiling_at_the_page_reserve_when_eviction_exceeds_idle() {
    // Reclaiming more than the idle footprint cannot go below zero, but the
    // pager still needs its one page reserve above the floor.
    assert_eq!(
        safe_minimum_active_memory_ceiling_bytes(100, 700, 25),
        25,
        "eviction beyond the idle footprint saturates at zero before the \
         page reserve is added"
    );
    assert_eq!(safe_minimum_active_memory_ceiling_bytes(100, 70, 25), 55);
    assert_eq!(safe_minimum_active_memory_ceiling_bytes(0, 0, 0), 0);
}
