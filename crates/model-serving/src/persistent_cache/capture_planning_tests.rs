//! Unit coverage for the shared capture-planning units: boundary window
//! derivation, causal-input selection, publication workspace decision, and
//! allocator-clearing accounting — positive, boundary, and negative cases.
//! These rules previously lived as untested copies inside both engines'
//! capture functions.

use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::persistent_cache::capture_planning::{
    BlockCausalInputSelection, CaptureBoundaryRejection, CaptureCausalInputRejection,
    PUBLICATION_COMFORT_HEADROOM_BYTES, allocator_bytes_cleared, capture_block_window,
    publication_workspace_decision, select_block_causal_input,
};

#[test]
fn should_derive_a_block_window_from_an_aligned_boundary_checkpoint() {
    let capture_window =
        capture_block_window(128, 64, 64, 200).expect("an aligned boundary is capturable");

    assert_eq!(capture_window.block_start, 128);
    assert_eq!(capture_window.block_end, 192);
    assert_eq!(capture_window.block_index, 2);
}

#[test]
fn should_reject_a_boundary_position_that_overflows_the_prefill_start() {
    let rejection = capture_block_window(usize::MAX, 1, 64, usize::MAX)
        .expect_err("an overflowing boundary must not capture");

    assert_eq!(rejection, CaptureBoundaryRejection::PositionOverflowed);
    assert_eq!(
        rejection.description(),
        "prompt-cache boundary position overflowed"
    );
}

#[test]
fn should_reject_a_boundary_position_off_the_block_grid() {
    let rejection =
        capture_block_window(100, 63, 64, 200).expect_err("a mid-block boundary must not capture");

    assert_eq!(rejection, CaptureBoundaryRejection::PositionInvalid);
    assert_eq!(
        rejection.description(),
        "prompt-cache boundary position is invalid"
    );
}

#[test]
fn should_reject_a_boundary_position_beyond_the_successful_prefill_end() {
    let rejection = capture_block_window(100, 128, 64, 150)
        .expect_err("a boundary past the successful span must not capture");

    assert_eq!(rejection, CaptureBoundaryRejection::PositionInvalid);
}

#[test]
fn should_reject_a_boundary_that_leaves_no_room_for_a_full_block() {
    // The boundary is aligned at position zero and inside the span, but
    // subtracting one block length would underflow below the first block.
    let rejection =
        capture_block_window(0, 0, 64, 0).expect_err("a zero-position window must not capture");

    assert_eq!(rejection, CaptureBoundaryRejection::BlockStartUnderflowed);
    assert_eq!(
        rejection.description(),
        "prompt-cache block start underflowed"
    );
}

#[test]
fn should_accept_a_boundary_exactly_at_the_successful_prefill_end() {
    let capture_window =
        capture_block_window(0, 128, 64, 128).expect("the span end is a valid boundary");

    assert_eq!(capture_window.block_start, 64);
    assert_eq!(capture_window.block_end, 128);
    assert_eq!(capture_window.block_index, 1);
}

#[test]
fn should_select_the_empty_causal_input_when_the_plan_is_empty() {
    let selection = select_block_causal_input(&[], 0)
        .expect("an empty plan selects the empty input everywhere");

    assert_eq!(selection, BlockCausalInputSelection::Empty);
}

#[test]
fn should_select_the_planned_causal_input_for_a_covered_block() {
    let planned_causal_inputs = [
        PersistentPromptCacheBlockCausalInput::empty(),
        PersistentPromptCacheBlockCausalInput::empty(),
    ];

    let selection = select_block_causal_input(&planned_causal_inputs, 1)
        .expect("a covered block resolves through the plan");

    assert_eq!(
        selection,
        BlockCausalInputSelection::FromPlan(&planned_causal_inputs[1])
    );
}

#[test]
fn should_reject_a_non_empty_plan_that_does_not_cover_the_block() {
    let planned_causal_inputs = [PersistentPromptCacheBlockCausalInput::empty()];

    let rejection = select_block_causal_input(&planned_causal_inputs, 5)
        .expect_err("a partial plan is a contract violation");

    assert_eq!(
        rejection,
        CaptureCausalInputRejection::PlanDoesNotCoverBlock
    );
    assert_eq!(
        rejection.description(),
        "prompt-cache causal input plan does not cover captured block"
    );
}

#[test]
fn should_require_workspace_only_inside_the_comfort_headroom_of_the_ceiling() {
    let ceiling_bytes = 20 * 1024 * 1024 * 1024;

    // Far below the ceiling: publication pays no allocator wipe.
    assert!(!publication_workspace_decision(
        Some(ceiling_bytes - PUBLICATION_COMFORT_HEADROOM_BYTES - 1),
        Some(ceiling_bytes),
    ));
    // Exactly at the headroom boundary: still roomy.
    assert!(!publication_workspace_decision(
        Some(ceiling_bytes - PUBLICATION_COMFORT_HEADROOM_BYTES),
        Some(ceiling_bytes),
    ));
    // One byte inside the headroom: the ceiling needs the room.
    assert!(publication_workspace_decision(
        Some(ceiling_bytes - PUBLICATION_COMFORT_HEADROOM_BYTES + 1),
        Some(ceiling_bytes),
    ));
}

#[test]
fn should_require_workspace_when_the_memory_state_is_unreadable() {
    // An unreadable snapshot or limit keeps the old unconditional wipe
    // behavior rather than skipping cleanup on a measurement gap.
    assert!(publication_workspace_decision(
        None,
        Some(20 * 1024 * 1024 * 1024)
    ));
    assert!(publication_workspace_decision(Some(0), None));
    assert!(publication_workspace_decision(None, None));
}

#[test]
fn should_count_allocator_bytes_cleared_between_snapshots() {
    let before_bytes = 512 * 1024 * 1024;
    let after_bytes = 64 * 1024 * 1024;

    assert_eq!(
        allocator_bytes_cleared(Some(before_bytes), Some(after_bytes)),
        448 * 1024 * 1024
    );
}

#[test]
fn should_treat_an_unreadable_snapshot_as_zero_cleared_bytes() {
    // Before unreadable: nothing can be proven cleared.
    assert_eq!(allocator_bytes_cleared(None, Some(512)), 0);
    // After unreadable: the full before value counts as cleared, matching the
    // original engines' accounting where an unreadable after-snapshot reads
    // as zero bytes remaining.
    assert_eq!(allocator_bytes_cleared(Some(64), None), 64);
    assert_eq!(allocator_bytes_cleared(None, None), 0);
}
