//! Beside-source coverage for the pure block-group window computation that
//! bounds how far one cancellable group advance may run within a stream.

use super::block_group_window;

#[test]
fn should_cover_the_full_requested_group_when_blocks_remain() {
    let window = block_group_window(2, 4, 19);

    assert_eq!(window.first_block_index, 2);
    assert_eq!(window.end_block_index, 6);
}

#[test]
fn should_clamp_the_final_group_to_the_remaining_blocks() {
    let window = block_group_window(16, 8, 19);

    assert_eq!(window.first_block_index, 16);
    assert_eq!(window.end_block_index, 19);
}

#[test]
fn should_clamp_an_oversized_group_request_to_the_stream_total() {
    let window = block_group_window(0, usize::MAX, 19);

    assert_eq!(window.first_block_index, 0);
    assert_eq!(window.end_block_index, 19);
}

#[test]
fn should_yield_an_empty_window_when_the_group_length_is_zero() {
    let window = block_group_window(5, 0, 19);

    assert_eq!(window.first_block_index, 5);
    assert_eq!(window.end_block_index, 5);
}
