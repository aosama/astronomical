//! Beside-source coverage for the block helpers that need no GPU. The attention itself is
//! covered by the oracle tests that run the block on the GPU.

use super::{AttentionSegment, target_query_start};

fn prefix_segment(query_start: usize, query_end: usize) -> AttentionSegment {
    AttentionSegment {
        query_start,
        query_end,
        key_end: query_end,
        is_text: false,
    }
}

#[test]
fn should_start_the_target_where_the_last_prefix_segment_ends() {
    let segments = [prefix_segment(0, 24), prefix_segment(24, 88)];

    assert_eq!(target_query_start(&segments), 88);
}

#[test]
fn should_start_the_target_at_the_first_token_without_prefix_segments() {
    assert_eq!(target_query_start(&[]), 0);
}

#[test]
fn should_follow_the_last_segment_even_when_an_earlier_one_ends_later() {
    let segments = [prefix_segment(0, 120), prefix_segment(0, 40)];

    assert_eq!(target_query_start(&segments), 40);
}
