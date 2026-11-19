//! Unit tests for the host-built attention mask values.

use super::{
    HIDDEN_POSITION_MASK_VALUE, key_padding_mask_values, local_window_hides_any_pair,
    local_window_mask_values,
};
use crate::modernbert::token_batch::PaddedTokenBatch;

const HIDDEN: f32 = f32::NEG_INFINITY;

#[test]
fn should_need_a_window_mask_only_past_the_window_half_plus_one() {
    assert!(!local_window_hides_any_pair(1, 4));
    assert!(!local_window_hides_any_pair(5, 4), "every pair is within 4");
    assert!(
        local_window_hides_any_pair(6, 4),
        "positions 0 and 5 are 5 apart"
    );
    assert!(
        local_window_hides_any_pair(2, 0),
        "a zero window hides neighbors"
    );
}

#[test]
fn should_show_exactly_the_positions_within_the_window_half() {
    let mask_values = local_window_mask_values(4, 1, HIDDEN);
    #[rustfmt::skip]
    let expected_mask_values = [
        0.0,    0.0,    HIDDEN, HIDDEN,
        0.0,    0.0,    0.0,    HIDDEN,
        HIDDEN, 0.0,    0.0,    0.0,
        HIDDEN, HIDDEN, 0.0,    0.0,
    ];
    assert_eq!(mask_values, expected_mask_values);
}

#[test]
fn should_show_only_the_diagonal_for_a_zero_window() {
    let mask_values = local_window_mask_values(3, 0, HIDDEN);
    for query_position in 0..3 {
        for key_position in 0..3 {
            let mask_value = mask_values[query_position * 3 + key_position];
            if query_position == key_position {
                assert_eq!(mask_value, 0.0);
            } else {
                assert_eq!(mask_value, HIDDEN);
            }
        }
    }
}

#[test]
fn should_show_everything_when_the_window_covers_the_sequence() {
    let mask_values = local_window_mask_values(3, 10, HIDDEN);
    assert_eq!(mask_values, [0.0; 9]);
}

#[test]
fn should_produce_no_values_for_an_empty_sequence() {
    assert!(local_window_mask_values(0, 4, HIDDEN).is_empty());
}

#[test]
fn should_hide_each_rows_padding_and_show_its_real_tokens() {
    let token_batch =
        PaddedTokenBatch::from_sequences(&[&[1, 2, 3], &[4]], 0).expect("a valid batch");
    assert_eq!(
        key_padding_mask_values(&token_batch),
        [
            0.0,
            0.0,
            0.0,
            0.0,
            HIDDEN_POSITION_MASK_VALUE,
            HIDDEN_POSITION_MASK_VALUE
        ]
    );
}
