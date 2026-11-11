use super::{causal_mask_values, single_query_sees_every_key};

const HIDDEN: f32 = f32::NEG_INFINITY;

#[test]
fn should_let_each_new_position_see_itself_and_the_past() {
    assert_eq!(causal_mask_values(0, 2, None), [0.0, HIDDEN, 0.0, 0.0]);
}

#[test]
fn should_let_a_later_block_see_every_stored_key() {
    assert_eq!(causal_mask_values(2, 1, None), [0.0, 0.0, 0.0]);
}

// Boundary: a window of 2 lets each position see itself and one before it.
#[test]
fn should_hide_keys_older_than_the_window() {
    assert_eq!(
        causal_mask_values(0, 3, Some(2)),
        [0.0, HIDDEN, HIDDEN, 0.0, 0.0, HIDDEN, HIDDEN, 0.0, 0.0]
    );
}

#[test]
fn should_apply_the_window_to_stored_keys_too() {
    // Two stored keys, one new query, window 2: the oldest key is out of reach.
    assert_eq!(causal_mask_values(2, 1, Some(2)), [HIDDEN, 0.0, 0.0]);
}

// Negative: no new positions, no mask rows.
#[test]
fn should_build_nothing_for_no_new_positions() {
    assert!(causal_mask_values(4, 0, Some(2)).is_empty());
}

#[test]
fn should_skip_the_mask_only_when_one_query_may_see_every_key() {
    assert!(single_query_sees_every_key(100, None));
    assert!(single_query_sees_every_key(2, Some(3)));
    assert!(!single_query_sees_every_key(3, Some(3)));
}
