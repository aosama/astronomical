use astronomical_model_serving::{
    sliding_window_position_is_visible, sliding_window_visibility_table,
};

#[test]
fn should_hide_future_keys_and_keys_outside_the_absolute_window() {
    assert!(sliding_window_position_is_visible(10, 10, 4).expect("positive window"));
    assert!(sliding_window_position_is_visible(10, 7, 4).expect("positive window"));
    assert!(!sliding_window_position_is_visible(10, 6, 4).expect("positive window"));
    assert!(!sliding_window_position_is_visible(10, 11, 4).expect("positive window"));
}

#[test]
fn should_build_a_prefix_plus_chunk_visibility_table() {
    let visibility_table = sliding_window_visibility_table(6, 4, 0, 10, 4)
        .expect("positive absolute ranges should build a table");
    assert_eq!(
        visibility_table[0],
        vec![
            false, false, false, true, true, true, true, false, false, false
        ]
    );
    assert_eq!(
        visibility_table[3],
        vec![
            false, false, false, false, false, false, true, true, true, true
        ]
    );
}

#[test]
fn should_reject_zero_window_or_token_counts() {
    assert!(sliding_window_position_is_visible(1, 0, 0).is_err());
    assert!(sliding_window_visibility_table(0, 0, 0, 1, 4).is_err());
    assert!(sliding_window_visibility_table(0, 1, 0, 0, 4).is_err());
}
