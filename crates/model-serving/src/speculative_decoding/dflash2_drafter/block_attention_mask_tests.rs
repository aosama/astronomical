use super::block_attention_mask::block_attention_visibility;

#[test]
fn should_let_every_block_row_see_every_block_row_when_not_causal() {
    let visibility = block_attention_visibility(0, 3, Some(8), false);
    assert_eq!(visibility, vec![true; 9]);
}

#[test]
fn should_hide_context_rows_that_fall_out_of_the_window_for_later_block_rows() {
    // 7 kept context rows, window 8: block row 0 (index 7) sees all 7;
    // block row 1 (index 8) loses context row 0, block row 7 (index 14) keeps only row 7.
    let visibility = block_attention_visibility(7, 8, Some(8), false);
    let row = |block_row: usize| &visibility[block_row * 15..block_row * 15 + 15];
    assert_eq!(&row(0)[..7], &[true; 7]);
    assert_eq!(&row(1)[..7], &[false, true, true, true, true, true, true]);
    assert_eq!(&row(7)[..7], &[false; 7]);
    assert!(row(7)[7..].iter().all(|is_visible| *is_visible));
}

#[test]
fn should_show_every_context_row_on_a_full_attention_layer() {
    let visibility = block_attention_visibility(100, 2, None, false);
    assert!(visibility.iter().all(|is_visible| *is_visible));
}

#[test]
fn should_hide_later_block_rows_when_the_checkpoint_asks_for_a_causal_block() {
    let visibility = block_attention_visibility(1, 2, Some(8), true);
    assert_eq!(visibility, vec![true, true, false, true, true, true]);
}

#[test]
fn should_produce_an_empty_mask_for_an_empty_block() {
    assert!(block_attention_visibility(5, 0, Some(8), false).is_empty());
}
