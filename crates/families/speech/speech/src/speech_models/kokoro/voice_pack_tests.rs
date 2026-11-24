use super::style_row_index;

#[test]
fn should_pick_the_row_for_the_phoneme_count() {
    assert_eq!(style_row_index(1, 510), 0);
    assert_eq!(style_row_index(37, 510), 36);
}

// Boundaries: no phonemes reads the first row, too many the last.
#[test]
fn should_stay_inside_the_table_at_both_ends() {
    assert_eq!(style_row_index(0, 510), 0);
    assert_eq!(style_row_index(510, 510), 509);
    assert_eq!(style_row_index(5000, 510), 509);
}
