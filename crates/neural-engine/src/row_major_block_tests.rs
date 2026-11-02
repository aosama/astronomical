use super::row_block;

// Two rows of four columns: [0 1 2 3] and [4 5 6 7].
const TWO_BY_FOUR: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

#[test]
fn should_take_the_matching_column_block_from_every_row() {
    assert_eq!(row_block(&TWO_BY_FOUR, 4, 0, 2), vec![0, 1, 4, 5]);
    assert_eq!(row_block(&TWO_BY_FOUR, 4, 2, 2), vec![2, 3, 6, 7]);
}

#[test]
fn should_return_the_whole_matrix_for_a_full_width_block() {
    assert_eq!(row_block(&TWO_BY_FOUR, 4, 0, 4), TWO_BY_FOUR.to_vec());
}

#[test]
fn should_take_a_single_column_at_the_right_edge() {
    assert_eq!(row_block(&TWO_BY_FOUR, 4, 3, 1), vec![3, 7]);
}

#[test]
#[should_panic(expected = "the values must form whole rows")]
fn should_refuse_values_that_end_in_a_partial_row() {
    let _ = row_block(&TWO_BY_FOUR[..7], 4, 0, 2);
}

#[test]
fn should_return_nothing_for_an_empty_matrix() {
    assert_eq!(row_block::<u16>(&[], 4, 0, 2), Vec::<u16>::new());
}

#[test]
#[should_panic(expected = "the block must lie inside the row")]
fn should_refuse_a_block_that_runs_past_the_row() {
    let _ = row_block(&TWO_BY_FOUR, 4, 3, 2);
}
