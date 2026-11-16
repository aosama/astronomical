use super::draft_tree_round::rows_beyond_the_context_limit;

#[test]
fn should_drop_only_the_rows_older_than_the_drafters_window() {
    assert_eq!(rows_beyond_the_context_limit(4_095, Some(2_047)), 2_048);
    assert_eq!(rows_beyond_the_context_limit(2_048, Some(2_047)), 1);
}

#[test]
fn should_keep_every_row_that_fits_the_window() {
    assert_eq!(rows_beyond_the_context_limit(2_047, Some(2_047)), 0);
    assert_eq!(rows_beyond_the_context_limit(1, Some(2_047)), 0);
    assert_eq!(rows_beyond_the_context_limit(0, Some(2_047)), 0);
}

#[test]
fn should_keep_every_row_for_a_drafter_that_attends_to_its_whole_context() {
    assert_eq!(rows_beyond_the_context_limit(1_000_000, None), 0);
}
