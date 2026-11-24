//! Contracts for planning embedding row reads: distinct sorted rows, token
//! positions among them, merged intervals, and rejected token ids.

use super::embedding_row_reads::{
    EmbeddingRowReadError, EmbeddingTableLayout, embedding_row_intervals, select_embedding_rows,
};

/// A 6-bit table of 4096-wide rows: 3072 packed weight bytes per row.
const SIX_BIT_WEIGHT_TABLE: EmbeddingTableLayout = EmbeddingTableLayout {
    first_row_file_offset: 1_000_000,
    row_byte_count: 3072,
    row_count: 248_320,
};

#[test]
fn should_read_each_repeated_token_once_and_point_every_token_at_its_row() {
    let row_selection = select_embedding_rows(&[42, 7, 42, 9], 248_320)
        .expect("valid token ids should select rows");

    assert_eq!(row_selection.distinct_row_ids, [7, 9, 42]);
    assert_eq!(row_selection.selected_row_index_by_token, [2, 0, 2, 1]);
}

#[test]
fn should_select_the_first_and_last_rows_of_the_table() {
    let row_selection =
        select_embedding_rows(&[248_319, 0], 248_320).expect("the table edges should select");

    assert_eq!(row_selection.distinct_row_ids, [0, 248_319]);
    assert_eq!(row_selection.selected_row_index_by_token, [1, 0]);
}

#[test]
fn should_reject_a_token_one_past_the_last_row() {
    let row_selection_error =
        select_embedding_rows(&[3, 248_320], 248_320).expect_err("an id past the table must fail");

    assert_eq!(
        row_selection_error,
        EmbeddingRowReadError::TokenOutsideTable {
            token_id: 248_320,
            row_count: 248_320,
        }
    );
}

#[test]
fn should_reject_a_negative_token_id() {
    let row_selection_error =
        select_embedding_rows(&[-1], 248_320).expect_err("a negative id must fail");

    assert_eq!(
        row_selection_error,
        EmbeddingRowReadError::TokenOutsideTable {
            token_id: -1,
            row_count: 248_320,
        }
    );
}

#[test]
fn should_reject_a_read_with_no_tokens() {
    assert_eq!(
        select_embedding_rows(&[], 248_320),
        Err(EmbeddingRowReadError::NoTokens)
    );
}

#[test]
fn should_merge_consecutive_rows_into_one_interval_and_split_at_gaps() {
    let row_intervals = embedding_row_intervals(SIX_BIT_WEIGHT_TABLE, &[4, 5, 6, 10]);

    assert_eq!(row_intervals.len(), 2);
    assert_eq!(row_intervals[0].source_file_offset, 1_000_000 + 4 * 3072);
    assert_eq!(row_intervals[0].source_byte_count, 3 * 3072);
    assert_eq!(row_intervals[0].virtual_payload_offset, 0);
    assert_eq!(row_intervals[1].source_file_offset, 1_000_000 + 10 * 3072);
    assert_eq!(row_intervals[1].source_byte_count, 3072);
    assert_eq!(
        row_intervals[1].virtual_payload_offset,
        3 * 3072,
        "the virtual payload stays contiguous across a gap in the file"
    );
}

#[test]
fn should_reject_the_largest_token_id_in_a_smaller_table() {
    assert_eq!(
        select_embedding_rows(&[i32::MAX], 248_320),
        Err(EmbeddingRowReadError::TokenOutsideTable {
            token_id: i64::from(i32::MAX),
            row_count: 248_320,
        })
    );
}

#[test]
fn should_plan_one_interval_per_row_when_no_rows_touch() {
    let row_intervals = embedding_row_intervals(SIX_BIT_WEIGHT_TABLE, &[1, 3, 5]);

    assert_eq!(row_intervals.len(), 3);
}
