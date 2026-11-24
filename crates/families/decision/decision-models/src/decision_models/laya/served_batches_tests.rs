//! Which forward passes score a training's questions as serving did.

use super::*;

fn sequence_row(token_count: usize) -> SequenceRow {
    SequenceRow {
        token_ids: vec![7; token_count],
        question_type_index: 0,
        option_marker_positions: vec![1, 2],
    }
}

/// One served pass of questions with `token_counts`, the first being training row `first_row`.
fn served_pass(token_counts: &[usize], first_row: usize) -> ServedPass {
    let mut sequence_rows = Vec::new();
    let mut training_rows = Vec::new();
    for (question_offset, token_count) in token_counts.iter().enumerate() {
        sequence_rows.push(sequence_row(*token_count));
        training_rows.push(Some(first_row + question_offset));
    }
    ServedPass {
        sequence_rows,
        training_rows,
    }
}

fn four_passes() -> ServedBatches {
    ServedBatches {
        passes: vec![
            served_pass(&[12], 0),
            served_pass(&[12], 1),
            served_pass(&[12, 30], 2),
            served_pass(&[20], 4),
        ],
        row_passes: vec![0, 1, 2, 2, 3],
    }
}

#[test]
fn should_run_every_pass_that_holds_an_asked_row_once_and_on_its_own() {
    assert_eq!(four_passes().passes_holding(&[4, 0, 1]), vec![0, 1, 3]);
    // Rows 2 and 3 were served together, so both come from their one pass.
    assert_eq!(four_passes().passes_holding(&[3, 2]), vec![2]);
    assert!(four_passes().passes_holding(&[]).is_empty());
}
