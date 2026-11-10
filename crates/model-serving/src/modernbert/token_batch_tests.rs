//! Unit tests for padding token sequences into one batch.

use super::{PaddedTokenBatch, TokenBatchError};

const PAD_TOKEN_ID: u32 = 9;

#[test]
fn should_pad_shorter_sequences_on_the_right_to_the_longest() {
    let token_batch = PaddedTokenBatch::from_sequences(&[&[1, 2, 3], &[4], &[5, 6]], PAD_TOKEN_ID)
        .expect("a valid batch");
    assert_eq!(token_batch.token_ids(), [1, 2, 3, 4, 9, 9, 5, 6, 9]);
    assert_eq!(token_batch.sequence_count(), 3);
    assert_eq!(token_batch.padded_length(), 3);
    assert_eq!(token_batch.sequence_lengths(), [3, 1, 2]);
    assert!(token_batch.has_padding());
}

#[test]
fn should_report_no_padding_when_every_sequence_has_the_same_length() {
    let token_batch =
        PaddedTokenBatch::from_sequences(&[&[1, 2], &[3, 4]], PAD_TOKEN_ID).expect("a valid batch");
    assert_eq!(token_batch.token_ids(), [1, 2, 3, 4]);
    assert!(!token_batch.has_padding());
}

#[test]
fn should_report_no_padding_for_a_single_sequence() {
    let token_batch =
        PaddedTokenBatch::from_sequences(&[&[7]], PAD_TOKEN_ID).expect("a valid batch");
    assert_eq!(token_batch.padded_length(), 1);
    assert!(!token_batch.has_padding());
}

#[test]
fn should_reject_a_batch_with_no_sequences() {
    assert_eq!(
        PaddedTokenBatch::from_sequences(&[], PAD_TOKEN_ID),
        Err(TokenBatchError::NoSequences)
    );
}

#[test]
fn should_reject_an_empty_sequence_and_name_its_position() {
    assert_eq!(
        PaddedTokenBatch::from_sequences(&[&[1], &[]], PAD_TOKEN_ID),
        Err(TokenBatchError::EmptySequence { sequence_index: 1 })
    );
}

#[test]
fn should_accept_the_largest_identifier_the_index_range_holds_and_reject_one_more() {
    let largest_identifier = i32::MAX as u32;
    assert!(PaddedTokenBatch::from_sequences(&[&[largest_identifier]], PAD_TOKEN_ID).is_ok());
    assert_eq!(
        PaddedTokenBatch::from_sequences(&[&[largest_identifier + 1]], PAD_TOKEN_ID),
        Err(TokenBatchError::IndexRangeExceeded)
    );
    assert_eq!(
        PaddedTokenBatch::from_sequences(&[&[1]], largest_identifier + 1),
        Err(TokenBatchError::IndexRangeExceeded)
    );
}
