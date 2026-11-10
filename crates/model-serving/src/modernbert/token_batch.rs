//! A batch of token sequences padded to one length for a single forward pass.
//!
//! The batch is built on the host, before any lazy graph exists, so a batch
//! the encoder cannot run fails here and never reaches the graphics processor.

use thiserror::Error;

/// Token sequences laid out row by row, each padded on the right to the longest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PaddedTokenBatch {
    /// `sequence_count * padded_length` identifiers, row-major.
    token_ids: Vec<i32>,
    padded_length: usize,
    /// The real token count of each row, before padding.
    sequence_lengths: Vec<usize>,
}

/// Why sequences cannot form a batch.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub(crate) enum TokenBatchError {
    #[error("a token batch needs at least one sequence")]
    NoSequences,
    #[error("sequence {sequence_index} of the token batch has no tokens")]
    EmptySequence { sequence_index: usize },
    #[error("the token batch does not fit the 32-bit index range the encoder uses")]
    IndexRangeExceeded,
}

impl PaddedTokenBatch {
    /// Pads `token_sequences` on the right with `pad_token_id` to the longest one.
    pub(crate) fn from_sequences(
        token_sequences: &[&[u32]],
        pad_token_id: u32,
    ) -> Result<Self, TokenBatchError> {
        if token_sequences.is_empty() {
            return Err(TokenBatchError::NoSequences);
        }
        let mut padded_length = 0;
        for (sequence_index, token_sequence) in token_sequences.iter().enumerate() {
            if token_sequence.is_empty() {
                return Err(TokenBatchError::EmptySequence { sequence_index });
            }
            padded_length = padded_length.max(token_sequence.len());
        }
        if i32::try_from(padded_length).is_err() || i32::try_from(token_sequences.len()).is_err() {
            return Err(TokenBatchError::IndexRangeExceeded);
        }
        let pad_token_id =
            i32::try_from(pad_token_id).map_err(|_| TokenBatchError::IndexRangeExceeded)?;
        let mut token_ids = Vec::with_capacity(token_sequences.len() * padded_length);
        let mut sequence_lengths = Vec::with_capacity(token_sequences.len());
        for token_sequence in token_sequences {
            for token_id in *token_sequence {
                token_ids.push(
                    i32::try_from(*token_id).map_err(|_| TokenBatchError::IndexRangeExceeded)?,
                );
            }
            for _padding_position in token_sequence.len()..padded_length {
                token_ids.push(pad_token_id);
            }
            sequence_lengths.push(token_sequence.len());
        }
        Ok(Self {
            token_ids,
            padded_length,
            sequence_lengths,
        })
    }

    pub(crate) fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    pub(crate) fn sequence_count(&self) -> usize {
        self.sequence_lengths.len()
    }

    pub(crate) const fn padded_length(&self) -> usize {
        self.padded_length
    }

    pub(crate) fn sequence_lengths(&self) -> &[usize] {
        &self.sequence_lengths
    }

    /// Whether any row is shorter than the padded length and so carries padding.
    pub(crate) fn has_padding(&self) -> bool {
        for sequence_length in &self.sequence_lengths {
            if *sequence_length != self.padded_length {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
#[path = "token_batch_tests.rs"]
mod tests;
