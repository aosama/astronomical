//! Which bytes to read for the embedding rows of a set of tokens.
//!
//! A token embedding is one row of the embedding table, and for a quantized
//! table one row each of the packed weight, the scales and the biases. A
//! model that does not fit its limit reads only the rows a forward needs
//! instead of keeping the whole table in memory (issue #1985). This module is
//! the arithmetic of that read: which distinct rows, in which order, and
//! which byte intervals of the shard they occupy. It reads nothing itself.

use astronomical_runtime_integration::BoundedReadInterval;

/// Where one row-major table lives in its shard: the absolute offset of row
/// zero and the bytes of one row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EmbeddingTableLayout {
    pub(crate) first_row_file_offset: u64,
    pub(crate) row_byte_count: usize,
    pub(crate) row_count: usize,
}

/// The distinct rows one forward needs, in ascending order, and where each
/// requested token finds its row among them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EmbeddingRowSelection {
    pub(crate) distinct_row_ids: Vec<usize>,
    pub(crate) selected_row_index_by_token: Vec<i32>,
}

/// Failures planning an embedding row read.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum EmbeddingRowReadError {
    #[error("token id {token_id} is outside the {row_count}-row embedding table")]
    TokenOutsideTable { token_id: i64, row_count: usize },
    #[error("an embedding row read needs at least one token")]
    NoTokens,
}

/// Deduplicates and sorts the token ids of one forward.
///
/// Sorted rows let adjacent token ids share one read, and deduplication
/// keeps a prompt that repeats a word from reading its row twice.
pub(crate) fn select_embedding_rows(
    token_ids: &[i32],
    row_count: usize,
) -> Result<EmbeddingRowSelection, EmbeddingRowReadError> {
    if token_ids.is_empty() {
        return Err(EmbeddingRowReadError::NoTokens);
    }
    let mut row_id_and_token_position = Vec::with_capacity(token_ids.len());
    for (token_position, token_id) in token_ids.iter().enumerate() {
        let row_id = match usize::try_from(*token_id) {
            Ok(row_id) if row_id < row_count => row_id,
            _ => {
                return Err(EmbeddingRowReadError::TokenOutsideTable {
                    token_id: i64::from(*token_id),
                    row_count,
                });
            }
        };
        row_id_and_token_position.push((row_id, token_position));
    }
    // One sorted pass both deduplicates the rows and records where each
    // token's row lands among them.
    row_id_and_token_position.sort_unstable();
    let mut distinct_row_ids: Vec<usize> = Vec::with_capacity(token_ids.len());
    let mut selected_row_index_by_token = vec![0_i32; token_ids.len()];
    for (row_id, token_position) in row_id_and_token_position {
        if distinct_row_ids.last() != Some(&row_id) {
            distinct_row_ids.push(row_id);
        }
        selected_row_index_by_token[token_position] = (distinct_row_ids.len() - 1) as i32;
    }
    Ok(EmbeddingRowSelection {
        distinct_row_ids,
        selected_row_index_by_token,
    })
}

/// The shard intervals holding `distinct_row_ids` of one table, placed in a
/// virtual payload that starts at zero.
///
/// Runs of consecutive rows merge into one interval, so a prompt of nearby
/// token ids becomes a few reads rather than one per token.
pub(crate) fn embedding_row_intervals(
    table_layout: EmbeddingTableLayout,
    distinct_row_ids: &[usize],
) -> Vec<BoundedReadInterval> {
    let mut row_intervals: Vec<BoundedReadInterval> = Vec::new();
    let mut virtual_payload_offset = 0_u64;
    let mut previous_row_id: Option<usize> = None;
    for row_id in distinct_row_ids {
        let continues_previous_run =
            matches!(previous_row_id, Some(previous_row) if previous_row + 1 == *row_id);
        match row_intervals.last_mut() {
            Some(current_interval) if continues_previous_run => {
                current_interval.source_byte_count += table_layout.row_byte_count;
            }
            _ => row_intervals.push(BoundedReadInterval {
                virtual_payload_offset,
                source_file_offset: table_layout.first_row_file_offset
                    + (*row_id as u64) * table_layout.row_byte_count as u64,
                source_byte_count: table_layout.row_byte_count,
            }),
        }
        virtual_payload_offset += table_layout.row_byte_count as u64;
        previous_row_id = Some(*row_id);
    }
    row_intervals
}
