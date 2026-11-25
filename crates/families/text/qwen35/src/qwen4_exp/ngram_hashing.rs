//! Which n-gram table rows a stretch of tokens reads.
//!
//! For every token Flash-Next hashes the pair (previous, current) into eight
//! table heads and the triple (two back, previous, current) into eight more,
//! and adds the sixteen gathered rows to its residual on one layer. History
//! does not cross an end-of-sequence token: a token whose earlier neighbours
//! lie before the last end-of-sequence sees that token in their place, as the
//! reference's `_shift_right_ignore_eos` does.
//!
//! This runs on the host: the row ids decide which bytes to read from the
//! SSD, so they are needed before any graph that uses them is built, and
//! sixteen integer hashes per token cost nothing next to the read itself.

/// The hash constants the checkpoint ships with its table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramHashTables {
    /// One odd multiplier per position of the longest n-gram.
    multipliers: Vec<u64>,
    head_vocabulary_sizes: Vec<u64>,
    head_row_offsets: Vec<u64>,
    heads_per_order: usize,
    end_of_sequence_token_id: u32,
}

impl NgramHashTables {
    /// `multipliers` has one entry per n-gram position; the head tables are
    /// ordered pairs first, then triples, `heads_per_order` each.
    #[must_use]
    pub fn new(
        multipliers: Vec<u64>,
        head_vocabulary_sizes: Vec<u64>,
        head_row_offsets: Vec<u64>,
        heads_per_order: usize,
        end_of_sequence_token_id: u32,
    ) -> Self {
        Self {
            multipliers,
            head_vocabulary_sizes,
            head_row_offsets,
            heads_per_order,
            end_of_sequence_token_id,
        }
    }

    #[must_use]
    pub fn head_vocabulary_sizes(&self) -> &[u64] {
        &self.head_vocabulary_sizes
    }

    #[must_use]
    pub fn head_row_offsets(&self) -> &[u64] {
        &self.head_row_offsets
    }

    #[must_use]
    pub fn head_count(&self) -> usize {
        self.head_vocabulary_sizes.len()
    }

    /// Tokens of history a stretch needs before its first token.
    #[must_use]
    pub fn context_length(&self) -> usize {
        self.multipliers.len() - 1
    }

    /// The history a request starts with: nothing before the first token,
    /// which reads as end-of-sequence.
    #[must_use]
    pub fn initial_context(&self) -> Vec<u32> {
        vec![self.end_of_sequence_token_id; self.context_length()]
    }

    /// The table rows of every token in `tokens`, token-major: row
    /// `token * head_count + head`. `context` is the history before the
    /// first token, `context_length` tokens long.
    #[must_use]
    pub fn row_ids(&self, context: &[u32], tokens: &[u32]) -> Vec<u64> {
        let history = [context, tokens].concat();
        let segment_starts = segment_starts(&history, self.end_of_sequence_token_id);
        let head_count = self.head_vocabulary_sizes.len();
        let mut row_ids = Vec::with_capacity(tokens.len() * head_count);
        for history_position in context.len()..history.len() {
            let mut mixed = 0_u64;
            for (shift, multiplier) in self.multipliers.iter().enumerate() {
                let token = self.shifted_token(&history, &segment_starts, history_position, shift);
                mixed ^= u64::from(token).wrapping_mul(*multiplier);
                // Shift 0 alone is not an n-gram; shift 1 closes the pair,
                // shift 2 the triple.
                if shift >= 1 {
                    self.append_order_rows(mixed, shift - 1, &mut row_ids);
                }
            }
        }
        row_ids
    }

    /// The history the next stretch starts from: the last `context_length`
    /// tokens of this one's history.
    #[must_use]
    pub fn next_context(&self, context: &[u32], tokens: &[u32]) -> Vec<u32> {
        let history = [context, tokens].concat();
        history[history.len() - self.context_length()..].to_vec()
    }

    fn shifted_token(
        &self,
        history: &[u32],
        segment_starts: &[usize],
        history_position: usize,
        shift: usize,
    ) -> u32 {
        let position_in_segment = history_position - segment_starts[history_position];
        if position_in_segment >= shift {
            history[history_position - shift]
        } else {
            self.end_of_sequence_token_id
        }
    }

    fn append_order_rows(&self, mixed: u64, order_index: usize, row_ids: &mut Vec<u64>) {
        let first_head = order_index * self.heads_per_order;
        for head in first_head..first_head + self.heads_per_order {
            row_ids.push(mixed % self.head_vocabulary_sizes[head] + self.head_row_offsets[head]);
        }
    }
}

/// For each history position, the first position after the last
/// end-of-sequence token strictly before it (zero when there is none).
fn segment_starts(history: &[u32], end_of_sequence_token_id: u32) -> Vec<usize> {
    let mut starts = Vec::with_capacity(history.len());
    let mut next_start = 0;
    for (position, token) in history.iter().enumerate() {
        starts.push(next_start);
        if *token == end_of_sequence_token_id {
            next_start = position + 1;
        }
    }
    starts
}

#[cfg(test)]
#[path = "ngram_hashing_tests.rs"]
mod ngram_hashing_tests;
