//! Where every row of Flash-Next's n-gram table lives.
//!
//! Each n-gram head hashes into a table of its own whose size is a prime:
//! the first head's is the first prime past the configured base, and every
//! next head's the next prime after that. The heads' tables are concatenated
//! in head order, padded to the configured alignment, and the result is
//! split into equal-as-possible shards that the checkpoint stores as
//! separate tensors. This geometry is fixed by the configuration alone, so
//! the artifact validator can name every shard's shape before reading any
//! byte, and the row reader can map a row to its shard without a table.

use crate::qwen3_5_core::configuration::Qwen4ExpArchitecture;

/// The layout of the n-gram table one Flash-Next checkpoint declares.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramTableGeometry {
    head_vocabulary_sizes: Vec<u64>,
    head_row_offsets: Vec<u64>,
    shard_first_rows: Vec<u64>,
    padded_row_count: u64,
}

impl NgramTableGeometry {
    /// The geometry of the table on the n-gram layer, which is the first
    /// (and only) layer the wire lists, so its heads draw primes from the
    /// first one on.
    #[must_use]
    pub fn for_architecture(architecture: &Qwen4ExpArchitecture) -> Self {
        let head_count = architecture.ngram_head_count() as usize;
        let mut head_vocabulary_sizes = Vec::with_capacity(head_count);
        let mut head_row_offsets = Vec::with_capacity(head_count);
        let mut previous_prime = architecture.ngram_vocabulary_size_base - 1;
        let mut row_count = 0_u64;
        for _ in 0..head_count {
            previous_prime = next_prime_after(previous_prime);
            head_vocabulary_sizes.push(previous_prime);
            head_row_offsets.push(row_count);
            row_count += previous_prime;
        }
        let padded_row_count = row_count.div_ceil(architecture.ngram_vocabulary_size_alignment)
            * architecture.ngram_vocabulary_size_alignment;
        Self {
            head_vocabulary_sizes,
            head_row_offsets,
            shard_first_rows: shard_first_rows(
                padded_row_count,
                u64::from(architecture.ngram_table_shard_count),
            ),
            padded_row_count,
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
    pub const fn padded_row_count(&self) -> u64 {
        self.padded_row_count
    }

    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shard_first_rows.len()
    }

    /// Rows stored in one shard tensor.
    #[must_use]
    pub fn shard_row_count(&self, shard_index: usize) -> u64 {
        let shard_end = self
            .shard_first_rows
            .get(shard_index + 1)
            .copied()
            .unwrap_or(self.padded_row_count);
        shard_end - self.shard_first_rows[shard_index]
    }

    /// The shard holding `row` and the row's index inside it, or `None` for
    /// a row past the table.
    #[must_use]
    pub fn shard_and_local_row(&self, row: u64) -> Option<(usize, u64)> {
        if row >= self.padded_row_count {
            return None;
        }
        let shard_index = self
            .shard_first_rows
            .partition_point(|first_row| *first_row <= row)
            - 1;
        Some((shard_index, row - self.shard_first_rows[shard_index]))
    }
}

/// The first row of each shard. The first `remainder` shards take one row
/// more than the rest, as the checkpoint's own splitter does.
fn shard_first_rows(padded_row_count: u64, shard_count: u64) -> Vec<u64> {
    let base_rows = padded_row_count / shard_count;
    let remainder = padded_row_count % shard_count;
    let mut first_rows = Vec::with_capacity(shard_count as usize);
    let mut next_first_row = 0;
    for shard_index in 0..shard_count {
        first_rows.push(next_first_row);
        next_first_row += base_rows + u64::from(shard_index < remainder);
    }
    first_rows
}

fn next_prime_after(start: u64) -> u64 {
    let mut candidate = start + 1;
    while !is_prime(candidate) {
        candidate += 1;
    }
    candidate
}

/// Trial division is enough: candidates sit near twenty million, so each
/// test divides by at most a few thousand odd numbers, once per load.
fn is_prime(candidate: u64) -> bool {
    if candidate < 2 {
        return false;
    }
    if candidate.is_multiple_of(2) {
        return candidate == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= candidate {
        if candidate.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

#[cfg(test)]
#[path = "ngram_table_geometry_tests.rs"]
mod ngram_table_geometry_tests;
