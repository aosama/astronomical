//! A per-token sequence held in storage that grows in fixed steps.
//!
//! The indexer's keys and block keys gain rows on every reply token. Joining
//! each token's rows onto the last array would allocate a new array per
//! token and leave the old one in the `mlx_buffer_cache`, and the request's
//! byte count would grow on every token, which makes the reply path empty
//! the `mlx_buffer_cache` after every token (see
//! `reply_token_mlx_buffer_cache_reclaim`, measured at 10 tokens per second
//! against 28). Writing into capacity reserved in steps, as the attention
//! keys and values already do, allocates once per step instead.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};
use text_generation::execution_error::TextExecutionError;

/// Rows a growing sequence reserves at a time.
pub const SEQUENCE_GROWTH_ROWS: i32 = 256;

/// Rows along one axis of a fixed-rank array, in reserved storage.
pub struct GrowingSequence {
    storage: MlxArray,
    used_rows: i32,
    axis: usize,
}

impl GrowingSequence {
    /// A sequence holding `first_rows`, which grows along `axis`.
    pub fn starting_with(
        runtime: &MlxRuntime,
        first_rows: &MlxArray,
        axis: usize,
    ) -> Result<Self, TextExecutionError> {
        let used_rows = first_rows.shape()[axis];
        let storage = reserved_copy(runtime, first_rows, axis, used_rows)?;
        Ok(Self {
            storage,
            used_rows,
            axis,
        })
    }

    /// Writes `new_rows` after the used rows, reserving another step when
    /// they do not fit.
    pub fn append(
        &mut self,
        runtime: &MlxRuntime,
        new_rows: &MlxArray,
    ) -> Result<(), TextExecutionError> {
        let added_rows = new_rows.shape()[self.axis];
        let needed_rows = self.used_rows + added_rows;
        if needed_rows > self.storage.shape()[self.axis] {
            let used = self.used(runtime)?;
            let grown = runtime.concatenate_axis(&[&used, new_rows], self.axis as i32)?;
            self.storage = reserved_copy(runtime, &grown, self.axis, needed_rows)?;
        } else {
            let (starts, stops) =
                row_window(&new_rows.shape(), self.axis, self.used_rows, needed_rows);
            let strides = vec![1; starts.len()];
            self.storage =
                runtime.slice_update(&self.storage, new_rows, &starts, &stops, &strides)?;
        }
        self.used_rows = needed_rows;
        Ok(())
    }

    /// The used rows, as a view.
    pub fn used(&self, runtime: &MlxRuntime) -> Result<MlxArray, MlxRuntimeError> {
        let shape = self.storage.shape();
        let (starts, stops) = row_window(&shape, self.axis, 0, self.used_rows);
        let strides = vec![1; starts.len()];
        Ok(runtime.slice(&self.storage, &starts, &stops, &strides)?)
    }

    /// Forgets the rows after the first `kept_rows`. Their storage stays
    /// reserved and the next append writes over it, so dropping the rows of
    /// refused draft tokens allocates nothing.
    pub fn truncate_rows(&mut self, kept_rows: i32) {
        self.used_rows = self.used_rows.min(kept_rows.max(0));
    }

    #[must_use]
    pub const fn used_rows(&self) -> i32 {
        self.used_rows
    }

    /// The reserved storage, which is what the request holds in memory.
    #[must_use]
    pub fn storage(&self) -> &MlxArray {
        &self.storage
    }

    pub fn retained(&self) -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            storage: self.storage.retain()?,
            used_rows: self.used_rows,
            axis: self.axis,
        })
    }
}

/// `rows` followed by empty rows up to the next whole step past `row_count`.
fn reserved_copy(
    runtime: &MlxRuntime,
    rows: &MlxArray,
    axis: usize,
    row_count: i32,
) -> Result<MlxArray, TextExecutionError> {
    let capacity = reserved_row_count(row_count);
    let mut padding_shape = rows.shape();
    padding_shape[axis] = capacity - row_count;
    if padding_shape[axis] == 0 {
        return Ok(rows.retain()?);
    }
    let padding = runtime.zeros(&padding_shape, rows.dtype())?;
    Ok(runtime.concatenate_axis(&[rows, &padding], axis as i32)?)
}

/// The capacity that holds `row_count` rows: whole steps, at least one.
#[must_use]
pub fn reserved_row_count(row_count: i32) -> i32 {
    (row_count.max(1) + SEQUENCE_GROWTH_ROWS - 1) / SEQUENCE_GROWTH_ROWS * SEQUENCE_GROWTH_ROWS
}

/// Start and stop vectors selecting rows `first..end` of `axis`.
fn row_window(shape: &[i32], axis: usize, first: i32, end: i32) -> (Vec<i32>, Vec<i32>) {
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.to_vec();
    starts[axis] = first;
    stops[axis] = end;
    (starts, stops)
}

#[cfg(test)]
#[path = "growing_sequence_tests.rs"]
mod growing_sequence_tests;
