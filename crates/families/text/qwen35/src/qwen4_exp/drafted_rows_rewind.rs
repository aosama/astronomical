//! Putting Flash-Next's per-request state back after a drafted block of
//! which only the first rows are kept.
//!
//! A verify forward runs the last chosen token and the drafted tokens after
//! it in one pass, and every row lands in the request's state. When the
//! target refuses a draft, the rows from there on must leave the state
//! again without running the kept rows a second time:
//!
//! - the indexer's keys only lower their row count, and a block key that
//!   averaged a refused row is dropped with it;
//! - the n-gram history is hashed again from the history before the forward
//!   and the kept token ids, on the host;
//! - the n-gram layer's convolution window is the rows before the forward
//!   followed by the kept rows, both of which the windows before and after
//!   the forward still hold while the forward is no longer than the window.

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;
use text_generation::execution_error::TextExecutionError;

use super::request_state::Qwen4ExpRequestState;

/// The rows a drafted block forwarded and how many of them are kept.
#[derive(Clone, Copy, Debug)]
pub struct KeptDraftedRows {
    pub forwarded_row_count: i32,
    pub kept_row_count: i32,
    /// Tokens one indexer block averages (`indexer_compress_ratio`).
    pub tokens_per_indexer_block: i32,
}

impl Qwen4ExpRequestState {
    /// Leaves this state as if only the first kept rows of the last forward
    /// had run. `before` is the snapshot taken before that forward and
    /// `kept_ngram_context` the history after the kept tokens.
    pub fn rewind_to_kept_drafted_rows(
        &mut self,
        runtime: &MlxRuntime,
        before: &Self,
        kept_ngram_context: Vec<u32>,
        rows: KeptDraftedRows,
    ) -> Result<(), TextExecutionError> {
        if rows.kept_row_count < 1 || rows.kept_row_count > rows.forwarded_row_count {
            return Err(TextExecutionError::InvalidInput {
                description: "a drafted block keeps at least its first row and at most all of them",
            });
        }
        if rows.kept_row_count == rows.forwarded_row_count {
            return Ok(());
        }
        self.ngram_convolution_window = rewound_convolution_window(
            runtime,
            before.ngram_convolution_window.as_ref(),
            self.ngram_convolution_window.as_ref(),
            rows,
        )?;
        self.ngram_context = kept_ngram_context;
        let dropped_row_count = rows.forwarded_row_count - rows.kept_row_count;
        for indexer in self.indexer_by_layer.iter_mut().flatten() {
            let kept_key_count = indexer.keys.used_rows() - dropped_row_count;
            indexer.keys.truncate_rows(kept_key_count);
            if let Some(block_keys) = indexer.block_keys.as_mut() {
                block_keys.truncate_rows(kept_key_count / rows.tokens_per_indexer_block.max(1));
            }
        }
        Ok(())
    }
}

/// The convolution window after the kept rows: the window before the
/// forward shifted by the kept rows, which sit in the window after it.
fn rewound_convolution_window(
    runtime: &MlxRuntime,
    window_before: Option<&MlxArray>,
    window_after: Option<&MlxArray>,
    rows: KeptDraftedRows,
) -> Result<Option<MlxArray>, TextExecutionError> {
    let Some(window_after) = window_after else {
        // The model has no n-gram layer state to rewind.
        return Ok(None);
    };
    let window_shape = window_after.shape();
    let (window_length, lane_width) = (window_shape[1], window_shape[2]);
    if rows.forwarded_row_count > window_length {
        return Err(TextExecutionError::InvalidInput {
            description: "a drafted block longer than the n-gram window cannot be rewound",
        });
    }
    let first_forwarded_row = window_length - rows.forwarded_row_count;
    let kept_rows = runtime.slice(
        window_after,
        &[0, first_forwarded_row, 0],
        &[1, first_forwarded_row + rows.kept_row_count, lane_width],
        &[1, 1, 1],
    )?;
    let rows_before = match window_before {
        Some(window_before) => runtime.slice(
            window_before,
            &[0, rows.kept_row_count, 0],
            &[1, window_length, lane_width],
            &[1, 1, 1],
        )?,
        // A request's first forward starts from a window of zeros.
        None => runtime.zeros(
            &[1, window_length - rows.kept_row_count, lane_width],
            window_after.dtype(),
        )?,
    };
    Ok(Some(
        runtime.concatenate_axis(&[&rows_before, &kept_rows], 1)?,
    ))
}

#[cfg(test)]
#[path = "drafted_rows_rewind_tests.rs"]
mod drafted_rows_rewind_tests;
