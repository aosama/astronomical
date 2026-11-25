//! The drafter's per-layer context keys and values: the part of a request's
//! `astro_decoder_state_cache` that belongs to the DFlash2 drafter.
//!
//! Only context rows are kept: projections of the fused target taps for
//! tokens the target has already committed. The drafted block's own keys and
//! values are written after them for one pass and dropped, because the block
//! is redrafted every round. A sliding-window layer shows at most
//! `sliding_window - 1` context rows, so with the block's rows a query never
//! sees more than the window.
//!
//! The rows live in one slab a layer that grows in place, as the target's
//! own attention state does (`FullAttentionKeyValueState`). Joining the kept
//! rows and the block with two concatenations copied every kept row twice a
//! layer every round: with 4,096 kept rows that was about 1.7 ms of a draft
//! round on a 35B-A3B at a 16 GB limit (7.6 ms against 5.9 with 64 rows).
//! A sliding-window layer's slab is cut back to its window once it holds
//! twice that, one copy every few thousand rows.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::drafter_configuration::{DFlashDrafterConfiguration, DrafterLayerAttention};
use crate::astro_decoder_state_cache::FullAttentionKeyValueState;

/// Rows a slab grows by when it is full.
const SLAB_GROWTH_ROWS: i32 = 256;

/// One layer's kept context: keys already rotated to their absolute positions.
pub struct DrafterLayerContextState {
    /// `[1, key-value heads, capacity, head dimension]`; its offset is the
    /// count of kept rows.
    slab: FullAttentionKeyValueState,
    /// Rows a query sees at most; None shows every row.
    retained_row_limit: Option<i32>,
    /// Absolute position of the next context row.
    next_context_position: i32,
}

/// The keys and values one block attends to, and how many of their rows
/// are kept context; the rest are the block's own.
pub struct AttendedRows {
    pub keys: MlxArray,
    pub values: MlxArray,
    pub kept_row_count: i32,
}

impl DrafterLayerContextState {
    fn new(retained_row_limit: Option<i32>, next_context_position: i32) -> Self {
        Self {
            slab: FullAttentionKeyValueState::empty_with_validated_growth_tokens(SLAB_GROWTH_ROWS),
            retained_row_limit,
            next_context_position,
        }
    }

    pub fn retained_row_limit(&self) -> Option<i32> {
        self.retained_row_limit
    }

    pub fn next_context_position(&self) -> i32 {
        self.next_context_position
    }

    /// Skips context rows the window will never show, advancing the position
    /// past them, as the reference does before projecting anything.
    pub fn skip_rows_beyond_window(&mut self, incoming_row_count: i32) -> i32 {
        match self.retained_row_limit {
            Some(row_limit) if incoming_row_count > row_limit => {
                let skipped_row_count = incoming_row_count - row_limit;
                self.next_context_position += skipped_row_count;
                skipped_row_count
            }
            _ => 0,
        }
    }

    /// Keeps this round's rotated context rows, writes the block's rows after
    /// them for this pass only, and returns the rows a query may see: the
    /// latest kept rows up to the window, then the block's.
    pub fn attended_rows(
        &mut self,
        runtime: &MlxRuntime,
        context_rows: (&MlxArray, &MlxArray),
        block_rows: (&MlxArray, &MlxArray),
    ) -> Result<AttendedRows, MlxRuntimeError> {
        let (context_keys, context_values) = context_rows;
        let (block_keys, block_values) = block_rows;
        self.next_context_position += context_keys.shape()[2];
        self.cut_back_to_the_window(runtime)?;
        let kept_before = self.slab.offset_tokens();
        if context_keys.shape()[2] > 0 {
            self.slab
                .update_and_fetch(runtime, context_keys, context_values, kept_before)?;
        }
        let kept_row_count = self.slab.offset_tokens();
        let (keys, values) =
            self.slab
                .update_and_fetch(runtime, block_keys, block_values, kept_row_count)?;
        // The block's rows are overwritten by the next round's context.
        self.slab.truncate_to_offset(kept_row_count)?;
        let first_shown_row = match self.retained_row_limit {
            Some(row_limit) => (kept_row_count - row_limit).max(0),
            None => 0,
        };
        Ok(AttendedRows {
            keys: rows_from(runtime, &keys, first_shown_row)?,
            values: rows_from(runtime, &values, first_shown_row)?,
            kept_row_count: kept_row_count - first_shown_row,
        })
    }

    /// Once a sliding-window layer keeps twice its window, keeps only the
    /// window's rows in a new slab.
    fn cut_back_to_the_window(&mut self, runtime: &MlxRuntime) -> Result<(), MlxRuntimeError> {
        let Some(row_limit) = self.retained_row_limit else {
            return Ok(());
        };
        let kept_row_count = self.slab.offset_tokens();
        let (Some(keys), Some(values)) = (self.slab.keys_state(), self.slab.values_state()) else {
            return Ok(());
        };
        if kept_row_count <= 2 * row_limit {
            return Ok(());
        }
        let first_row = kept_row_count - row_limit;
        let window_keys = row_range(runtime, keys, first_row, kept_row_count)?;
        let window_values = row_range(runtime, values, first_row, kept_row_count)?;
        let mut window_slab =
            FullAttentionKeyValueState::empty_with_validated_growth_tokens(SLAB_GROWTH_ROWS);
        window_slab.update_and_fetch(runtime, &window_keys, &window_values, 0)?;
        self.slab = window_slab;
        Ok(())
    }
}

/// `rows` `[1, heads, rows, dimension]` from `first_row` to the end.
fn rows_from(
    runtime: &MlxRuntime,
    rows: &MlxArray,
    first_row: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    if first_row == 0 {
        return Ok(rows.retain()?);
    }
    row_range(runtime, rows, first_row, rows.shape()[2])
}

fn row_range(
    runtime: &MlxRuntime,
    rows: &MlxArray,
    first_row: i32,
    end_row: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let shape = rows.shape();
    Ok(runtime.slice(
        rows,
        &[0, 0, first_row, 0],
        &[shape[0], shape[1], end_row, shape[3]],
        &[1, 1, 1, 1],
    )?)
}

/// Every drafter layer's context state for one request.
pub struct DrafterContextStates {
    pub layers: Vec<DrafterLayerContextState>,
}

impl DrafterContextStates {
    /// Empty states whose first context row sits at `first_context_position`:
    /// 0 for a request drafted from its first token, or the prompt length minus
    /// the rows handed over when only the prompt's tail is passed in.
    pub fn new(configuration: &DFlashDrafterConfiguration, first_context_position: i32) -> Self {
        let mut layers = Vec::with_capacity(configuration.layer_attention.len());
        for layer_attention in &configuration.layer_attention {
            layers.push(DrafterLayerContextState::new(
                match layer_attention {
                    DrafterLayerAttention::SlidingWindow => Some(configuration.sliding_window - 1),
                    DrafterLayerAttention::Full => None,
                },
                first_context_position,
            ));
        }
        Self { layers }
    }
}
