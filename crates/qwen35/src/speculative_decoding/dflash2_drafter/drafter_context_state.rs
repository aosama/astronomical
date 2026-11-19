//! The drafter's per-layer context keys and values: the part of a request's
//! `astro_decoder_state_cache` that belongs to the DFlash2 drafter.
//!
//! Only context rows are kept: projections of the fused target taps for
//! tokens the target has already committed. The drafted block's own keys and
//! values are joined for one pass and dropped, because the block is redrafted
//! every round. A sliding-window layer keeps at most `sliding_window - 1`
//! rows, so with the block's rows a query never sees more than the window.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::drafter_configuration::{DFlash2DrafterConfiguration, DrafterLayerAttention};

/// One layer's kept context: keys already rotated to their absolute positions.
pub struct DrafterLayerContextState {
    /// `[1, key-value heads, rows, head dimension]`, None before the first round.
    keys_and_values: Option<(MlxArray, MlxArray)>,
    /// Rows kept at most; None keeps every row.
    retained_row_limit: Option<i32>,
    /// Absolute position of the next context row.
    next_context_position: i32,
}

impl DrafterLayerContextState {
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

    /// Appends this round's rotated context keys and values and returns every
    /// kept row in temporal order.
    pub fn append_and_fetch(
        &mut self,
        runtime: &MlxRuntime,
        new_keys: MlxArray,
        new_values: MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
        self.next_context_position += new_keys.shape()[2];
        let (joined_keys, joined_values) = match self.keys_and_values.take() {
            Some((kept_keys, kept_values)) => (
                runtime.concatenate_axis(&[&kept_keys, &new_keys], 2)?,
                runtime.concatenate_axis(&[&kept_values, &new_values], 2)?,
            ),
            None => (new_keys, new_values),
        };
        let trimmed_keys = self.keep_latest_rows(runtime, joined_keys)?;
        let trimmed_values = self.keep_latest_rows(runtime, joined_values)?;
        self.keys_and_values = Some((trimmed_keys.retain()?, trimmed_values.retain()?));
        Ok((trimmed_keys, trimmed_values))
    }

    fn keep_latest_rows(
        &self,
        runtime: &MlxRuntime,
        rows: MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let shape = rows.shape();
        match self.retained_row_limit {
            Some(row_limit) if shape[2] > row_limit => Ok(runtime.slice(
                &rows,
                &[0, 0, shape[2] - row_limit, 0],
                &[shape[0], shape[1], shape[2], shape[3]],
                &[1, 1, 1, 1],
            )?),
            _ => Ok(rows),
        }
    }
}

/// Every drafter layer's context state for one request.
pub struct DrafterContextStates {
    pub layers: Vec<DrafterLayerContextState>,
}

impl DrafterContextStates {
    /// Empty states whose first context row sits at `first_context_position`:
    /// 0 for a request drafted from its first token, or the prompt length minus
    /// the rows handed over when only the prompt's tail is passed in.
    pub fn new(configuration: &DFlash2DrafterConfiguration, first_context_position: i32) -> Self {
        let mut layers = Vec::with_capacity(configuration.layer_attention.len());
        for layer_attention in &configuration.layer_attention {
            layers.push(DrafterLayerContextState {
                keys_and_values: None,
                retained_row_limit: match layer_attention {
                    DrafterLayerAttention::SlidingWindow => Some(configuration.sliding_window - 1),
                    DrafterLayerAttention::Full => None,
                },
                next_context_position: first_context_position,
            });
        }
        Self { layers }
    }
}
