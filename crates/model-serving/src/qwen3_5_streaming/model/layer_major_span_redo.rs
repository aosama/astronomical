//! Putting a layer-at-a-time prompt back to where a span began, so a span
//! that ran out of memory is run again and nothing else is (issue #2221).
//!
//! A span's work is submitted and the engine moves on; MLX reports a
//! failure only when the next span, or the layer's end, waits for it. Until
//! that wait succeeds the span is unconfirmed. At most two spans are ever
//! unconfirmed: the one being built and the one before it. Each keeps what
//! it found when it began: its input, the layer's state, and the output the
//! next wait was going to be for. All three are handles to arrays that are
//! alive anyway, so a redo point costs one span's input held one span
//! longer.
//!
//! Redoing a whole layer instead would hold every span's input until the
//! layer ended, a quarter to a half of the prompt's stored state on the
//! models measured; redoing the whole prompt, which is what this replaces,
//! read every layer again.

use mlx_bridge::MlxArray;

use super::layer_major_span_state::{SpanState, forget_layer_end_state};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::{DecoderCacheStateAllocationCheckpoint, RequestDecoderStateStack};

/// What one span found when it began to run through one layer.
struct SpanRedoPoint {
    span_index: usize,
    span_input: MlxArray,
    layer_state: DecoderCacheStateAllocationCheckpoint,
    previous_layer_output: Option<MlxArray>,
}

/// The redo points of the spans MLX has not confirmed yet, oldest first.
#[derive(Default)]
pub(super) struct UnconfirmedSpans {
    redo_points: Vec<SpanRedoPoint>,
}

impl UnconfirmedSpans {
    /// Remembers what span `span_index` finds as it begins layer
    /// `layer_index`.
    pub(super) fn remember_span_start(
        &mut self,
        layer_index: usize,
        span_index: usize,
        span_states: &[SpanState],
        request_decoder_state: &RequestDecoderStateStack,
        previous_layer_output: Option<&MlxArray>,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let span_input = span_states
            .get(span_index)
            .and_then(|span_state| span_state.hidden_states.as_ref())
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "a layer-major span lost its hidden states",
            })?
            .retain()?;
        let layer_state = request_decoder_state
            .layer(layer_index)
            .ok_or(Qwen3_5ExecutionError::InvalidRequestDecoderState {
                layer_index,
                description: "decoder layer entry is missing",
            })?
            .allocation_checkpoint()?;
        let previous_layer_output = match previous_layer_output {
            Some(previous_layer_output) => Some(previous_layer_output.retain()?),
            None => None,
        };
        self.redo_points.push(SpanRedoPoint {
            span_index,
            span_input,
            layer_state,
            previous_layer_output,
        });
        Ok(())
    }

    /// The newest span's submission waited for the span before it, which
    /// therefore finished: only the newest is still unconfirmed.
    pub(super) fn confirm_all_but_newest(&mut self) {
        let confirmed_span_count = self.redo_points.len().saturating_sub(1);
        self.redo_points.drain(..confirmed_span_count);
    }

    /// The layer's end waited for its last span: nothing is unconfirmed.
    pub(super) fn confirm_all(&mut self) {
        self.redo_points.clear();
    }

    /// Puts layer `layer_index` back to where its oldest unconfirmed span
    /// began and returns that span's index, or `None` when no span is
    /// unconfirmed and there is nothing to run again.
    ///
    /// The newest point is undone first, so the oldest one's layer state
    /// and awaited output are what remain.
    pub(super) fn rewind_to_oldest(
        &mut self,
        layer_index: usize,
        span_states: &mut [SpanState],
        request_decoder_state: &mut RequestDecoderStateStack,
        previous_layer_output: &mut Option<MlxArray>,
    ) -> Result<Option<usize>, Qwen3_5ExecutionError> {
        let mut oldest_unconfirmed_span_index = None;
        while let Some(redo_point) = self.redo_points.pop() {
            let span_state = span_states.get_mut(redo_point.span_index).ok_or(
                Qwen3_5ExecutionError::InvalidInput {
                    description: "a span redo point names a span the prompt does not have",
                },
            )?;
            span_state.hidden_states = Some(redo_point.span_input);
            if let Some(collector) = span_state.collector.as_mut() {
                collector.forget_linear_attention_layer(layer_index);
            }
            if let Some(end_state_snapshot) = span_state.end_state_snapshot.as_mut() {
                forget_layer_end_state(end_state_snapshot, layer_index);
            }
            request_decoder_state
                .layer_mut(layer_index)
                .ok_or(Qwen3_5ExecutionError::InvalidRequestDecoderState {
                    layer_index,
                    description: "decoder layer entry is missing",
                })?
                .restore_allocation_checkpoint(redo_point.layer_state)?;
            *previous_layer_output = redo_point.previous_layer_output;
            oldest_unconfirmed_span_index = Some(redo_point.span_index);
        }
        Ok(oldest_unconfirmed_span_index)
    }
}
