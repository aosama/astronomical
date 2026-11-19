//! A layer-at-a-time prefill between its first layer and its last: what it
//! carries from one advance to the next, and how a failed layer is put back
//! so only its failed span runs again (issue #2221).

use astronomical_runtime_integration::MlxRuntimeError;
use mlx_bridge::MlxArray;

use super::layer_major_prefill::{LayerMajorCheckpointContract, LayerMajorSpan};
use super::layer_major_span_redo::UnconfirmedSpans;
use super::layer_major_span_state::SpanState;
use crate::RequestDecoderStateStack;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

/// A layer-major prefill between its first layer and its last. It advances
/// one layer per call (`advance_layer_major_prefill`).
pub struct LayerMajorPrefillRun {
    pub(super) spans: Vec<LayerMajorSpan>,
    pub(super) checkpoint_contract: Option<LayerMajorCheckpointContract>,
    pub(super) should_materialize_logits: bool,
    pub(super) starting_position_tokens: u32,
    pub(super) span_states: Vec<SpanState>,
    /// Image embedding rows the spans consumed when they were embedded.
    pub(super) consumed_visual_embedding_count: usize,
    pub(super) previous_layer_output: Option<MlxArray>,
    pub(super) completed_layer_count: usize,
    /// The span the layer in flight runs next: zero, except after a failed
    /// span put the layer back to where that span began.
    pub(super) next_span_index: usize,
    /// The spans of the layer in flight that MLX has not confirmed. Only an
    /// expert model fills this; it waits for every layer anyway.
    pub(super) unconfirmed_spans: UnconfirmedSpans,
    /// Whether the last failure left the layer in flight ready to run its
    /// failed span again.
    pub(super) is_rewound_to_failed_span: bool,
    /// A span a test fails once on purpose, as (layer, span).
    pub(super) span_to_fail_once: Option<(usize, usize)>,
}

impl LayerMajorPrefillRun {
    /// Layers that have run over every span so far.
    pub const fn completed_layer_count(&self) -> usize {
        self.completed_layer_count
    }

    /// Whether the failure the last advance returned left this prefill
    /// ready to run the failed span again, with everything before it
    /// intact.
    pub const fn can_run_failed_span_again(&self) -> bool {
        self.is_rewound_to_failed_span
    }

    /// Has the span at (`layer_index`, `span_index`) fail once after it is
    /// submitted, the way a span MLX could not fit is reported.
    pub const fn fail_span_once_for_tests(&mut self, layer_index: usize, span_index: usize) {
        self.span_to_fail_once = Some((layer_index, span_index));
    }

    pub(super) fn fail_marked_span_once(
        &mut self,
        layer_index: usize,
        span_index: usize,
    ) -> Result<(), Qwen3_5ExecutionError> {
        if self.span_to_fail_once != Some((layer_index, span_index)) {
            return Ok(());
        }
        self.span_to_fail_once = None;
        // Worded as MLX words a command buffer that ran out of memory, so
        // the engine classifies it as it classifies the real thing.
        Err(Qwen3_5ExecutionError::Runtime(
            MlxRuntimeError::RuntimeOperation {
                operation: "run a prompt span failed on purpose by a test",
                description: "[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".to_owned(),
            },
        ))
    }
}

/// After a failed advance: puts the layer in flight back to where its
/// oldest unconfirmed span began, when there is one. A prefill that cannot
/// be put back stays failed, and whoever drives it starts the prompt again.
pub(super) fn rewind_to_failed_span(
    layer_major_prefill_run: &mut LayerMajorPrefillRun,
    request_decoder_state: &mut RequestDecoderStateStack,
) {
    let rewind_outcome = layer_major_prefill_run.unconfirmed_spans.rewind_to_oldest(
        layer_major_prefill_run.completed_layer_count,
        &mut layer_major_prefill_run.span_states,
        request_decoder_state,
        &mut layer_major_prefill_run.previous_layer_output,
    );
    layer_major_prefill_run.is_rewound_to_failed_span = false;
    match rewind_outcome {
        Ok(Some(failed_span_index)) => {
            layer_major_prefill_run.next_span_index = failed_span_index;
            layer_major_prefill_run.is_rewound_to_failed_span = true;
        }
        Ok(None) => {}
        Err(rewind_error) => {
            tracing::warn!(
                error = %rewind_error,
                "could not put a failed prompt layer back to its failed span"
            );
        }
    }
}
