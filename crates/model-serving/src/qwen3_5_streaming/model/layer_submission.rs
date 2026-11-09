//! When a forward submits completed decoder layers to the GPU.
//!
//! Multi-token prefill may submit completed layer groups at a configured
//! interval so macOS can schedule other graphics work between command buffers.
//! A streamed dense layer always submits on its own, and the forward then
//! waits for the streamed layer before it (issue #1985).

use mlx_c_rust::MlxArray;

use super::Qwen3_5Model;
use super::dense_layer_residency::ForwardLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::{PerformanceAttribution, PerformanceOperation};

/// Where a forward stands for the configured layer-group submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LayerGroupSubmission {
    pub(crate) completed_layer_count: usize,
    pub(crate) decoder_layer_count: usize,
    pub(crate) graph_submission_layer_interval: usize,
}

impl LayerGroupSubmission {
    /// Whether a configured layer group ends here, short of the final layer:
    /// final normalization and logits extend the last group's graph, and the
    /// caller owns that terminal evaluation.
    pub(crate) const fn is_due(self) -> bool {
        self.graph_submission_layer_interval > 0
            && self
                .completed_layer_count
                .is_multiple_of(self.graph_submission_layer_interval)
            && self.completed_layer_count < self.decoder_layer_count
    }
}

impl Qwen3_5Model {
    /// Submits work after a completed decoder layer: always for a streamed
    /// dense layer, otherwise when a configured layer group ends.
    pub(crate) fn submit_completed_layer(
        &self,
        forward_layer_weights: &ForwardLayerWeights<'_>,
        hidden_states: &MlxArray,
        previous_streamed_layer_output: &mut Option<MlxArray>,
        layer_group_submission: LayerGroupSubmission,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        if forward_layer_weights.is_streamed() {
            return self.submit_streamed_dense_layer(
                hidden_states,
                previous_streamed_layer_output,
                performance_attribution,
            );
        }
        if layer_group_submission.is_due() {
            performance_attribution.measure_operation(
                PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                |_performance_attribution| self.runtime.async_eval_arrays(&[hidden_states]),
            )?;
        }
        Ok(())
    }

    /// Submits a streamed dense layer's output, then waits for the streamed
    /// layer before it, prefill and decode alike.
    ///
    /// Building the graph runs ahead of the GPU, so without the wait every
    /// streamed layer would be read and held before the first one finished
    /// (seen live: a fresh prompt at a 3 GB limit hit the MLX limit). Waiting
    /// bounds the layers in flight to the two slots the plan reserved: the one
    /// computing and the one being read for it. Unlike a configured group, the
    /// final layer submits too; the caller's terminal evaluation then extends
    /// work that is already running, which costs nothing.
    fn submit_streamed_dense_layer(
        &self,
        streamed_layer_output: &MlxArray,
        previous_streamed_layer_output: &mut Option<MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.runtime.async_eval_arrays(&[streamed_layer_output])?;
        if let Some(previous_layer_output) =
            previous_streamed_layer_output.replace(streamed_layer_output.retain()?)
        {
            performance_attribution.measure_operation(
                PerformanceOperation::StreamedDenseLayerCompletionWait,
                |_performance_attribution| self.runtime.evaluate_arrays(&[&previous_layer_output]),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "layer_submission_tests.rs"]
mod tests;
