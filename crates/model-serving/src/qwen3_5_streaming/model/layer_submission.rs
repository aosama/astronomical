//! When a forward submits completed decoder layers to the GPU.
//!
//! Multi-token prefill may submit completed layer groups at a configured
//! interval so macOS can schedule other graphics work between command buffers.
//! A dense model's layers submit one at a time instead, so a forward holds no
//! more than the dense layer plan reserved (issue #1985).

use mlx_bridge::MlxArray;

use super::Qwen3_5Model;
use super::forward_layer_weights::ForwardLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::{PerformanceAttribution, PerformanceOperation};

/// Where a forward stands for the configured layer-group submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LayerGroupSubmission {
    pub(crate) completed_layer_count: usize,
    pub(crate) decoder_layer_count: usize,
    pub(crate) graph_submission_layer_interval: usize,
    pub(crate) forward_token_count: i32,
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

/// What one completed decoder layer leaves to evaluate: its output, and for
/// a linear-attention layer the convolution state and any prompt-cache
/// boundary snapshots, which the output does not reach.
pub(crate) struct CompletedLayerRoots<'layer> {
    pub(crate) layer_output: &'layer MlxArray,
    pub(crate) convolution_state: Option<&'layer MlxArray>,
    pub(crate) boundary_checkpoint_states: Vec<&'layer MlxArray>,
}

/// How one completed decoder layer reaches the GPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LayerSubmission {
    /// A sparse model's layer, or a kept dense layer in a one-token forward:
    /// only a configured layer group submits it.
    ConfiguredGroup,
    /// A dense layer in a multi-token forward: its working memory is the
    /// plan's largest term, so the previous layer finishes before it starts,
    /// and a streamed layer's read starts before that wait so it still
    /// overlaps the previous layer's compute.
    DensePromptLayer,
    /// A streamed dense layer in a one-token forward: its activations are
    /// tiny, so it starts at once and the forward waits for the streamed
    /// layer before it, which keeps two read layers in flight.
    StreamedDecodeLayer,
}

impl LayerSubmission {
    pub(crate) const fn for_layer(
        is_dense_layer: bool,
        is_streamed_layer: bool,
        forward_token_count: i32,
    ) -> Self {
        match (is_dense_layer, is_streamed_layer, forward_token_count > 1) {
            (true, _, true) => Self::DensePromptLayer,
            (true, true, false) => Self::StreamedDecodeLayer,
            _ => Self::ConfiguredGroup,
        }
    }
}

impl Qwen3_5Model {
    /// Submits work after a completed decoder layer, as [`LayerSubmission`]
    /// decides for this layer and forward.
    pub(crate) fn submit_completed_layer(
        &self,
        forward_layer_weights: &ForwardLayerWeights<'_>,
        completed_layer: CompletedLayerRoots<'_>,
        previous_dense_layer_output: &mut Option<MlxArray>,
        layer_group_submission: LayerGroupSubmission,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let hidden_states = completed_layer.layer_output;
        match forward_layer_weights.submission(layer_group_submission) {
            LayerSubmission::DensePromptLayer => self.submit_dense_prompt_layer(
                forward_layer_weights,
                completed_layer,
                previous_dense_layer_output,
                performance_attribution,
            ),
            LayerSubmission::StreamedDecodeLayer => self.submit_streamed_decode_layer(
                hidden_states,
                previous_dense_layer_output,
                performance_attribution,
            ),
            LayerSubmission::ConfiguredGroup if layer_group_submission.is_due() => {
                performance_attribution.measure_operation(
                    PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                    |_performance_attribution| self.runtime.async_eval_arrays(&[hidden_states]),
                )?;
                Ok(())
            }
            LayerSubmission::ConfiguredGroup => Ok(()),
        }
    }

    /// Starts a streamed layer's read, waits for the previous dense layer,
    /// then submits this one with its convolution state.
    ///
    /// Submitting before the wait let two layers' activations coexist, and
    /// with no configured groups every kept layer joined one graph: a fresh
    /// 1,287-token prompt at a 3 GB limit overran the limit with seven kept
    /// layers although one layer at the planned chunk measured 683 MB. The
    /// read stays ahead of the wait because the loads run on MLX's CPU
    /// stream, so the two layer slots still overlap reading with compute.
    ///
    /// The convolution state is a lazy copy of the input's last tokens that
    /// the layer output does not reach. Left for the forward's terminal
    /// evaluation, each linear-attention layer's copy pinned that layer's
    /// whole convolution input: measured, active memory climbed about 20 MB
    /// per such layer at 1,246 tokens and the forward peaked 2.2 GB above
    /// its weights at 4,096 tokens, against 683 MB for one layer.
    fn submit_dense_prompt_layer(
        &self,
        forward_layer_weights: &ForwardLayerWeights<'_>,
        completed_layer: CompletedLayerRoots<'_>,
        previous_dense_layer_output: &mut Option<MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let layer_output = completed_layer.layer_output;
        if forward_layer_weights.is_streamed() {
            let mut streamed_weight_arrays = Vec::new();
            forward_layer_weights
                .decoder_layer_weights()
                .append_array_references(&mut streamed_weight_arrays);
            self.runtime.async_eval_arrays(&streamed_weight_arrays)?;
        }
        self.wait_for_previous_dense_layer(previous_dense_layer_output, performance_attribution)?;
        let mut layer_roots = vec![layer_output];
        layer_roots.extend(completed_layer.convolution_state);
        layer_roots.extend(completed_layer.boundary_checkpoint_states);
        self.runtime.async_eval_arrays(&layer_roots)?;
        *previous_dense_layer_output = Some(layer_output.retain()?);
        Ok(())
    }

    /// Submits a streamed layer's output, then waits for the streamed layer
    /// before it.
    ///
    /// Building the graph runs ahead of the GPU, so without the wait every
    /// streamed layer would be read and held before the first one finished.
    /// Unlike a configured group, the final layer submits too; the caller's
    /// terminal evaluation then extends work that is already running.
    fn submit_streamed_decode_layer(
        &self,
        streamed_layer_output: &MlxArray,
        previous_dense_layer_output: &mut Option<MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.runtime.async_eval_arrays(&[streamed_layer_output])?;
        let mut previous_layer_output =
            previous_dense_layer_output.replace(streamed_layer_output.retain()?);
        self.wait_for_previous_dense_layer(&mut previous_layer_output, performance_attribution)
    }

    fn wait_for_previous_dense_layer(
        &self,
        previous_dense_layer_output: &mut Option<MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let Some(previous_layer_output) = previous_dense_layer_output.take() else {
            return Ok(());
        };
        performance_attribution.measure_operation(
            PerformanceOperation::StreamedDenseLayerCompletionWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&[&previous_layer_output]),
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "layer_submission_tests.rs"]
mod tests;
