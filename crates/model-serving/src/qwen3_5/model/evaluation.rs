use astronomical_runtime_integration::MlxArray;

use crate::{PerformanceAttribution, PerformanceOperation};

use super::{Qwen3_5ExecutionError, Qwen3_5Model, RequestDecoderStateStack};
use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;

impl Qwen3_5Model {
    /// Returns the highest-logit token ID for one final-position output.
    pub fn greedy_token_id(
        &self,
        final_position_logits: &MlxArray,
    ) -> Result<u32, Qwen3_5ExecutionError> {
        let greedy_token = self.build_greedy_token(final_position_logits)?;
        Ok(greedy_token.item_u32()?)
    }

    /// Returns highest-logit token IDs for each row in a logits tensor.
    pub fn greedy_token_ids(
        &self,
        position_logits: &MlxArray,
    ) -> Result<Vec<u32>, Qwen3_5ExecutionError> {
        let greedy_tokens = self.build_greedy_token(position_logits)?;
        Ok(greedy_tokens.to_vec_u32()?)
    }

    pub(crate) fn build_greedy_token(
        &self,
        final_position_logits: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        Ok(self.runtime.argmax_axis(final_position_logits, -1)?)
    }

    /// Materializes final logits and mutable decoder state for one forward pass.
    pub(crate) fn evaluate_forward_state(
        &self,
        final_logits: &MlxArray,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let evaluation_arrays =
            super::forward_contract::forward_state_arrays(final_logits, request_decoder_state)?;
        self.runtime.evaluate_arrays(&evaluation_arrays)?;
        Ok(())
    }

    /// Synchronizes reusable decoder state after an intermediate prefill chunk.
    pub(crate) fn evaluate_decoder_state_with_performance_attribution(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.evaluate_decoder_state_and_optional_boundary_checkpoints_with_performance_attribution(
            request_decoder_state,
            None,
            performance_attribution,
        )
    }

    pub(crate) fn evaluate_decoder_state_and_boundary_checkpoints_with_performance_attribution(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        boundary_checkpoint_collector: &Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.evaluate_decoder_state_and_optional_boundary_checkpoints_with_performance_attribution(
            request_decoder_state,
            Some(boundary_checkpoint_collector),
            performance_attribution,
        )
    }

    fn evaluate_decoder_state_and_optional_boundary_checkpoints_with_performance_attribution(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let mut evaluation_arrays =
            super::forward_contract::decoder_state_arrays(request_decoder_state)?;
        if let Some(boundary_checkpoint_collector) = boundary_checkpoint_collector {
            evaluation_arrays.extend(boundary_checkpoint_collector.evaluation_arrays());
        }
        // mlx_async_eval is asynchronous only with respect to the submitted
        // graphics-processor work. Before returning, MLX still traverses the
        // lazy graph, resolves dependencies, compiles first-use Metal libraries
        // and pipelines, and encodes commands. Keep that host-side submission
        // boundary separate from the remaining stream completion wait so cold
        // compilation cannot be misreported as graphics-processor execution.
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
            |_performance_attribution| self.runtime.async_eval_arrays(&evaluation_arrays),
        )?;
        // Prefill state and optional cache boundary checkpoints must be fully
        // materialized before allocator cleanup or publication can observe
        // their buffers. This explicit barrier measures only work still in
        // flight after async evaluation submission returned.
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| self.runtime.synchronize_gpu_stream(),
        )?;
        Ok(())
    }

    /// Submits decode evaluation without synchronizing the graphics processor.
    pub(crate) fn async_evaluate_generation(
        &self,
        generated_token: &MlxArray,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let evaluation_arrays =
            super::forward_contract::forward_state_arrays(generated_token, request_decoder_state)?;
        self.runtime.async_eval_arrays(&evaluation_arrays)?;
        Ok(())
    }
}
