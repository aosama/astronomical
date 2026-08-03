use astronomical_runtime_integration::MlxArray;

use crate::{PerformanceAttribution, PerformanceOperation};

use super::{Qwen3_5ExecutionError, Qwen3_5Model, RequestDecoderStateStack};

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
        let evaluation_arrays =
            super::forward_contract::decoder_state_arrays(request_decoder_state)?;
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateEvaluationSynchronizationWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&evaluation_arrays),
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
