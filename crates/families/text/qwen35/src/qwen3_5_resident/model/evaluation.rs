use mlx_bridge::MlxArray;

use crate::{PerformanceAttribution, PerformanceOperation};

use super::{Qwen3_5ExecutionError, Qwen3_5ResidentModel, RequestDecoderStateStack};
use crate::qwen3_5_core::decoder::Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector;

impl Qwen3_5ResidentModel {
    crate::qwen3_5_core::model::shared_model_methods::qwen3_5_shared_model_evaluation_methods!();

    /// Evaluates forward roots and resolves deferred GPU missing-route bitmaps.
    pub fn evaluate_forward_state_with_performance_attribution(
        &self,
        final_logits: &MlxArray,
        request_decoder_state: &RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let evaluation_arrays =
            super::forward_contract::forward_state_arrays(final_logits, request_decoder_state)?;
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&evaluation_arrays),
        )?;
        Ok(())
    }

    /// Evaluates decoder-state roots and returns paged-route replay outcome.
    pub fn evaluate_decoder_state_with_performance_attribution(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let mut evaluation_arrays =
            super::forward_contract::decoder_state_arrays(request_decoder_state)?;
        if let Some(boundary_checkpoint_collector) = boundary_checkpoint_collector {
            evaluation_arrays.extend(boundary_checkpoint_collector.evaluation_arrays());
        }
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&evaluation_arrays),
        )?;
        Ok(())
    }

    /// Evaluates terminal prefill roots: logits, decoder state, and checkpoint arrays.
    ///
    /// The terminal chunk needs vocabulary logits for first-token seeding while also
    /// publishing astro SSD prompt cache boundary checkpoints. Evaluating all roots in one pass
    /// keeps the terminal chunk as a single forward instead of a prefix+tail split.
    pub fn evaluate_terminal_prefill_state_with_performance_attribution(
        &self,
        final_logits: &MlxArray,
        request_decoder_state: &RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let mut evaluation_arrays =
            super::forward_contract::decoder_state_arrays(request_decoder_state)?;
        evaluation_arrays.push(final_logits);
        if let Some(boundary_checkpoint_collector) = boundary_checkpoint_collector {
            evaluation_arrays.extend(boundary_checkpoint_collector.evaluation_arrays());
        }
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&evaluation_arrays),
        )?;
        Ok(())
    }
}
