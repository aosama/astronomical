use mlx_bridge::MlxArray;

use crate::PerformanceAttribution;
use crate::qwen3_5_streaming::PagedRouteValidationOutcome;

use super::{Qwen3_5ExecutionError, Qwen3_5Model, RequestDecoderStateStack};
use crate::qwen3_5_core::decoder::Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector;

impl Qwen3_5Model {
    /// Returns the highest-logit token ID for one final-position output.
    pub fn highest_logit_token_id(
        &self,
        final_position_logits: &MlxArray,
    ) -> Result<u32, Qwen3_5ExecutionError> {
        let selected_token = self.select_highest_logit_token(final_position_logits)?;
        Ok(selected_token.item_u32()?)
    }

    /// Materializes final logits and mutable decoder state for one forward pass.
    pub(crate) fn evaluate_forward_state(
        &self,
        final_logits: &MlxArray,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.evaluate_forward_state_with_performance_attribution(
            final_logits,
            request_decoder_state,
            &mut disabled_performance_attribution,
        )?;
        Ok(())
    }

    /// Evaluates forward roots and resolves deferred GPU missing-route bitmaps.
    pub(crate) fn evaluate_forward_state_with_performance_attribution(
        &self,
        final_logits: &MlxArray,
        request_decoder_state: &RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PagedRouteValidationOutcome, Qwen3_5ExecutionError> {
        let evaluation_arrays =
            super::forward_contract::forward_state_arrays(final_logits, request_decoder_state)?;
        self.evaluate_arrays_resolving_paged_routes(&evaluation_arrays, performance_attribution)
    }

    /// Evaluates the decoder-state roots of a prompt forward. A prompt is
    /// always read the exact way, so every route it took was resolved before
    /// its layer was built and nothing can be found missing here.
    pub(crate) fn evaluate_decoder_state_for_paged_route_resolution(
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
        self.evaluate_arrays_resolving_paged_routes(&evaluation_arrays, performance_attribution)
            .and_then(require_every_route_resolved)
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

/// A prompt forward cannot be run again from here, so a miss is an error.
fn require_every_route_resolved(
    validation_outcome: PagedRouteValidationOutcome,
) -> Result<(), Qwen3_5ExecutionError> {
    match validation_outcome {
        PagedRouteValidationOutcome::CompleteHit => Ok(()),
        PagedRouteValidationOutcome::MissedExperts => Err(Qwen3_5ExecutionError::InvalidInput {
            description: "a prompt forward reported experts missing from memory",
        }),
    }
}
