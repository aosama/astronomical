use mlx_bridge::MlxArray;

use crate::PerformanceAttribution;
use crate::qwen3_5_streaming::PagedRouteValidationOutcome;

use super::{Qwen3_5Model, RequestDecoderStateStack, TextExecutionError};
use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

impl Qwen3_5Model {
    crate::qwen3_5_core::model::shared_model_methods::qwen3_5_shared_model_evaluation_methods!();

    /// Evaluates forward roots and resolves deferred GPU missing-route bitmaps.
    pub fn evaluate_forward_state_with_performance_attribution(
        &self,
        final_logits: &MlxArray,
        request_decoder_state: &RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PagedRouteValidationOutcome, TextExecutionError> {
        let evaluation_arrays =
            super::forward_contract::forward_state_arrays(final_logits, request_decoder_state)?;
        self.evaluate_arrays_resolving_paged_routes(&evaluation_arrays, performance_attribution)
    }

    /// Evaluates the decoder-state roots of a prompt forward. A prompt is
    /// always read the exact way, so every route it took was resolved before
    /// its layer was built and nothing can be found missing here.
    pub fn evaluate_decoder_state_for_paged_route_resolution(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&AstroSsdPromptCacheBoundaryCheckpointCollector>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        let mut evaluation_arrays =
            super::forward_contract::decoder_state_arrays(request_decoder_state)?;
        if let Some(boundary_checkpoint_collector) = boundary_checkpoint_collector {
            evaluation_arrays.extend(boundary_checkpoint_collector.evaluation_arrays());
        }
        self.evaluate_arrays_resolving_paged_routes(&evaluation_arrays, performance_attribution)
            .and_then(require_every_route_resolved)
    }
}

/// A prompt forward cannot be run again from here, so a miss is an error.
fn require_every_route_resolved(
    validation_outcome: PagedRouteValidationOutcome,
) -> Result<(), TextExecutionError> {
    match validation_outcome {
        PagedRouteValidationOutcome::CompleteHit => Ok(()),
        PagedRouteValidationOutcome::MissedExperts => Err(TextExecutionError::InvalidInput {
            description: "a prompt forward reported experts missing from memory",
        }),
    }
}
