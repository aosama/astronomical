//! The streaming model's own attributed forward. The prefill forwards both
//! models run are in `qwen3_5_core::model::prefill_forward`.

use crate::qwen3_5_core::model::prefill_forward::Qwen3_5PrefillForward;
use mlx_bridge::MlxArray;

use crate::PerformanceAttribution;
use crate::qwen3_5_text::Qwen3_5MoEPagedExecutionMode;

use super::Qwen3_5Model;
use super::{RequestDecoderStateStack, TextExecutionError};
use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

impl Qwen3_5Model {
    pub fn forward_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let signed_token_ids = token_ids
            .iter()
            .map(|token_id| {
                i32::try_from(*token_id).map_err(|_| TextExecutionError::InvalidInput {
                    description: "token ID exceeds the MLX int32 range",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let token_indices = self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_ids.len() as i32])?;
        self.forward_resolving_expert_routes(
            &token_indices,
            token_ids.len() as i32,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
        )
    }
}

impl Qwen3_5PrefillForward for Qwen3_5Model {
    const PRODUCTION_FORWARD_MODE: Qwen3_5MoEPagedExecutionMode =
        Qwen3_5MoEPagedExecutionMode::ProductionDefault;

    /// Evaluates the decoder state and resolves the expert routes the
    /// forward deferred.
    fn evaluate_prefill_decoder_state(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&AstroSsdPromptCacheBoundaryCheckpointCollector>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        self.evaluate_decoder_state_for_paged_route_resolution(
            request_decoder_state,
            boundary_checkpoint_collector,
            performance_attribution,
        )
    }
}
