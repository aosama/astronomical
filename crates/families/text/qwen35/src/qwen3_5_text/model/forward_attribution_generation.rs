//! Generated-token and graph-building attributed forwards for Qwen3.5.
//!
//! These `Qwen3_5Model` methods construct one-token decode graphs and the
//! pre-final-normalization hidden-state variants; they are split from the
//! prefill-side attributed forwards in `forward_attribution.rs` to keep both
//! files inside the source-size budget.

use crate::PerformanceAttribution;
use crate::qwen3_5_core::model::target_forward_graph::Qwen3_5TargetForwardGraph;
use crate::qwen3_5_text::Qwen3_5MoEPagedExecutionMode;

use super::Qwen3_5Model;
use super::forward_contract;
use super::{RequestDecoderStateStack, TextExecutionError};
use mlx_bridge::{MlxArray, MlxDtype};

impl Qwen3_5Model {
    pub fn build_forward_chunk_with_performance_attribution(
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
        self.build_forward_graph(
            &token_indices,
            token_ids.len() as i32,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            performance_attribution,
        )
    }

    pub fn build_generated_token_forward_with_performance_attribution(
        &self,
        generated_token: &MlxArray,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        forward_contract::validate_generated_token_forward(
            generated_token,
            starting_position_tokens,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.maximum_position_count(),
        )?;
        let token_indices = self.runtime.astype(generated_token, MlxDtype::Int32)?;
        // The token becomes visible to the request only once its routes are
        // known to have been served from memory or read from storage.
        self.forward_resolving_expert_routes(
            &token_indices,
            1,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
        )
    }
}
