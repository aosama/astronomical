//! Draft attention capture for prompts containing projected image rows.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{PerformanceAttribution, Qwen3_5TargetForwardOutput};

use super::{
    Qwen3_5AttentionCapture, Qwen3_5ExecutionError, Qwen3_5Model, RequestDecoderStateStack,
};

impl Qwen3_5Model {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_visual_chunk_with_speculative_prefill_attention_capture_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        image_pad_token_id: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        attention_capture: &mut Qwen3_5AttentionCapture,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5TargetForwardOutput, usize), Qwen3_5ExecutionError> {
        let token_count = super::forward_contract::validate_forward_input(
            token_ids,
            starting_position_tokens,
            None,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.vocabulary_size(),
            self.config.maximum_position_count(),
        )?;
        let signed_token_ids = token_ids
            .iter()
            .map(|token_id| {
                i32::try_from(*token_id).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                    description: "token ID exceeds the MLX int32 range",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let token_indices = self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_count])?;
        let text_embeddings = self.embedding_lookup(&token_indices)?;
        let (injected_embeddings, consumed_visual_embedding_count) =
            super::visual_embedding_injection::qwen3_5_inject_visual_embeddings(
                &self.runtime,
                &text_embeddings,
                token_ids,
                visual_embeddings,
                starting_visual_embedding_index,
                image_pad_token_id,
            )?;
        let target_forward_output = self
            .build_target_forward_graph_from_embeddings_with_attention_capture(
                injected_embeddings,
                token_count,
                starting_position_tokens,
                None,
                request_decoder_state,
                Some(attention_capture),
                None,
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                performance_attribution,
                false,
                true,
            )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)?;
        Ok((target_forward_output, consumed_visual_embedding_count))
    }
}
