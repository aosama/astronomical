use astronomical_runtime_integration::{MlxArray, MlxDtype};

use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{PerformanceAttribution, Qwen3_5ExecutionError};

use super::{Qwen3_5Model, RequestDecoderStateStack};

impl Qwen3_5Model {
    pub(crate) fn prefill_chunck_with_speculative_prefill_gpu_token_indices_and_position_offsets_and_performance_attribution(
        &self,
        selected_token_indices_on_gpu: &MlxArray,
        selected_token_count: i32,
        starting_position_tokens: u32,
        selected_prompt_position_offsets: &MlxArray,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        if selected_token_count <= 0 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill GPU token indices must not be empty",
            });
        }
        if selected_token_indices_on_gpu.shape() != [1, selected_token_count]
            || selected_token_indices_on_gpu.dtype() != MlxDtype::Int32
        {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill GPU token indices must be an int32 row",
            });
        }
        if selected_prompt_position_offsets.shape() != [selected_token_count]
            || selected_prompt_position_offsets.dtype() != MlxDtype::Int32
        {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill position offsets must match GPU token indices",
            });
        }
        if request_decoder_state.layer_count() != self.config.layer_count() as usize {
            return Err(Qwen3_5ExecutionError::DecoderLayerCountMismatch {
                actual_decoder_layer_count: request_decoder_state.layer_count(),
                expected_decoder_layer_count: self.config.layer_count() as usize,
            });
        }

        let target_forward_output = self
            .build_target_forward_graph_from_token_indices_with_attention_capture(
                selected_token_indices_on_gpu,
                selected_token_count,
                starting_position_tokens,
                Some(selected_prompt_position_offsets),
                request_decoder_state,
                None,
                None,
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                performance_attribution,
                false,
                true,
            )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_chunck_with_speculative_prefill_gpu_token_indices_and_visual_embeddings_and_position_offsets_and_performance_attribution(
        &self,
        selected_token_indices_on_gpu: &MlxArray,
        selected_prompt_token_ids: &[u32],
        starting_position_tokens: u32,
        selected_prompt_position_offsets: &MlxArray,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        image_pad_token_id: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<usize, Qwen3_5ExecutionError> {
        let selected_token_count = i32::try_from(selected_prompt_token_ids.len()).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill selected visual token count exceeds the MLX range",
            }
        })?;
        if selected_token_count <= 0
            || selected_token_indices_on_gpu.shape() != [1, selected_token_count]
            || selected_token_indices_on_gpu.dtype() != MlxDtype::Int32
            || selected_prompt_position_offsets.shape() != [selected_token_count]
            || selected_prompt_position_offsets.dtype() != MlxDtype::Int32
        {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill sparse visual inputs have incompatible layouts",
            });
        }
        let text_embeddings = self.embedding_lookup(selected_token_indices_on_gpu)?;
        let (injected_embeddings, consumed_visual_embedding_count) =
            super::visual_embedding_injection::qwen3_5_inject_visual_embeddings(
                &self.runtime,
                &text_embeddings,
                selected_prompt_token_ids,
                visual_embeddings,
                starting_visual_embedding_index,
                image_pad_token_id,
            )?;
        let target_forward_output = self
            .build_target_forward_graph_from_embeddings_with_attention_capture(
                injected_embeddings,
                selected_token_count,
                starting_position_tokens,
                Some(selected_prompt_position_offsets),
                request_decoder_state,
                None,
                None,
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                performance_attribution,
                false,
                true,
            )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)?;
        Ok(consumed_visual_embedding_count)
    }
}
