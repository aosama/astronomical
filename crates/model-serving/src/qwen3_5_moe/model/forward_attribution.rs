//! Request-owned performance-attribution forwarding for Qwen3.5-MoE MLX graphs.

use astronomical_runtime_integration::{MlxArray, MlxDtype};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::forward_contract::{validate_forward_input, validate_generated_token_forward};
use super::model::Qwen3_5MoEModel;
use super::visual_embedding_injection::qwen3_5_moe_inject_visual_embeddings;
use super::{
    Qwen3_5MoEExecutionError, Qwen3_5MoEPagedPrefillExecutionMode, Qwen3_5MoETargetForwardOutput,
    RequestDecoderStateStack,
};

impl Qwen3_5MoEModel {
    // Visual-prefill inputs stay explicit rather than introducing a parameter facade.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::qwen3_5_moe) fn prefill_chunck_with_visual_embeddings_and_performance_attribution(
        &self,
        chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        image_pad_token_id: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<usize, Qwen3_5MoEExecutionError> {
        let token_count = validate_forward_input(
            chunk_token_ids,
            starting_position_tokens,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.vocabulary_size(),
            self.config.maximum_position_count(),
        )?;
        let signed_token_ids = chunk_token_ids
            .iter()
            .map(|token_id| {
                i32::try_from(*token_id).map_err(|_| Qwen3_5MoEExecutionError::InvalidInput {
                    description: "token ID exceeds the MLX int32 range",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let token_indices = self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_count])?;
        let text_embeddings = self.embedding_lookup(&token_indices)?;
        let (injected_embeddings, consumed_visual_embedding_count) =
            qwen3_5_moe_inject_visual_embeddings(
                &self.runtime,
                &text_embeddings,
                chunk_token_ids,
                visual_embeddings,
                starting_visual_embedding_index,
                image_pad_token_id,
            )?;
        drop(self.build_forward_graph_from_embeddings(
            injected_embeddings,
            token_count,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )?);
        self.evaluate_decoder_state_with_performance_attribution(
            request_decoder_state,
            performance_attribution,
        )?;
        Ok(consumed_visual_embedding_count)
    }

    pub(in crate::qwen3_5_moe) fn prefill_chunck_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5MoEExecutionError> {
        drop(self.build_forward_chunk_with_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
        )?);
        self.evaluate_decoder_state_with_performance_attribution(
            request_decoder_state,
            performance_attribution,
        )
    }

    pub(in crate::qwen3_5_moe) fn forward_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let final_logits = self.build_forward_chunk_with_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
        )?;
        self.evaluate_forward_state(&final_logits, request_decoder_state)?;
        Ok(final_logits)
    }

    pub(in crate::qwen3_5_moe) fn forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5MoETargetForwardOutput, Qwen3_5MoEExecutionError> {
        self.forward_chunk_with_pre_final_normalization_hidden_states_and_synchronization_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
            None,
        )
    }

    pub(in crate::qwen3_5_moe) fn replay_rejected_mtp_draft_with_performance_attribution(
        &self,
        current_generated_token_id: u32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5MoETargetForwardOutput, Qwen3_5MoEExecutionError> {
        self.forward_chunk_with_pre_final_normalization_hidden_states_and_synchronization_attribution(
            &[current_generated_token_id],
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
            Some(PerformanceOperation::MtpRejectedDraftReplaySynchronizationWait),
        )
    }

    fn forward_chunk_with_pre_final_normalization_hidden_states_and_synchronization_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
        synchronization_operation: Option<PerformanceOperation>,
    ) -> Result<Qwen3_5MoETargetForwardOutput, Qwen3_5MoEExecutionError> {
        let target_forward_output = self.build_target_forward_graph(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )?;
        let synchronize_target_forward_output =
            |_performance_attribution: &mut PerformanceAttribution| -> Result<
                (),
                Qwen3_5MoEExecutionError,
            > {
                self.evaluate_forward_state(
                    target_forward_output.final_logits(),
                    request_decoder_state,
                )?;
                self.runtime.evaluate_arrays(&[
                    target_forward_output.pre_final_normalization_hidden_states(),
                ])?;
                Ok(())
            };
        match synchronization_operation {
            Some(synchronization_operation) => performance_attribution
                .measure_operation(synchronization_operation, synchronize_target_forward_output)?,
            None => synchronize_target_forward_output(performance_attribution)?,
        }
        Ok(target_forward_output)
    }

    pub(in crate::qwen3_5_moe) fn forward_chunk_with_all_position_logits_and_pre_final_normalization_hidden_states_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5MoETargetForwardOutput, Vec<u32>), Qwen3_5MoEExecutionError> {
        let token_count = validate_forward_input(
            token_ids,
            starting_position_tokens,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.vocabulary_size(),
            self.config.maximum_position_count(),
        )?;
        let signed_token_ids = token_ids
            .iter()
            .map(|token_id| {
                i32::try_from(*token_id).map_err(|_| Qwen3_5MoEExecutionError::InvalidInput {
                    description: "token ID exceeds the MLX int32 range",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let token_indices = self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_count])?;
        let target_forward_output = self.build_target_forward_graph_from_token_indices(
            &token_indices,
            token_count,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDecodeVerification,
            performance_attribution,
            true,
        )?;
        let all_position_logits = target_forward_output.all_position_logits().ok_or(
            Qwen3_5MoEExecutionError::InvalidInput {
                description: "target verification forward did not retain all-position logits",
            },
        )?;
        let target_verify_token_indices = self.build_greedy_token(all_position_logits)?;
        let target_verify_token_ids = performance_attribution.measure_operation(
            PerformanceOperation::MtpTargetVerificationSynchronizationWait,
            |_performance_attribution| -> Result<Vec<u32>, Qwen3_5MoEExecutionError> {
                self.evaluate_forward_state(&target_verify_token_indices, request_decoder_state)?;
                self.runtime.evaluate_arrays(&[
                    target_forward_output.pre_final_normalization_hidden_states()
                ])?;
                Ok(target_verify_token_indices.to_vec_u32()?)
            },
        )?;
        Ok((target_forward_output, target_verify_token_ids))
    }

    pub(in crate::qwen3_5_moe) fn build_forward_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        self.build_forward_chunk_with_paged_prefill_execution_mode_and_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )
    }

    pub(in crate::qwen3_5_moe) fn build_generated_token_forward_with_performance_attribution(
        &self,
        generated_token: &MlxArray,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        validate_generated_token_forward(
            generated_token,
            starting_position_tokens,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.maximum_position_count(),
        )?;
        let token_indices = self.runtime.astype(generated_token, MlxDtype::Int32)?;
        self.build_forward_graph(
            &token_indices,
            1,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )
    }

    pub(in crate::qwen3_5_moe) fn generated_token_forward_with_pre_final_normalization_hidden_states_and_performance_attribution(
        &self,
        generated_token: &MlxArray,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5MoETargetForwardOutput, Qwen3_5MoEExecutionError> {
        validate_generated_token_forward(
            generated_token,
            starting_position_tokens,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.maximum_position_count(),
        )?;
        let token_indices = self.runtime.astype(generated_token, MlxDtype::Int32)?;
        let target_forward_output = self.build_target_forward_graph_from_token_indices(
            &token_indices,
            1,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
            false,
        )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)?;
        self.runtime
            .evaluate_arrays(&[target_forward_output.pre_final_normalization_hidden_states()])?;
        Ok(target_forward_output)
    }
}
