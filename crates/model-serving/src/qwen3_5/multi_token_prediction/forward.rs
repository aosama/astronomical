use astronomical_runtime_integration::{MlxArray, MlxDtype};

use crate::qwen3_5::multi_token_prediction::Qwen3_5MtpRequestState;
use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{PerformanceAttribution, PerformanceOperation};

use crate::qwen3_5::model::decoder_layer_weights::{
    Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights, Qwen3_5FullAttentionWeights,
};
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};

/// Evaluated Qwen MTP-head outputs for one or more draft positions.
pub struct Qwen3_5MtpForwardOutput {
    draft_logits: MlxArray,
    post_normalization_hidden_states: MlxArray,
}

impl Qwen3_5MtpForwardOutput {
    /// Returns float32 draft logits for every supplied MTP input position.
    #[must_use]
    pub fn draft_logits(&self) -> &MlxArray {
        &self.draft_logits
    }

    /// Returns MTP final-normalization rows used to chain a second draft token.
    #[must_use]
    pub fn post_normalization_hidden_states(&self) -> &MlxArray {
        &self.post_normalization_hidden_states
    }
}

impl Qwen3_5Model {
    /// Commits shifted prompt history to the request-local MTP attention state.
    pub fn prefill_mtp_history(
        &self,
        target_pre_final_normalization_hidden_states: &MlxArray,
        shifted_prompt_token_indices: &MlxArray,
        mtp_request_state: &mut Qwen3_5MtpRequestState,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.prefill_mtp_history_with_performance_attribution(
            target_pre_final_normalization_hidden_states,
            shifted_prompt_token_indices,
            mtp_request_state,
            &mut disabled_performance_attribution,
        )
    }

    pub(crate) fn prefill_mtp_history_with_performance_attribution(
        &self,
        target_pre_final_normalization_hidden_states: &MlxArray,
        shifted_prompt_token_indices: &MlxArray,
        mtp_request_state: &mut Qwen3_5MtpRequestState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let post_normalization_hidden_states = performance_attribution.measure_operation(
            PerformanceOperation::MtpHeadForwardGraphConstruction,
            |performance_attribution| {
                self.build_mtp_hidden_graph(
                    target_pre_final_normalization_hidden_states,
                    shifted_prompt_token_indices,
                    mtp_request_state,
                    performance_attribution,
                )
            },
        )?;
        self.evaluate_mtp_updated_state(
            mtp_request_state,
            &[&post_normalization_hidden_states],
            performance_attribution,
        )
    }

    pub(crate) fn prefill_mtp_history_from_token_ids_with_performance_attribution(
        &self,
        target_pre_final_normalization_hidden_states: &MlxArray,
        shifted_prompt_token_ids: &[u32],
        mtp_request_state: &mut Qwen3_5MtpRequestState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let shifted_prompt_token_count =
            i32::try_from(shifted_prompt_token_ids.len()).map_err(|_| {
                Qwen3_5ExecutionError::InvalidInput {
                    description: "MTP prompt-history token count exceeds the MLX int32 range",
                }
            })?;
        let shifted_prompt_token_indices = self
            .runtime
            .array_from_u32(shifted_prompt_token_ids, &[1, shifted_prompt_token_count])?;
        self.prefill_mtp_history_with_performance_attribution(
            target_pre_final_normalization_hidden_states,
            &shifted_prompt_token_indices,
            mtp_request_state,
            performance_attribution,
        )
    }

    /// Runs the resident Qwen MTP head and advances its request-local attention state.
    ///
    /// The first call fuses target pre-final-normalization hidden states with the
    /// sampled target token. A chained call instead supplies the prior MTP
    /// post-normalization hidden states. The MTP KV state begins at position zero
    /// and is intentionally independent of the target decoder cache length.
    pub fn forward_mtp_draft(
        &self,
        hidden_states_for_mtp_fusion: &MlxArray,
        next_token_indices: &MlxArray,
        mtp_request_state: &mut Qwen3_5MtpRequestState,
    ) -> Result<Qwen3_5MtpForwardOutput, Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.forward_mtp_draft_with_performance_attribution(
            hidden_states_for_mtp_fusion,
            next_token_indices,
            mtp_request_state,
            &mut disabled_performance_attribution,
        )
        .map(|(mtp_forward_output, _draft_token_id)| mtp_forward_output)
    }

    pub(crate) fn forward_mtp_draft_with_performance_attribution(
        &self,
        hidden_states_for_mtp_fusion: &MlxArray,
        next_token_indices: &MlxArray,
        mtp_request_state: &mut Qwen3_5MtpRequestState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5MtpForwardOutput, u32), Qwen3_5ExecutionError> {
        let mtp_forward_output = performance_attribution.measure_operation(
            PerformanceOperation::MtpHeadForwardGraphConstruction,
            |performance_attribution| {
                self.build_mtp_draft_graph(
                    hidden_states_for_mtp_fusion,
                    next_token_indices,
                    mtp_request_state,
                    performance_attribution,
                )
            },
        )?;
        let draft_token_indices = self.build_greedy_token(mtp_forward_output.draft_logits())?;
        self.evaluate_mtp_updated_state(
            mtp_request_state,
            &[
                &draft_token_indices,
                mtp_forward_output.post_normalization_hidden_states(),
            ],
            performance_attribution,
        )?;
        Ok((mtp_forward_output, draft_token_indices.item_u32()?))
    }

    fn evaluate_mtp_updated_state(
        &self,
        mtp_request_state: &Qwen3_5MtpRequestState,
        forward_evaluation_roots: &[&MlxArray],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let mtp_full_attention_state = mtp_request_state.full_attention_key_value_state();
        let attention_keys = mtp_full_attention_state.keys_state().ok_or_else(|| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "MTP attention forward did not populate key state",
            }
        })?;
        let attention_values = mtp_full_attention_state.values_state().ok_or_else(|| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "MTP attention forward did not populate value state",
            }
        })?;
        let mut state_evaluation_roots = Vec::with_capacity(forward_evaluation_roots.len() + 2);
        state_evaluation_roots.extend_from_slice(forward_evaluation_roots);
        state_evaluation_roots.extend_from_slice(&[attention_keys, attention_values]);
        Ok(performance_attribution.measure_operation(
            PerformanceOperation::MtpHeadStateEvaluationSynchronizationWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&state_evaluation_roots),
        )?)
    }

    fn build_mtp_draft_graph(
        &self,
        hidden_states_for_mtp_fusion: &MlxArray,
        next_token_indices: &MlxArray,
        mtp_request_state: &mut Qwen3_5MtpRequestState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5MtpForwardOutput, Qwen3_5ExecutionError> {
        let post_normalization_hidden_states = self.build_mtp_hidden_graph(
            hidden_states_for_mtp_fusion,
            next_token_indices,
            mtp_request_state,
            performance_attribution,
        )?;
        let draft_logits = self.quantized_linear(
            &post_normalization_hidden_states,
            &self.weights.language_model_head_weights,
        )?;
        Ok(Qwen3_5MtpForwardOutput {
            draft_logits: self.runtime.astype(&draft_logits, MlxDtype::Float32)?,
            post_normalization_hidden_states,
        })
    }

    fn build_mtp_hidden_graph(
        &self,
        hidden_states_for_mtp_fusion: &MlxArray,
        next_token_indices: &MlxArray,
        mtp_request_state: &mut Qwen3_5MtpRequestState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let token_count = validate_mtp_forward_inputs(
            hidden_states_for_mtp_fusion,
            next_token_indices,
            self.config.hidden_size() as i32,
        )?;
        let mtp_weights =
            self.mtp_weights
                .as_ref()
                .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
                    tensor_name: "resident Qwen MTP head".to_owned(),
                })?;
        let next_token_embeddings = self.embedding_lookup(next_token_indices)?;
        let normalized_token_embeddings = self.runtime.rms_norm(
            &next_token_embeddings,
            &mtp_weights.pre_fc_normalization_embedding_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let normalized_hidden_states = self.runtime.rms_norm(
            hidden_states_for_mtp_fusion,
            &mtp_weights.pre_fc_normalization_hidden_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let fused_mtp_inputs = self.runtime.concatenate_axis(
            &[&normalized_token_embeddings, &normalized_hidden_states],
            -1,
        )?;
        let fused_mtp_hidden_states =
            self.quantized_linear(&fused_mtp_inputs, &mtp_weights.fusion_projection)?;
        let normalized_mtp_input = self.runtime.rms_norm(
            &fused_mtp_hidden_states,
            &mtp_weights.decoder_layer_weights.input_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let full_attention_weights = match &mtp_weights.decoder_layer_weights.attention_weights {
            Qwen3_5AttentionWeights::Full(full_attention_weights) => full_attention_weights,
            Qwen3_5AttentionWeights::Linear(_) => {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "the Qwen MTP head must use full attention",
                });
            }
        };
        let attention_output = self.forward_mtp_full_attention(
            &normalized_mtp_input,
            token_count,
            full_attention_weights,
            mtp_request_state.full_attention_key_value_state_mut(),
        )?;
        let attention_residual = self
            .runtime
            .add(&fused_mtp_hidden_states, &attention_output)?;
        let normalized_attention = self.runtime.rms_norm(
            &attention_residual,
            &mtp_weights
                .decoder_layer_weights
                .post_attention_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let mlp_output = match &mtp_weights.decoder_layer_weights.mlp_weights {
            Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) => self
                .forward_qwen3_5_dense_mlp(
                    &normalized_attention,
                    dense_mlp_weights,
                    Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                )?,
            Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(mixture_of_experts_weights) => {
                let expert_pager = self.expert_pager.as_ref().ok_or_else(|| {
                    Qwen3_5ExecutionError::MissingTensor {
                        tensor_name: "sparse model expert pager".to_owned(),
                    }
                })?;
                // The MTP sparse layer is appended after every target decoder
                // layer in the shared pager. It participates in the same global
                // byte ceiling and recency policy without pretending to belong
                // to the language trunk's artifact namespace.
                self.forward_qwen3_5_moe_with_paging(
                    &normalized_attention,
                    mixture_of_experts_weights,
                    expert_pager,
                    self.config.layer_count() as usize,
                    token_count != 1,
                    Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                    performance_attribution,
                )?
            }
        };
        let decoder_output = self.runtime.add(&attention_residual, &mlp_output)?;
        let post_normalization_hidden_states = self.runtime.rms_norm(
            &decoder_output,
            &mtp_weights.final_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        Ok(post_normalization_hidden_states)
    }

    fn forward_mtp_full_attention(
        &self,
        normalized_mtp_input: &MlxArray,
        token_count: i32,
        full_attention_weights: &Qwen3_5FullAttentionWeights,
        mtp_full_attention_state: &mut crate::FullAttentionKeyValueState,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        self.forward_full_attention(
            normalized_mtp_input,
            token_count,
            mtp_full_attention_state.offset_tokens(),
            full_attention_weights,
            mtp_full_attention_state,
            0,
            None,
            None,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
        )
    }
}

fn validate_mtp_forward_inputs(
    hidden_states_for_mtp_fusion: &MlxArray,
    next_token_indices: &MlxArray,
    hidden_size: i32,
) -> Result<i32, Qwen3_5ExecutionError> {
    let hidden_state_shape = hidden_states_for_mtp_fusion.shape();
    let next_token_shape = next_token_indices.shape();
    let [batch_size, token_count, hidden_dimension] = hidden_state_shape.as_slice() else {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "MTP fusion hidden states must have shape [1, tokens, hidden_size]",
        });
    };
    if *batch_size != 1
        || *token_count <= 0
        || *hidden_dimension != hidden_size
        || next_token_shape.as_slice() != [1, *token_count]
    {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "MTP fusion hidden states and next-token indices have incompatible shapes",
        });
    }
    Ok(*token_count)
}
