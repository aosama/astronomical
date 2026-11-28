//! Flash-Next's multi-token-prediction head on the engine.
//!
//! Given a token and the target's residual lanes at the position before it,
//! the head returns logits for the token after it and its own lanes, which
//! stand in for the target's when it drafts one token further. It keeps its
//! own attention keys and values and its own indexer keys, in positions that
//! count the head's own rows.
//!
//! Its experts stay in memory whole, so a draft step waits for no storage
//! read and no route readback: the routed arithmetic is the streaming
//! engine's own, over one page that holds every expert.

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntimeError;
use text_generation::execution_error::TextExecutionError;

use jangtq2_qwen35::configuration::Jangtq2CodebookRotation;

use crate::PerformanceAttribution;
use crate::qwen3_5_core::configuration::Qwen4ExpArchitecture;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
};
use crate::qwen3_5_text::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;
use crate::qwen3_5_text::model::Qwen3_5Model;
use crate::qwen4_exp::centered_norm::grouped_rms_norm;
use crate::qwen4_exp::mtp_head_weights::FlashNextMtpHeadWeights;

use super::mtp_head::{MtpHeadOutput, MtpHeadState};

/// Every expert of the head's layer as one page, in expert order, so an
/// expert's id is its slot.
fn head_expert_page(
    head: &FlashNextMtpHeadWeights,
) -> Result<Qwen3_5PagedExpertWeights, MlxRuntimeError> {
    Ok(Qwen3_5PagedExpertWeights {
        gate_projection: head.expert_gate_projection.retained_reference()?,
        up_projection: head.expert_up_projection.retained_reference()?,
        down_projection: head.expert_down_projection.retained_reference()?,
        gate_activation_rotation: Jangtq2CodebookRotation::NoRotation,
        up_activation_rotation: Jangtq2CodebookRotation::NoRotation,
        down_activation_rotation: Jangtq2CodebookRotation::NoRotation,
    })
}

impl Qwen3_5Model {
    /// Runs Flash-Next's head over `token_indices`, each paired with the
    /// target's lanes at the position before it (`preceding_lanes`,
    /// `[1, tokens, lanes * hidden]`).
    pub(super) fn forward_flash_next_mtp_head(
        &self,
        head: &FlashNextMtpHeadWeights,
        token_indices: &MlxArray,
        preceding_lanes: &MlxArray,
        head_state: &mut MtpHeadState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MtpHeadOutput, TextExecutionError> {
        let architecture = self.config.qwen4_exp_architecture().ok_or(
            TextExecutionError::InvalidInput {
                description: "Flash-Next's multi-token-prediction head needs a Flash-Next model",
            },
        )?;
        let token_count = token_indices.shape()[1];
        let fused_lanes =
            self.fuse_mtp_head_inputs(head, token_indices, preceding_lanes, &architecture)?;
        let lanes = self.forward_mtp_head_layer(
            head,
            &fused_lanes,
            token_count,
            head_state,
            &architecture,
            performance_attribution,
        )?;
        let (mixed, _) = self.read_lanes(
            &lanes,
            &head.final_norm_scale,
            &head.final_mixer,
            &architecture,
        )?;
        let logits = self.quantized_linear(&mixed, &self.weights.language_model_head_weights)?;
        Ok(MtpHeadOutput {
            logits: self
                .runtime
                .astype(&logits, mlx_bridge::MlxDtype::Float32)?,
            states: lanes,
        })
    }

    /// The head's first lanes: the projected embedding of each token added
    /// to every projected lane. The lanes are normalized together, as one
    /// vector, and projected one by one.
    fn fuse_mtp_head_inputs(
        &self,
        head: &FlashNextMtpHeadWeights,
        token_indices: &MlxArray,
        preceding_lanes: &MlxArray,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let hidden_size = self.config.hidden_size() as i32;
        let lane_count = architecture.residual_lane_count as i32;
        let epsilon = f32::from_bits(self.config.rms_norm_epsilon_bits());
        let token_count = token_indices.shape()[1];
        let embeddings = self.context_free_token_meanings(token_indices)?;
        let projected_embeddings = self.quantized_linear(
            &grouped_rms_norm(
                runtime,
                &embeddings,
                &head.embedding_norm_scale,
                hidden_size,
                epsilon,
            )?,
            &head.embedding_projection,
        )?;
        let normalized_lanes = grouped_rms_norm(
            runtime,
            preceding_lanes,
            &head.lane_norm_scale,
            hidden_size * lane_count,
            epsilon,
        )?;
        let projected_lanes = self.quantized_linear(
            &runtime.reshape(
                &normalized_lanes,
                &[1, token_count, lane_count, hidden_size],
            )?,
            &head.lane_projection,
        )?;
        let fused = runtime.add(
            &runtime.expand_dims(&projected_embeddings, -2)?,
            &runtime.astype(&projected_lanes, projected_embeddings.dtype())?,
        )?;
        Ok(runtime.reshape(&fused, &[1, token_count, lane_count * hidden_size])?)
    }

    /// The head's decoder layer: attention, then routed experts, each read
    /// from and written back to the lanes.
    fn forward_mtp_head_layer(
        &self,
        head: &FlashNextMtpHeadWeights,
        lanes: &MlxArray,
        token_count: i32,
        head_state: &mut MtpHeadState,
        architecture: &Qwen4ExpArchitecture,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let layer = &head.layer;
        let (
            Some(flash_next),
            Qwen3_5AttentionWeights::Full(attention_weights),
            Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(expert_weights),
        ) = (
            layer.flash_next(),
            &layer.attention_weights,
            &layer.mlp_weights,
        )
        else {
            return Err(TextExecutionError::InvalidInput {
                description: "the head's layer is a Flash-Next full-attention layer with experts",
            });
        };
        let indexer = flash_next.sparse_attention_indexer.as_ref().ok_or(
            TextExecutionError::MissingTensor {
                tensor_name: "mtp.layers.0.self_attn.indexer.index_qk_proj".to_owned(),
            },
        )?;
        let attention_read = self.read_lanes_for_block(
            lanes,
            &layer.input_normalization_weight,
            &flash_next.lane_connections.attention,
            architecture,
        )?;
        let rope_offset_tokens = head_state.row_count();
        let attention_output = self.forward_sparse_full_attention(
            &attention_read.block_input,
            token_count,
            rope_offset_tokens,
            attention_weights,
            indexer,
            &mut head_state.key_value_state,
            &mut head_state.indexer_state,
            architecture,
        )?;
        let lanes = self.write_back_to_lanes(&attention_output, &attention_read)?;
        let expert_read = self.read_lanes_for_block(
            &lanes,
            &layer.post_attention_normalization_weight,
            &flash_next.lane_connections.feed_forward,
            architecture,
        )?;
        let (selected_expert_ids, selected_scores) =
            self.routed_expert_selection(&expert_read.block_input, expert_weights)?;
        let routed_output = self.forward_moe_with_streamed_weights(
            &expert_read.block_input,
            &head_expert_page(head)?,
            &selected_expert_ids,
            &selected_scores,
            performance_attribution,
        )?;
        let expert_output = self.combine_paged_sparse_and_shared_outputs(
            &expert_read.block_input,
            expert_weights,
            &routed_output,
            false,
        )?;
        self.write_back_to_lanes(&expert_output, &expert_read)
    }
}
