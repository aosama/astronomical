//! A dense Qwen 3.5 multi-token-prediction head on the engine.
//!
//! Given a token and the model's final hidden state at the position before
//! it, the head returns logits for the token after it and its own final
//! hidden state, which stands in for the model's when it drafts one token
//! further. The hidden state it reads is the one after the model's final
//! norm, the state the model's output head reads; the caller normalizes the
//! model's rows before they arrive here (`target_rows_for_head`), while the
//! head's own output is already normalized.
//!
//! The math follows the head's published reference: each input is
//! normalized with its own scale, the embedding comes first in the joined
//! row, one projection brings the joined row back to the hidden width, and
//! one pre-norm full-attention layer with a dense SwiGLU MLP refines it.

use mlx_bridge::{MlxArray, MlxDtype};
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::dense::mtp_head_weights::DenseMtpHeadWeights;
use crate::qwen3_5_text::model::Qwen3_5Model;

use super::mtp_head::{MtpHeadOutput, MtpHeadState};

impl Qwen3_5Model {
    /// Runs the dense head over `token_indices` (`[1, tokens]` int32), each
    /// paired with the normalized hidden state at the position before it
    /// (`preceding_hidden_states`, `[1, tokens, hidden]`).
    pub(super) fn forward_dense_mtp_head(
        &self,
        head: &DenseMtpHeadWeights,
        token_indices: &MlxArray,
        preceding_hidden_states: &MlxArray,
        head_state: &mut MtpHeadState,
    ) -> Result<MtpHeadOutput, TextExecutionError> {
        let token_count = token_indices.shape()[1];
        let fused_states =
            self.fuse_dense_mtp_head_inputs(head, token_indices, preceding_hidden_states)?;
        let refined_states =
            self.forward_dense_mtp_head_layer(head, &fused_states, token_count, head_state)?;
        let normalized_states = self.runtime.rms_norm(
            &refined_states,
            &head.final_norm_scale,
            self.rms_norm_epsilon(),
        )?;
        let logits = self.quantized_linear(
            &normalized_states,
            &self.weights.language_model_head_weights,
        )?;
        Ok(MtpHeadOutput {
            logits: self.runtime.astype(&logits, MlxDtype::Float32)?,
            states: normalized_states,
        })
    }

    /// `[embedding norm(embedding), hidden norm(hidden)]`, projected back to
    /// the hidden width.
    fn fuse_dense_mtp_head_inputs(
        &self,
        head: &DenseMtpHeadWeights,
        token_indices: &MlxArray,
        preceding_hidden_states: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let epsilon = self.rms_norm_epsilon();
        let embeddings = self.context_free_token_meanings(token_indices)?;
        let normalized_embeddings =
            runtime.rms_norm(&embeddings, &head.embedding_norm_scale, epsilon)?;
        let normalized_hidden_states = runtime.astype(
            &runtime.rms_norm(preceding_hidden_states, &head.hidden_norm_scale, epsilon)?,
            normalized_embeddings.dtype(),
        )?;
        let joined_rows =
            runtime.concatenate_axis(&[&normalized_embeddings, &normalized_hidden_states], -1)?;
        self.quantized_linear(&joined_rows, &head.fusion_projection)
    }

    /// The head's one pre-norm decoder layer. Its attention keeps its own
    /// keys and values, at positions that count the head's own rows; RoPE
    /// depends only on the distance between rows, so that matches the
    /// reference's absolute positions.
    fn forward_dense_mtp_head_layer(
        &self,
        head: &DenseMtpHeadWeights,
        fused_states: &MlxArray,
        token_count: i32,
        head_state: &mut MtpHeadState,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let epsilon = self.rms_norm_epsilon();
        let attention_input =
            runtime.rms_norm(fused_states, &head.input_normalization_weight, epsilon)?;
        let rope_offset_tokens = head_state.row_count();
        let attention_output = self.forward_full_attention(
            &attention_input,
            token_count,
            rope_offset_tokens,
            &head.attention,
            &mut head_state.key_value_state,
            None,
        )?;
        let after_attention = runtime.add(fused_states, &attention_output)?;
        let mlp_input = runtime.rms_norm(
            &after_attention,
            &head.post_attention_normalization_weight,
            epsilon,
        )?;
        let mlp_output = self.forward_qwen3_5_dense_mlp(&mlp_input, &head.mlp)?;
        Ok(runtime.add(&after_attention, &mlp_output)?)
    }

    fn rms_norm_epsilon(&self) -> f32 {
        f32::from_bits(self.config.rms_norm_epsilon_bits())
    }
}
