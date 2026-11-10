//! What a decision model reads from a Qwen3.5 backbone instead of sampling
//! from it.
//!
//! A decision model built on Qwen3.5 (Kev, Clef) runs the backbone without
//! generating: it reads the backbone's final hidden states at chosen
//! positions and scores the allowed answers with its own head. Clef also
//! reads the rows of the output projection that belong to each option's
//! tokens. These two reads are all such a model needs from the engine.
//!
//! A third need, answering several questions that each continue one shared
//! state, is served by the decoder state's own checkpoint and restore: the
//! state is prefilled once, checkpointed, and restored before each question
//! continues it from the state's length. Positions therefore simply continue
//! from the state for every question; nothing restarts.

use mlx_bridge::{MlxArray, MlxDtype};

use super::decoder_layer_weights::Qwen3_5AffineWeights;
use super::{Qwen3_5ExecutionError, Qwen3_5ResidentModel, RequestDecoderStateStack};
use crate::PerformanceAttribution;

impl Qwen3_5ResidentModel {
    /// The width of one hidden state: the readout's row width, and the
    /// width a decision head's projections take.
    #[must_use]
    pub fn hidden_size(&self) -> u32 {
        self.config.hidden_size()
    }

    /// How many rows the output projection has; a token id must be below it.
    #[must_use]
    pub fn vocabulary_size(&self) -> u32 {
        self.config.vocabulary_size()
    }

    /// The backbone's final hidden states for every position of `token_ids`,
    /// after the final normalization, as float32 `[token_count, hidden_size]`.
    ///
    /// The tokens continue `request_decoder_state` from
    /// `starting_position_tokens`, and the state advances past them, exactly
    /// as a prefill would. No vocabulary logits are computed: a decision
    /// model never samples, and projecting every position through the
    /// vocabulary would cost far more than the forward it reads.
    ///
    /// Every position is returned in float32, so a caller that reads only
    /// some of a long input should prefill the rest with `prefill_chunk`
    /// and read out only the run it needs: sixteen thousand rows of a
    /// 4,096-wide model are 256 MB nobody reads.
    pub fn final_normalized_hidden_states(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let (token_indices, token_count) = self.validated_token_indices(
            token_ids,
            starting_position_tokens,
            None,
            request_decoder_state,
        )?;
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        let forward_output = self
            .build_target_forward_graph_from_token_indices_with_position_offsets(
                &token_indices,
                token_count,
                starting_position_tokens,
                None,
                request_decoder_state,
                None,
                &mut disabled_performance_attribution,
                false,
            )?;
        let normalized_hidden_states = self.runtime.rms_norm(
            forward_output.pre_final_normalization_hidden_states(),
            &self.weights.final_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        // The decision heads score in float32, as the reference implementations
        // do; widening here keeps the readout identical for every consumer.
        let float32_hidden_states = self
            .runtime
            .astype(&normalized_hidden_states, MlxDtype::Float32)?;
        let readout_rows = self.runtime.reshape(
            &float32_hidden_states,
            &[token_count, self.config.hidden_size() as i32],
        )?;
        self.evaluate_forward_state(&readout_rows, request_decoder_state)?;
        Ok(readout_rows)
    }

    /// The rows of the output projection for `token_ids`, as float32
    /// `[token_ids.len(), hidden_size]`, dequantized when the projection is
    /// stored quantized. With tied embeddings the projection is the
    /// embedding table, so these are the tokens' input embeddings.
    pub fn output_embedding_rows(
        &self,
        token_ids: &[u32],
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        if token_ids.is_empty() {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "an output-embedding read needs at least one token",
            });
        }
        let mut row_ids = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            if *token_id >= self.config.vocabulary_size() {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "token ID is outside the output projection",
                });
            }
            // Below the vocabulary size, which fits an i32, so this cannot fail.
            row_ids.push(*token_id as i32);
        }
        let row_id_array = self
            .runtime
            .array_from_i32(&row_ids, &[row_ids.len() as i32])?;
        let selected_rows = match &self.weights.language_model_head_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                self.runtime.take_axis(weight, &row_id_array, 0)?
            }
            Qwen3_5AffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_bits,
                quantization_group_size,
            } => self.runtime.dequantize_affine(
                &self.runtime.take_axis(packed_weight, &row_id_array, 0)?,
                &self
                    .runtime
                    .take_axis(quantization_scales, &row_id_array, 0)?,
                &self
                    .runtime
                    .take_axis(quantization_biases, &row_id_array, 0)?,
                *quantization_group_size,
                *quantization_bits,
            )?,
            Qwen3_5AffineWeights::Jangtq2Codebook { .. } => {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "a codebook-encoded output projection has no readable rows",
                });
            }
        };
        let float32_rows = self.runtime.astype(&selected_rows, MlxDtype::Float32)?;
        self.runtime.evaluate_arrays(&[&float32_rows])?;
        Ok(float32_rows)
    }
}
