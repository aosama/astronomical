//! What a decision model reads from a Qwen3.5 backbone instead of sampling
//! from it.
//!
//! A decision model built on Qwen3.5 (Kev, Clef) runs the backbone without
//! generating: it reads the backbone's final hidden states at chosen
//! positions and scores the allowed answers with its own head. Clef also
//! reads the rows of the output projection that belong to each option's
//! tokens. Serving needs only these two reads; a whole-model training also
//! runs several sequences through the layers at once
//! (`final_normalized_hidden_states_of_rows`), recomputing each layer
//! during the backward pass when one text is too long to keep.
//!
//! A third need, answering several questions that each continue one shared
//! state, is served by the decoder state's own checkpoint and restore: the
//! state is prefilled once, checkpointed, and restored before each question
//! continues it from the state's length. Positions therefore simply continue
//! from the state for every question; nothing restarts.

use crate::qwen3_5_core::model::target_forward_graph::{
    Qwen3_5ForwardPositions, Qwen3_5TargetForwardGraph,
};
use mlx_bridge::{MlxArray, MlxDtype};

use super::{Qwen3_5Model, RequestDecoderStateStack, TextExecutionError};
use crate::PerformanceAttribution;
use crate::qwen3_5_text::Qwen3_5MoEPagedExecutionMode;

/// Runs one decoder layer of a training's batched forward pass so that
/// its activations are recomputed during the backward pass instead of kept
/// (`adapter_training::layer_recomputation` says why and what it costs).
pub trait DecoderLayerRecomputation {
    /// Layer `layer_index` over `hidden_states`, as
    /// [`Qwen3_5Model::empty_state_decoder_layer`] computes it.
    fn recomputed_layer(
        &self,
        hidden_states: &MlxArray,
        layer_index: usize,
        token_count: i32,
    ) -> Result<MlxArray, TextExecutionError>;
}

impl Qwen3_5Model {
    /// Layer `layer_index` of a training's batched forward pass over
    /// `[rows, token_count, hidden]` hidden states (`empty_state_forward`).
    ///
    /// The layer comes from where every forward of this model finds it: a
    /// dense model's streamer, which holds the layers a backbone keeps
    /// (`materialize_and_keep_every_dense_layer`). The model base's own
    /// decoder layers of a dense model are never evaluated, so reading
    /// them here would load a second copy of every layer.
    pub fn empty_state_decoder_layer(
        &self,
        hidden_states: &MlxArray,
        layer_index: usize,
        token_count: i32,
    ) -> Result<MlxArray, TextExecutionError> {
        if layer_index >= self.config.layer_count() as usize {
            return Err(TextExecutionError::InvalidInput {
                description: "a decoder layer index is past the model's layers",
            });
        }
        let forward_layer_weights = self.forward_layer_weights(layer_index)?;
        self.decoder_layer_from_empty_state(
            hidden_states,
            token_count,
            layer_index,
            forward_layer_weights.decoder_layer_weights(),
        )
    }

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
    ) -> Result<MlxArray, TextExecutionError> {
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
                Qwen3_5ForwardPositions {
                    token_count,
                    starting_position_tokens,
                    token_position_offsets: None,
                },
                request_decoder_state,
                None,
                Qwen3_5MoEPagedExecutionMode::ProductionDefault,
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

    /// The final normalized hidden states of several token sequences at
    /// once, each read from its first token with nothing before it, as a
    /// lazy float32 graph `[rows, longest, hidden_size]`: what a decision
    /// model's training differentiates (`empty_state_forward` says why).
    /// Shorter rows are padded at the end with token 0; every layer is
    /// causal, so a real position's states never see the padding. With
    /// `layer_recomputation`, each layer runs through it, so its activations
    /// are recomputed during the backward pass rather than kept.
    pub fn final_normalized_hidden_states_of_rows(
        &self,
        token_rows: &[&[u32]],
        layer_recomputation: Option<&dyn DecoderLayerRecomputation>,
    ) -> Result<MlxArray, TextExecutionError> {
        let (token_indices, token_count) = self.padded_token_indices(token_rows)?;
        let mut hidden_states = self.context_free_token_meanings(&token_indices)?;
        for layer_index in 0..self.config.layer_count() as usize {
            hidden_states = match layer_recomputation {
                Some(layer_recomputation) => layer_recomputation.recomputed_layer(
                    &hidden_states,
                    layer_index,
                    token_count,
                )?,
                None => self.empty_state_decoder_layer(&hidden_states, layer_index, token_count)?,
            };
        }
        let normalized_hidden_states = self.runtime.rms_norm(
            &hidden_states,
            &self.weights.final_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        Ok(self
            .runtime
            .astype(&normalized_hidden_states, MlxDtype::Float32)?)
    }

    /// `token_rows` as one `[rows, longest]` index array, shorter rows padded
    /// at the end with token 0, and the longest row's length.
    fn padded_token_indices(
        &self,
        token_rows: &[&[u32]],
    ) -> Result<(MlxArray, i32), TextExecutionError> {
        let mut longest_row = 0;
        for token_row in token_rows {
            longest_row = longest_row.max(token_row.len());
        }
        if token_rows.is_empty() || longest_row == 0 {
            return Err(TextExecutionError::InvalidInput {
                description: "a batched readout needs at least one token",
            });
        }
        let mut padded_token_ids = vec![0_i32; token_rows.len() * longest_row];
        for (row_index, token_row) in token_rows.iter().enumerate() {
            for (position, token_id) in token_row.iter().enumerate() {
                if *token_id >= self.config.vocabulary_size() {
                    return Err(TextExecutionError::InvalidInput {
                        description: "token ID is outside the vocabulary",
                    });
                }
                // Below the vocabulary size, which fits an i32.
                padded_token_ids[row_index * longest_row + position] = *token_id as i32;
            }
        }
        let token_count = longest_row as i32;
        let token_indices = self
            .runtime
            .array_from_i32(&padded_token_ids, &[token_rows.len() as i32, token_count])?;
        Ok((token_indices, token_count))
    }

    /// The rows of the output projection for `token_ids`, as float32
    /// `[token_ids.len(), hidden_size]`, dequantized when the projection is
    /// stored quantized. With tied embeddings the projection is the
    /// embedding table, so these are the tokens' input embeddings.
    pub fn output_embedding_rows(&self, token_ids: &[u32]) -> Result<MlxArray, TextExecutionError> {
        if token_ids.is_empty() {
            return Err(TextExecutionError::InvalidInput {
                description: "an output-embedding read needs at least one token",
            });
        }
        let mut row_ids = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            if *token_id >= self.config.vocabulary_size() {
                return Err(TextExecutionError::InvalidInput {
                    description: "token ID is outside the output projection",
                });
            }
            // Below the vocabulary size, which fits an i32, so this cannot fail.
            row_ids.push(*token_id as i32);
        }
        let selected_rows = self
            .weights
            .language_model_head_weights
            .dequantized_rows(&self.runtime, &row_ids)
            .ok_or(TextExecutionError::InvalidInput {
                description: "a codebook-encoded output projection has no readable rows",
            })??;
        let float32_rows = self.runtime.astype(&selected_rows, MlxDtype::Float32)?;
        self.runtime.evaluate_arrays(&[&float32_rows])?;
        Ok(float32_rows)
    }
}
