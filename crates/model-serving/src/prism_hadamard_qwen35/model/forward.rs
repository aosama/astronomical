//! The straight text forward pass: embedding, decoder layers, final norm,
//! and the language-model head.
//!
//! Every rotated projection runs on activations the Hadamard transform
//! produced — the transform is part of the projection's input type, not a
//! branch — and each activation is rotated once for every projection that
//! reads it, with sibling projections of one activation fused into a single
//! launch. The layer arithmetic mirrors the qwen3_5 core's validated GDN
//! and full-attention composition exactly, minus the production serving
//! machinery (chunked prefill boundaries, fused decode prework, expert
//! paging), which the parity milestone does not exercise.

use astronomical_mlx_c_rust::MlxArray;

use crate::{ConvolutionState, FullAttentionKeyValueState, GatedDeltaRecurrentState};

use super::PrismHadamardModelError;
use super::weights::{DecoderLayerAttentionWeights, DenseMlpWeights};

mod full_attention;
mod gated_delta;

/// The per-sequence generation state for the straight forward pass.
pub(crate) struct StraightGenerationState {
    convolution_states: Vec<ConvolutionState>,
    recurrent_states: Vec<GatedDeltaRecurrentState>,
    key_value_states: Vec<FullAttentionKeyValueState>,
    /// How many tokens the state already covers (prefill chunks and decoded
    /// tokens together), feeding RoPE offsets and KV-cache views.
    sequence_position_offset: i32,
}

impl StraightGenerationState {
    pub(crate) fn empty() -> Self {
        Self {
            convolution_states: Vec::new(),
            recurrent_states: Vec::new(),
            key_value_states: Vec::new(),
            sequence_position_offset: 0,
        }
    }

    pub(crate) const fn sequence_position_offset(&self) -> i32 {
        self.sequence_position_offset
    }

    /// Advances the covered position by `token_count` without running a
    /// forward, for probes that drive one layer at a time.
    pub(crate) const fn advance_position(&mut self, token_count: i32) {
        self.sequence_position_offset += token_count;
    }

    /// Every retained state array plus the forward output, as the evaluation
    /// roots one step must materialize together. Convolution state branches
    /// off the forward input rather than feeding the output, so it is always
    /// an explicit root; the recurrent and key-value arrays are listed too so
    /// one step never leaves a lazy state graph for the next step to drag.
    pub(crate) fn evaluation_arrays<'state>(
        &'state self,
        output: &'state MlxArray,
    ) -> Vec<&'state MlxArray> {
        let mut evaluation_arrays = Vec::with_capacity(
            1 + self.convolution_states.len()
                + self.recurrent_states.len()
                + 2 * self.key_value_states.len(),
        );
        evaluation_arrays.push(output);
        evaluation_arrays.extend(
            self.convolution_states
                .iter()
                .filter_map(ConvolutionState::state),
        );
        evaluation_arrays.extend(
            self.recurrent_states
                .iter()
                .filter_map(GatedDeltaRecurrentState::state),
        );
        for key_value_state in &self.key_value_states {
            evaluation_arrays.extend(key_value_state.keys_state());
            evaluation_arrays.extend(key_value_state.values_state());
        }
        evaluation_arrays
    }
}

impl super::PrismHadamardTextModel {
    /// Runs one forward pass and returns vocabulary logits for its final token.
    pub(crate) fn forward_pass(
        &self,
        token_indices: &MlxArray,
        state: &mut StraightGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let final_hidden_states = self.forward_hidden_states(token_indices, state)?;
        self.logits_for_last_hidden_state(&final_hidden_states)
    }

    /// Runs every input position but retains only the final hidden row: earlier
    /// prompt rows advance decoder state, while only the terminal row is sampled.
    pub(crate) fn forward_hidden_states(
        &self,
        token_indices: &MlxArray,
        state: &mut StraightGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let text = self.pack.config().text();
        // The core route carries the leading batch axis: token indices are
        // [1, token_count] and every activation stays [1, token_count, ...].
        let token_count = token_indices.shape()[1] as i32;
        let rms_norm_epsilon = f32::from_bits(text.rms_norm_epsilon_bits);

        let mut hidden_states = self
            .embed_tokens(token_indices)
            .map_err(PrismHadamardModelError::Forward)?;

        for layer_index in 0..self.weights.decoder_layers.len() {
            hidden_states = self.forward_decoder_layer(layer_index, &hidden_states, state)?;
        }
        state.sequence_position_offset += token_count;

        let hidden_state_shape = hidden_states.shape();
        // Final RMS normalization is position-local, so discard earlier rows first;
        // their decoder work was needed for state, but no consumer needs their norm.
        let final_hidden_state_row = self
            .runtime
            .slice(
                &hidden_states,
                &[0, token_count - 1, 0],
                &[1, token_count, hidden_state_shape[2]],
                &[1, 1, 1],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        self.runtime
            .rms_norm(
                &final_hidden_state_row,
                &self.weights.final_normalization,
                rms_norm_epsilon,
            )
            .map_err(PrismHadamardModelError::Forward)
    }

    /// Runs one decoder layer (attention half, dense MLP, both residual adds)
    /// over `hidden_states` of shape `[1, token_count, hidden]`, advancing the
    /// layer's slot in `state`. The embedding and final norm stay outside so
    /// the measurement surface can time one layer in isolation.
    pub(crate) fn forward_decoder_layer(
        &self,
        layer_index: usize,
        hidden_states: &MlxArray,
        state: &mut StraightGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let text = self.pack.config().text();
        let token_count = hidden_states.shape()[1] as i32;
        let rms_norm_epsilon = f32::from_bits(text.rms_norm_epsilon_bits);
        let decoder_layer = &self.weights.decoder_layers[layer_index];
        let state_slot = self.decoder_layer_state_slots[layer_index];
        let normalized_input = self
            .runtime
            .rms_norm(
                hidden_states,
                &self.weights.input_normalization_weights[layer_index],
                rms_norm_epsilon,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let attention_output = match &decoder_layer.attention {
            DecoderLayerAttentionWeights::Linear(linear_layer) => {
                if state.convolution_states.len() <= state_slot {
                    state
                        .convolution_states
                        .push(ConvolutionState::empty_with_shape(
                            text.linear_conv_kernel_dim as i32,
                            (2 * text.linear_num_key_heads * text.linear_key_head_dim
                                + text.linear_num_value_heads * text.linear_value_head_dim)
                                as i32,
                        ));
                    state
                        .recurrent_states
                        .push(GatedDeltaRecurrentState::empty_with_shape(
                            text.linear_num_value_heads as i32,
                            text.linear_value_head_dim as i32,
                            text.linear_key_head_dim as i32,
                        ));
                }
                self.forward_linear_attention_layer(
                    &normalized_input,
                    token_count,
                    linear_layer,
                    state_slot,
                    state,
                )?
            }
            DecoderLayerAttentionWeights::Full(full_layer) => {
                if state.key_value_states.len() <= state_slot {
                    let key_value_state = FullAttentionKeyValueState::empty_with_growth_tokens(256)
                        .map_err(PrismHadamardModelError::ForwardRuntime)?;
                    state.key_value_states.push(key_value_state);
                }
                self.forward_full_attention_layer(
                    &normalized_input,
                    token_count,
                    full_layer,
                    rms_norm_epsilon,
                    state_slot,
                    state,
                )?
            }
        };
        let hidden_states = self
            .runtime
            .add(hidden_states, &attention_output)
            .map_err(PrismHadamardModelError::Forward)?;

        let normalized_attention = self
            .runtime
            .rms_norm(
                &hidden_states,
                &self.weights.post_attention_normalization_weights[layer_index],
                rms_norm_epsilon,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let feed_forward_output =
            self.forward_dense_mlp(&normalized_attention, &decoder_layer.mlp)?;
        self.runtime
            .add(&hidden_states, &feed_forward_output)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// Applies the vocabulary projection to a single final hidden row.
    pub(crate) fn logits_for_last_hidden_state(
        &self,
        final_hidden_states: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let rotated_hidden_states = self
            .rotate(final_hidden_states)
            .map_err(PrismHadamardModelError::Forward)?;
        self.weights
            .language_model_head
            .forward(&self.runtime, &rotated_hidden_states)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// The packed embedding lookup with its inverse rotation at the hidden
    /// width.
    pub(crate) fn embed_tokens(
        &self,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, astronomical_mlx_c_rust::MlxCError> {
        let hidden_width = self.pack.config().text().hidden_size as i32;
        let rotation = self.weights.rotation(hidden_width)?;
        self.weights
            .embedding
            .lookup(&self.runtime, rotation, token_indices)
    }

    /// Runs the dense SwiGLU MLP for one decoder layer.
    pub(super) fn forward_dense_mlp(
        &self,
        normalized_attention: &MlxArray,
        dense_mlp: &DenseMlpWeights,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        // One rotation of the normalized input feeds the one fused launch
        // that produces both the gate and the up projection.
        let rotated_input = self
            .rotate(normalized_attention)
            .map_err(PrismHadamardModelError::Forward)?;
        let [gate_activations, up_activations] = dense_mlp
            .gate_and_up
            .forward(&self.runtime, &rotated_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let activated_intermediate_states = self
            .runtime
            .apply_compiled_swiglu(&self.compiled_swiglu, &gate_activations, &up_activations)
            .map_err(PrismHadamardModelError::Forward)?;
        let rotated_intermediate_states = self
            .rotate(&activated_intermediate_states)
            .map_err(PrismHadamardModelError::Forward)?;
        dense_mlp
            .down
            .forward(&self.runtime, &rotated_intermediate_states)
            .map_err(PrismHadamardModelError::Forward)
    }
}
