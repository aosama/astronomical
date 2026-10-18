//! The straight text forward pass: embedding, decoder layers, final norm,
//! and the language-model head.
//!
//! Every rotated projection applies its Hadamard transform unconditionally —
//! the transform is part of the projection type, not a branch. The layer
//! arithmetic mirrors the qwen3_5 core's validated GDN and full-attention
//! composition exactly, minus the production serving machinery (chunked
//! prefill boundaries, fused decode prework, expert paging), which the
//! parity milestone does not exercise.

use astronomical_mlx_c_rust::MlxArray;

use crate::qwen3_5_core::model::full_attention::qwen3_5_full_attention_step;
use crate::qwen3_5_core::model_math::gated_delta_sequence::qwen3_5_gated_delta_sequence;
use crate::{ConvolutionState, FullAttentionKeyValueState, GatedDeltaRecurrentState};

use super::PrismHadamardModelError;
use super::weights::{DecoderLayerAttentionWeights, DenseMlpWeights, LinearLayerWeights};

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
            .weights
            .embedding
            .lookup(&self.runtime, token_indices)
            .map_err(PrismHadamardModelError::Forward)?;

        let mut linear_layer_index = 0_usize;
        let mut full_layer_index = 0_usize;
        for (layer_index, decoder_layer) in self.weights.decoder_layers.iter().enumerate() {
            let normalized_input = self
                .runtime
                .rms_norm(
                    &hidden_states,
                    &self.weights.input_normalization_weights[layer_index],
                    rms_norm_epsilon,
                )
                .map_err(PrismHadamardModelError::Forward)?;
            let attention_output = match &decoder_layer.attention {
                DecoderLayerAttentionWeights::Linear(linear_layer) => {
                    if state.convolution_states.len() <= linear_layer_index {
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
                    let attention_output = self.forward_linear_attention_layer(
                        &normalized_input,
                        token_count,
                        linear_layer,
                        rms_norm_epsilon,
                        linear_layer_index,
                        state,
                    )?;
                    linear_layer_index += 1;
                    attention_output
                }
                DecoderLayerAttentionWeights::Full(full_layer) => {
                    if state.key_value_states.len() <= full_layer_index {
                        let key_value_state =
                            FullAttentionKeyValueState::empty_with_growth_tokens(256)
                                .map_err(PrismHadamardModelError::ForwardRuntime)?;
                        state.key_value_states.push(key_value_state);
                    }
                    let attention_output = self.forward_full_attention_layer(
                        &normalized_input,
                        token_count,
                        full_layer,
                        rms_norm_epsilon,
                        full_layer_index,
                        state,
                    )?;
                    full_layer_index += 1;
                    attention_output
                }
            };
            hidden_states = self
                .runtime
                .add(&hidden_states, &attention_output)
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
            hidden_states = self
                .runtime
                .add(&hidden_states, &feed_forward_output)
                .map_err(PrismHadamardModelError::Forward)?;
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

    /// Applies the vocabulary projection to a single final hidden row.
    pub(crate) fn logits_for_last_hidden_state(
        &self,
        final_hidden_states: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        self.weights
            .language_model_head
            .forward(&self.runtime, final_hidden_states)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// Runs one linear-attention layer through the selected shared gated-delta route.
    #[allow(clippy::too_many_arguments)]
    fn forward_linear_attention_layer(
        &self,
        normalized_input: &MlxArray,
        token_count: i32,
        layer: &LinearLayerWeights,
        rms_norm_epsilon: f32,
        linear_layer_index: usize,
        state: &mut StraightGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let text = self.pack.config().text();
        let linear_value_head_count = text.linear_num_value_heads as i32;
        let linear_value_dimension =
            (text.linear_num_value_heads * text.linear_value_head_dim) as i32;

        let mixed_queries_keys_values = layer
            .query_key_value
            .forward(&self.runtime, normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let output_gate = layer
            .output_gate
            .forward(&self.runtime, normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let output_gate = self
            .runtime
            .reshape(
                &output_gate,
                &[
                    1,
                    token_count,
                    linear_value_head_count,
                    text.linear_value_head_dim as i32,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let update_logits = layer
            .update_rate
            .forward(&self.runtime, normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let decay_inputs = layer
            .decay_interval
            .forward(&self.runtime, normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let update_rates = self
            .runtime
            .sigmoid(&update_logits)
            .map_err(PrismHadamardModelError::Forward)?;
        let decays = self
            .runtime
            .apply_compiled_gated_delta_decay(
                &self.compiled_elementwise_graphs,
                &layer.decay_rate_logarithm,
                &decay_inputs,
                &layer.decay_interval_bias,
            )
            .map_err(PrismHadamardModelError::Forward)?;

        let convolution_input = state.convolution_states[linear_layer_index]
            .update(&self.runtime, &mixed_queries_keys_values, token_count)
            .map_err(PrismHadamardModelError::ForwardRuntime)?;
        let head_queries_keys_values = self.convolution_to_normalized_heads(
            &convolution_input,
            token_count,
            layer,
            rms_norm_epsilon,
        )?;
        let recurrent_state = &mut state.recurrent_states[linear_layer_index];
        let current_recurrent_state = recurrent_state
            .current_or_zero(&self.runtime)
            .map_err(PrismHadamardModelError::ForwardRuntime)?;
        let (recurrent_output, next_recurrent_state) = qwen3_5_gated_delta_sequence(
            &self.runtime,
            self.gated_delta_kernel.as_ref(),
            &head_queries_keys_values.0,
            &head_queries_keys_values.1,
            &head_queries_keys_values.2,
            &decays,
            &update_rates,
            &current_recurrent_state,
        )
        .map_err(PrismHadamardModelError::ForwardRuntime)?;
        recurrent_state.set_next(next_recurrent_state);

        let normalized_output = self
            .runtime
            .rms_norm(&recurrent_output, &layer.normalization, rms_norm_epsilon)
            .map_err(PrismHadamardModelError::Forward)?;
        let gated_output = self
            .runtime
            .apply_compiled_precise_swiglu(
                &self.compiled_elementwise_graphs,
                &normalized_output,
                &output_gate,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let gated_output = self
            .runtime
            .reshape(&gated_output, &[1, token_count, linear_value_dimension])
            .map_err(PrismHadamardModelError::Forward)?;
        layer
            .output
            .forward(&self.runtime, &gated_output)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// The composed depthwise convolution, SiLU, head split, and both
    /// folded-scale RMS normalizations, mirroring the core's prefill
    /// arithmetic exactly.
    fn convolution_to_normalized_heads(
        &self,
        convolution_input: &MlxArray,
        token_count: i32,
        layer: &LinearLayerWeights,
        rms_norm_epsilon: f32,
    ) -> Result<(MlxArray, MlxArray, MlxArray), PrismHadamardModelError> {
        let text = self.pack.config().text();
        let linear_key_head_count = text.linear_num_key_heads as i32;
        let linear_value_head_count = text.linear_num_value_heads as i32;
        let linear_head_dimension = text.linear_key_head_dim as i32;
        let linear_key_dimension = linear_key_head_count * linear_head_dimension;
        let linear_convolution_dimension =
            2 * linear_key_dimension + linear_value_head_count * text.linear_value_head_dim as i32;

        let convolution_output = self
            .runtime
            .conv1d(
                convolution_input,
                &layer.convolution,
                1,
                0,
                1,
                linear_convolution_dimension,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let convolution_output = self
            .runtime
            .apply_compiled_silu(&self.compiled_elementwise_graphs, &convolution_output)
            .map_err(PrismHadamardModelError::Forward)?;

        let queries = super::weights::slice_last_dimension(
            &self.runtime,
            &convolution_output,
            0,
            linear_key_dimension,
        )?;
        let queries = self
            .runtime
            .reshape(
                &queries,
                &[1, token_count, linear_key_head_count, linear_head_dimension],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let keys = super::weights::slice_last_dimension(
            &self.runtime,
            &convolution_output,
            linear_key_dimension,
            linear_key_dimension * 2,
        )?;
        let keys = self
            .runtime
            .reshape(
                &keys,
                &[1, token_count, linear_key_head_count, linear_head_dimension],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let values = super::weights::slice_last_dimension(
            &self.runtime,
            &convolution_output,
            linear_key_dimension * 2,
            linear_convolution_dimension,
        )?;
        let values = self
            .runtime
            .reshape(
                &values,
                &[
                    1,
                    token_count,
                    linear_value_head_count,
                    text.linear_value_head_dim as i32,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;

        let queries = self
            .runtime
            .rms_norm(&queries, &self.gdn_query_scale_weight, rms_norm_epsilon)
            .map_err(PrismHadamardModelError::Forward)?;
        let keys = self
            .runtime
            .rms_norm(&keys, &self.gdn_key_scale_weight, rms_norm_epsilon)
            .map_err(PrismHadamardModelError::Forward)?;
        Ok((queries, keys, values))
    }

    /// Runs one full-attention layer, mirroring the core's forward exactly:
    /// gated-query split, q/k RMS norms, RoPE, append-only KV state, and the
    /// shared attention step with the learned output gate.
    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_layer(
        &self,
        normalized_input: &MlxArray,
        token_count: i32,
        layer: &super::weights::FullLayerWeights,
        rms_norm_epsilon: f32,
        full_layer_index: usize,
        state: &mut StraightGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let text = self.pack.config().text();
        let query_head_count = text.num_attention_heads as i32;
        let key_value_head_count = text.num_key_value_heads as i32;
        let attention_head_dimension = text.head_dim as i32;
        let attention_projection_head_dimension = attention_head_dimension * 2;
        let rope_offset_tokens = state.sequence_position_offset;

        let query_projection = layer
            .query
            .forward(&self.runtime, normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let query_projection = self
            .runtime
            .reshape(
                &query_projection,
                &[
                    1,
                    token_count,
                    query_head_count,
                    attention_projection_head_dimension,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let queries = super::weights::slice_last_dimension(
            &self.runtime,
            &query_projection,
            0,
            attention_head_dimension,
        )?;
        let output_gate = super::weights::slice_last_dimension(
            &self.runtime,
            &query_projection,
            attention_head_dimension,
            attention_projection_head_dimension,
        )?;
        let output_gate = self
            .runtime
            .reshape(
                &output_gate,
                &[1, token_count, query_head_count * attention_head_dimension],
            )
            .map_err(PrismHadamardModelError::Forward)?;

        let keys = layer
            .key
            .forward(&self.runtime, normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let keys = self
            .runtime
            .reshape(
                &keys,
                &[
                    1,
                    token_count,
                    key_value_head_count,
                    attention_head_dimension,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let values = layer
            .value
            .forward(&self.runtime, normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let values = self
            .runtime
            .reshape(
                &values,
                &[
                    1,
                    token_count,
                    key_value_head_count,
                    attention_head_dimension,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;

        let normalized_queries = self
            .runtime
            .rms_norm(&queries, &layer.query_normalization, rms_norm_epsilon)
            .map_err(PrismHadamardModelError::Forward)?;
        let normalized_keys = self
            .runtime
            .rms_norm(&keys, &layer.key_normalization, rms_norm_epsilon)
            .map_err(PrismHadamardModelError::Forward)?;

        let transposed_queries = self
            .runtime
            .transpose_axes(&normalized_queries, &[0, 2, 1, 3])
            .map_err(PrismHadamardModelError::Forward)?;
        let transposed_keys = self
            .runtime
            .transpose_axes(&normalized_keys, &[0, 2, 1, 3])
            .map_err(PrismHadamardModelError::Forward)?;
        let transposed_values = self
            .runtime
            .transpose_axes(&values, &[0, 2, 1, 3])
            .map_err(PrismHadamardModelError::Forward)?;

        let rope_base = f32::from_bits(text.rope_theta_bits);
        let rotary_dimension =
            (text.head_dim as f32 * f32::from_bits(text.partial_rotary_factor_bits)) as i32;
        let rotated_queries = self
            .runtime
            .rope(
                &transposed_queries,
                rotary_dimension,
                rope_base,
                rope_offset_tokens,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let rotated_keys = self
            .runtime
            .rope(
                &transposed_keys,
                rotary_dimension,
                rope_base,
                rope_offset_tokens,
            )
            .map_err(PrismHadamardModelError::Forward)?;

        let key_value_state = &mut state.key_value_states[full_layer_index];
        let previous_storage_offset_tokens = key_value_state.offset_tokens();
        let (active_keys, active_values) = key_value_state
            .update_and_fetch(
                &self.runtime,
                &rotated_keys,
                &transposed_values,
                previous_storage_offset_tokens,
            )
            .map_err(PrismHadamardModelError::ForwardRuntime)?;

        let attention_scale = 1.0 / (text.head_dim as f32).sqrt();
        let gated_output = qwen3_5_full_attention_step(
            &self.runtime,
            &self.compiled_elementwise_graphs,
            &rotated_queries,
            &active_keys,
            &active_values,
            &output_gate,
            attention_scale,
        )
        .map_err(PrismHadamardModelError::ForwardRuntime)?;
        layer
            .output
            .forward(&self.runtime, &gated_output)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// Runs the dense SwiGLU MLP for one decoder layer.
    fn forward_dense_mlp(
        &self,
        normalized_attention: &MlxArray,
        dense_mlp: &DenseMlpWeights,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let gate_activations = dense_mlp
            .gate
            .forward(&self.runtime, normalized_attention)
            .map_err(PrismHadamardModelError::Forward)?;
        let up_activations = dense_mlp
            .up
            .forward(&self.runtime, normalized_attention)
            .map_err(PrismHadamardModelError::Forward)?;
        let activated_intermediate_states = self
            .runtime
            .apply_compiled_swiglu(&self.compiled_swiglu, &gate_activations, &up_activations)
            .map_err(PrismHadamardModelError::Forward)?;
        dense_mlp
            .down
            .forward(&self.runtime, &activated_intermediate_states)
            .map_err(PrismHadamardModelError::Forward)
    }
}
