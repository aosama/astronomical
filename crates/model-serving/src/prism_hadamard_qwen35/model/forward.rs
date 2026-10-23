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

use astronomical_runtime_integration::MlxRuntimeError;
use mlx_c_rust::MlxArray;

use crate::{ConvolutionState, FullAttentionKeyValueState, GatedDeltaRecurrentState};

use super::PrismHadamardModelError;
use super::PrismHadamardTextModel;
use super::weights::{DecoderLayerAttentionWeights, DenseMlpWeights};

/// The per-kind slot layout of the straight generation state, so persisted
/// tensor names (decoder-layer indexed) can address per-kind state slots.
pub(crate) struct PrismStateSlotLayout {
    /// Decoder-layer index of each full-attention slot, in slot order.
    pub(crate) full_attention_decoder_layer_indices: Vec<usize>,
    /// Decoder-layer index of each linear-attention slot, in slot order.
    pub(crate) linear_attention_decoder_layer_indices: Vec<usize>,
}

impl PrismStateSlotLayout {
    /// Rebuilds the decoder-layer-index to per-kind-slot mapping from the
    /// loaded weights, mirroring `decoder_layer_state_slots`.
    pub(crate) fn for_model(model: &PrismHadamardTextModel) -> Self {
        let mut full_attention_decoder_layer_indices = Vec::new();
        let mut linear_attention_decoder_layer_indices = Vec::new();
        for (decoder_layer_index, decoder_layer) in model.weights.decoder_layers.iter().enumerate()
        {
            match &decoder_layer.attention {
                DecoderLayerAttentionWeights::Linear(_) => {
                    linear_attention_decoder_layer_indices.push(decoder_layer_index);
                }
                DecoderLayerAttentionWeights::Full(_) => {
                    full_attention_decoder_layer_indices.push(decoder_layer_index);
                }
            }
        }
        Self {
            full_attention_decoder_layer_indices,
            linear_attention_decoder_layer_indices,
        }
    }
}

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

    /// Builds the per-layer slot owners a restored prompt prefix needs,
    /// mirroring the allocation order a fresh forward would have produced:
    /// one convolution/recurrent pair per linear layer and one key-value
    /// owner per full-attention layer, each still unallocated until the
    /// restored tensors seat them. Keep the kind matching in sync with
    /// `forward_decoder_layer`, which pushes the same owners lazily.
    pub(crate) fn restored_prefix_slot_owners(model: &PrismHadamardTextModel) -> Self {
        let text = model.pack.config().text();
        let mut generation_state = Self::empty();
        for decoder_layer in &model.weights.decoder_layers {
            match &decoder_layer.attention {
                DecoderLayerAttentionWeights::Linear(_) => {
                    generation_state
                        .convolution_states
                        .push(ConvolutionState::empty_with_shape(
                            text.linear_conv_kernel_dim as i32,
                            (2 * text.linear_num_key_heads * text.linear_key_head_dim
                                + text.linear_num_value_heads * text.linear_value_head_dim)
                                as i32,
                        ));
                    generation_state.recurrent_states.push(
                        GatedDeltaRecurrentState::empty_with_shape(
                            text.linear_num_value_heads as i32,
                            text.linear_value_head_dim as i32,
                            text.linear_key_head_dim as i32,
                        ),
                    );
                }
                DecoderLayerAttentionWeights::Full(_) => {
                    // The fresh-forward allocation path uses the same growth
                    // granularity, so a restored slab behaves identically to
                    // a forward-grown one.
                    generation_state.key_value_states.push(
                        FullAttentionKeyValueState::empty_with_growth_tokens(256)
                            .expect("the fixed growth granularity is always valid"),
                    );
                }
            }
        }
        generation_state
    }

    /// The allocated K/V slabs of one full-attention layer's slot, for the
    /// persistent prompt-cache capture bridge.
    pub(crate) fn key_value_slabs(&self, slot: usize) -> Option<(&MlxArray, &MlxArray)> {
        let key_value_state = self.key_value_states.get(slot)?;
        Some((
            key_value_state.keys_state()?,
            key_value_state.values_state()?,
        ))
    }

    /// Replaces one full-attention layer's slabs from restored SSD prompt-cache
    /// block tensors (already concatenated to the full restored length).
    pub(crate) fn restore_key_value_slabs(
        &mut self,
        slot: usize,
        restored_keys: MlxArray,
        restored_values: MlxArray,
    ) -> Result<(), MlxRuntimeError> {
        self.key_value_states[slot].restore_from_blocks(restored_keys, restored_values)
    }

    /// The allocated convolution rolling buffer of one linear layer's slot.
    pub(crate) fn convolution_state_array(&self, slot: usize) -> Option<&MlxArray> {
        self.convolution_states.get(slot)?.state()
    }

    /// Installs a restored convolution rolling buffer after its shape is
    /// validated against the owner's configured geometry.
    pub(crate) fn restore_convolution_state(
        &mut self,
        slot: usize,
        restored_state: MlxArray,
    ) -> Result<(), MlxRuntimeError> {
        self.convolution_states[slot].replace_state(restored_state)
    }

    /// The allocated gated-delta recurrent accumulator of one linear layer's slot.
    pub(crate) fn recurrent_state_array(&self, slot: usize) -> Option<&MlxArray> {
        self.recurrent_states.get(slot)?.state()
    }

    /// Installs a restored gated-delta recurrent accumulator.
    pub(crate) fn restore_recurrent_state(&mut self, slot: usize, restored_state: MlxArray) {
        self.recurrent_states[slot].restore_from_snapshot(restored_state)
    }

    /// Retargets the covered position after a restored prefix so RoPE offsets
    /// and KV-cache views continue from the restored length.
    pub(crate) fn set_sequence_position_offset(&mut self, covered_token_count: i32) {
        self.sequence_position_offset = covered_token_count;
    }

    pub(crate) const fn sequence_position_offset(&self) -> i32 {
        self.sequence_position_offset
    }

    /// Host-side byte estimate of every retained state array, matching the
    /// payload accounting the decoder-cache state types publish. The memory
    /// telemetry breakdown uses it as the live-context owner share.
    pub(crate) fn payload_byte_count(&self) -> u64 {
        let convolution_state_payload_bytes = self
            .convolution_states
            .iter()
            .map(|convolution_state| convolution_state.payload_byte_count())
            .fold(0_u64, u64::saturating_add);
        let recurrent_state_payload_bytes = self
            .recurrent_states
            .iter()
            .map(|recurrent_state| recurrent_state.payload_byte_count())
            .fold(0_u64, u64::saturating_add);
        let key_value_state_payload_bytes = self
            .key_value_states
            .iter()
            .map(|key_value_state| key_value_state.payload_byte_count())
            .fold(0_u64, u64::saturating_add);
        convolution_state_payload_bytes
            .saturating_add(recurrent_state_payload_bytes)
            .saturating_add(key_value_state_payload_bytes)
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
    ) -> Result<MlxArray, mlx_c_rust::MlxCError> {
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
