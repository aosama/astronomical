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

use mlx_bridge::MlxArray;

use crate::{
    AstroDecoderStateCacheState, AstroSsdPromptCacheBoundaryCheckpointCollector,
    RequestDecoderStateStack,
};

use super::PrismHadamardModelError;
use super::weights::{DecoderLayerAttentionWeights, DenseMlpWeights};

mod full_attention;
mod gated_delta;

use gated_delta::{LayerBoundaryCheckpoints, LinearLayerState};

/// Every retained `astro_decoder_state_cache` array plus the forward output,
/// as the evaluation roots one step must materialize together. Convolution
/// state branches off the forward input rather than feeding the output, so it
/// is always an explicit root; the recurrent and key-value arrays are listed
/// too so one step never leaves a lazy state graph for the next step to drag.
///
/// The roots keep the order the per-kind state slots gave them before the
/// state moved onto the shared decoder-layer stack: the output, every
/// convolution buffer, every recurrent accumulator, then every key and value
/// slab, each group in decoder-layer order. Each array's value does not
/// depend on that order, but MLX schedules a multi-root evaluation in root
/// order, so keeping it keeps the dispatch sequence the speed proofs measured.
pub fn request_decoder_state_evaluation_arrays<'state>(
    request_decoder_state: &'state RequestDecoderStateStack,
    output: &'state MlxArray,
) -> Vec<&'state MlxArray> {
    let decoder_layer_states = request_decoder_state.layers();
    let mut evaluation_arrays = Vec::with_capacity(1 + 2 * decoder_layer_states.len());
    evaluation_arrays.push(output);
    for decoder_layer_state in decoder_layer_states {
        if let AstroDecoderStateCacheState::Composite { convolution, .. } = decoder_layer_state {
            evaluation_arrays.extend(convolution.state());
        }
    }
    for decoder_layer_state in decoder_layer_states {
        if let AstroDecoderStateCacheState::Composite { recurrent, .. } = decoder_layer_state {
            evaluation_arrays.extend(recurrent.state());
        }
    }
    for decoder_layer_state in decoder_layer_states {
        if let AstroDecoderStateCacheState::AppendOnlyAttention { attention } = decoder_layer_state
        {
            evaluation_arrays.extend(attention.keys_state());
            evaluation_arrays.extend(attention.values_state());
        }
    }
    evaluation_arrays
}

/// The rotary offset of a forward's first token, in the i32 form RoPE takes,
/// mirroring how the shared Qwen 3.5 forward converts its starting position.
pub(super) fn rope_offset_tokens(
    starting_position_tokens: u32,
) -> Result<i32, PrismHadamardModelError> {
    i32::try_from(starting_position_tokens).map_err(|_| {
        PrismHadamardModelError::InvalidGenerationRequest {
            description: "the starting position exceeds the MLX int32 range".to_owned(),
        }
    })
}

impl super::PrismHadamardTextModel {
    /// Creates one request's empty decoder state: one owner per decoder layer,
    /// each unallocated until its first forward or a restored prefix seats it.
    pub fn empty_request_decoder_state(
        &self,
    ) -> Result<RequestDecoderStateStack, PrismHadamardModelError> {
        super::super::astro_decoder_state_cache_layout::prism_request_decoder_state(
            &self.astro_decoder_state_cache_layout,
        )
        .map_err(PrismHadamardModelError::ForwardRuntime)
    }

    /// Runs one forward pass from `starting_position_tokens` and returns
    /// vocabulary logits for its final token. As in the shared Qwen 3.5
    /// engines, the caller owns the position and advances it afterwards.
    pub fn forward_pass(
        &self,
        token_indices: &MlxArray,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let final_hidden_states = self.forward_hidden_states(
            token_indices,
            starting_position_tokens,
            request_decoder_state,
        )?;
        self.logits_for_last_hidden_state(&final_hidden_states)
    }

    /// Runs every input position but retains only the final hidden row: earlier
    /// prompt rows advance decoder state, while only the terminal row is sampled.
    pub fn forward_hidden_states(
        &self,
        token_indices: &MlxArray,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        self.forward_hidden_states_recording_boundary_states(
            token_indices,
            starting_position_tokens,
            request_decoder_state,
            None,
        )
    }

    /// The same forward, which also keeps every gated-delta layer's state at
    /// the `astro_ssd_prompt_cache` block boundaries inside the chunk when a
    /// collector is given; without one it is exactly `forward_hidden_states`.
    pub fn forward_hidden_states_recording_boundary_states(
        &self,
        token_indices: &MlxArray,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let text = self.pack.config().text();
        // The core route carries the leading batch axis: token indices are
        // [1, token_count] and every activation stays [1, token_count, ...].
        let token_count = token_indices.shape()[1];
        let rms_norm_epsilon = f32::from_bits(text.rms_norm_epsilon_bits);
        let hidden_states = self.forward_decoder_layers(
            token_indices,
            starting_position_tokens,
            request_decoder_state,
            boundary_checkpoint_collector,
        )?;

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

    /// Embeds `token_indices` and runs every decoder layer over them, returning
    /// the last layer's output before the final normalization, the same span
    /// the shared Qwen 3.5 forward graph asks a model's decoder-layer loop for.
    fn forward_decoder_layers(
        &self,
        token_indices: &MlxArray,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        mut boundary_checkpoint_collector: Option<
            &mut AstroSsdPromptCacheBoundaryCheckpointCollector,
        >,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let rope_offset_tokens = rope_offset_tokens(starting_position_tokens)?;
        let mut hidden_states = self
            .embed_tokens(token_indices)
            .map_err(PrismHadamardModelError::Forward)?;
        for layer_index in 0..self.weights.decoder_layers.len() {
            hidden_states = self.run_decoder_layer(
                layer_index,
                &hidden_states,
                rope_offset_tokens,
                request_decoder_state,
                boundary_checkpoint_collector.as_deref_mut(),
            )?;
        }
        Ok(hidden_states)
    }

    /// Runs one decoder layer (attention half, dense MLP, both residual adds)
    /// over `hidden_states` of shape `[1, token_count, hidden]`, advancing the
    /// layer's own entry in `request_decoder_state`. The embedding and final
    /// norm stay outside so the measurement surface can time one layer in
    /// isolation.
    pub fn forward_decoder_layer(
        &self,
        layer_index: usize,
        hidden_states: &MlxArray,
        rope_offset_tokens: i32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        self.run_decoder_layer(
            layer_index,
            hidden_states,
            rope_offset_tokens,
            request_decoder_state,
            None,
        )
    }

    /// One decoder layer, with the gated-delta layers handing their boundary
    /// states to the collector when one is given.
    fn run_decoder_layer(
        &self,
        layer_index: usize,
        hidden_states: &MlxArray,
        rope_offset_tokens: i32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let text = self.pack.config().text();
        let token_count = hidden_states.shape()[1];
        let rms_norm_epsilon = f32::from_bits(text.rms_norm_epsilon_bits);
        let decoder_layer = &self.weights.decoder_layers[layer_index];
        let normalized_input = self
            .runtime
            .rms_norm(
                hidden_states,
                &self.weights.input_normalization_weights[layer_index],
                rms_norm_epsilon,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let attention_output = match (
            &decoder_layer.attention,
            request_decoder_state.layer_mut(layer_index),
        ) {
            (
                DecoderLayerAttentionWeights::Linear(linear_layer),
                Some(AstroDecoderStateCacheState::Composite {
                    convolution,
                    recurrent,
                }),
            ) => self.forward_linear_attention_layer(
                &normalized_input,
                token_count,
                linear_layer,
                LinearLayerState {
                    convolution_state: convolution,
                    recurrent_state: recurrent,
                    boundary_checkpoints: boundary_checkpoint_collector.map(|collector| {
                        LayerBoundaryCheckpoints {
                            collector,
                            decoder_layer_index: layer_index,
                        }
                    }),
                },
            )?,
            (
                DecoderLayerAttentionWeights::Full(full_layer),
                Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }),
            ) => self.forward_full_attention_layer(
                &normalized_input,
                token_count,
                full_layer,
                rms_norm_epsilon,
                rope_offset_tokens,
                attention,
            )?,
            _ => return Err(PrismHadamardModelError::DecoderStateLayerMismatch { layer_index }),
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
    pub fn logits_for_last_hidden_state(
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
    pub fn embed_tokens(
        &self,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, mlx_bridge::MlxCError> {
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
