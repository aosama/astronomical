//! The decoder layers over several sequences at once, each starting empty:
//! the forward pass a decision model's whole-model training differentiates.
//!
//! Serving reads one sequence at a time through the decoder state (rolling
//! convolution buffers, recurrent state, the key-value store), which is
//! built for one row. A training's sequences all start at their first token
//! with nothing before them, so they need none of that: the convolution
//! sees zeros before the sequence, the recurrence starts from a zero state,
//! and attention is causal over the sequence itself. That lets many
//! sequences share one pass. Run one at a time, a 0.8B Kev question cost
//! 133 ms forwards and backwards, nearly all of it fixed per-kernel cost;
//! sharing the pass spreads that cost over every row.
//!
//! Rows shorter than the longest are padded at the end. Every operation here
//! is causal, so padding never changes a real position's result.
//!
//! The layers are serving's own pieces: the same projections, convolution,
//! normalizations, recurrence kernel, gates and feed-forward. Scores a
//! training compares with serving never come from here; they run serving's
//! path.

use mlx_bridge::MlxArray;

use super::Qwen3_5ModelBase;
use crate::performance_attribution::PerformanceAttribution;
use crate::qwen3_5_core::model::full_attention::RotatedAttentionInputs;
use crate::qwen3_5_core::model::gated_delta::normalize_linear_attention_convolution_weight;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights, Qwen3_5DecoderLayerWeights,
    Qwen3_5FullAttentionWeights, Qwen3_5LinearAttentionWeights,
};
use crate::qwen3_5_core::model_math::gated_delta_inputs::GatedDeltaInputs;
use crate::qwen3_5_core::model_math::gated_delta_sequence::qwen3_5_gated_delta_sequence;
use text_generation::execution_error::TextExecutionError;

impl Qwen3_5ModelBase {
    /// One decoder layer over `[rows, token_count, hidden]` hidden states.
    pub fn decoder_layer_from_empty_state(
        &self,
        hidden_states: &MlxArray,
        token_count: i32,
        layer_index: usize,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
    ) -> Result<MlxArray, TextExecutionError> {
        let rms_norm_epsilon = f32::from_bits(self.config.rms_norm_epsilon_bits());
        let normalized_input = self.runtime.rms_norm(
            hidden_states,
            &decoder_layer_weights.input_normalization_weight,
            rms_norm_epsilon,
        )?;
        let attention_output = match &decoder_layer_weights.attention_weights {
            Qwen3_5AttentionWeights::Linear(linear_attention_weights) => self
                .linear_attention_from_empty_state(
                    &normalized_input,
                    token_count,
                    layer_index,
                    linear_attention_weights,
                )?,
            Qwen3_5AttentionWeights::Full(full_attention_weights) => self
                .full_attention_from_empty_state(
                    &normalized_input,
                    token_count,
                    full_attention_weights,
                )?,
        };
        let attention_residual = self.runtime.add(hidden_states, &attention_output)?;
        let normalized_attention = self.runtime.rms_norm(
            &attention_residual,
            &decoder_layer_weights.post_attention_normalization_weight,
            rms_norm_epsilon,
        )?;
        let Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) =
            &decoder_layer_weights.mlp_weights
        else {
            return Err(TextExecutionError::InvalidInput {
                description: "a mixture-of-experts layer has no empty-state forward",
            });
        };
        let mlp_output =
            self.forward_qwen3_5_dense_mlp(&normalized_attention, dense_mlp_weights)?;
        Ok(self.runtime.add(&attention_residual, &mlp_output)?)
    }

    /// A Gated DeltaNet layer from an empty state: zeros before the
    /// convolution, a zero recurrent state.
    fn linear_attention_from_empty_state(
        &self,
        normalized_input: &MlxArray,
        token_count: i32,
        layer_index: usize,
        linear_attention_weights: &Qwen3_5LinearAttentionWeights,
    ) -> Result<MlxArray, TextExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        let (mixed_queries_keys_values, output_gate, update_logits, decay_inputs) = self
            .project_linear_attention_inputs(
                normalized_input,
                linear_attention_weights,
                token_count,
                layer_index,
                &mut disabled_performance_attribution,
            )?;
        let row_count = mixed_queries_keys_values.shape()[0];
        let channel_count = mixed_queries_keys_values.shape()[2];
        let kernel_width = self.config.linear_convolution_kernel_dimension() as i32;
        let convolution_weight = normalize_linear_attention_convolution_weight(
            &self.runtime,
            &linear_attention_weights.convolution_weight,
            channel_count,
            kernel_width,
        )?;
        let empty_window = self.runtime.zeros(
            &[row_count, kernel_width - 1, channel_count],
            mixed_queries_keys_values.dtype(),
        )?;
        let convolution_input = self
            .runtime
            .concatenate_axis(&[&empty_window, &mixed_queries_keys_values], 1)?;
        let (queries, keys, values) = self.convolution_to_normalized_heads(
            &convolution_input,
            token_count,
            &convolution_weight,
            &mut disabled_performance_attribution,
        )?;
        let update_rates = self.runtime.sigmoid(&update_logits)?;
        let decays = self.runtime.apply_compiled_gated_delta_decay(
            &self.compiled_elementwise_graphs,
            &linear_attention_weights.decay_rate_logarithm,
            &decay_inputs,
            &linear_attention_weights.decay_interval_bias,
        )?;
        let empty_recurrent_state = self.runtime.zeros(
            &[
                row_count,
                self.config.linear_value_head_count() as i32,
                self.config.linear_value_head_dimension() as i32,
                self.config.linear_key_head_dimension() as i32,
            ],
            mlx_bridge::MlxDtype::Float32,
        )?;
        // A training building its graph puts serving's recurrence with a
        // backward pass in force; otherwise serving's own kernel runs.
        let layer_additions = self.layer_additions_in_force();
        let gated_delta_inputs = GatedDeltaInputs {
            queries: &queries,
            keys: &keys,
            values: &values,
            decays: &decays,
            update_rates: &update_rates,
            recurrent_state: &empty_recurrent_state,
        };
        let recurrent_output = if let Some(layer_additions) = &layer_additions
            && let Some(differentiable_recurrence) = &layer_additions.differentiable_recurrence
        {
            differentiable_recurrence.sequence(&gated_delta_inputs)?.0
        } else {
            qwen3_5_gated_delta_sequence(
                &self.runtime,
                self.gated_delta_kernel.as_ref(),
                &gated_delta_inputs,
            )?
            .0
        };
        self.gated_linear_attention_output(
            &recurrent_output,
            &output_gate,
            token_count,
            linear_attention_weights,
        )
    }

    /// A full-attention layer over the sequences themselves: causal
    /// attention with nothing stored before them.
    fn full_attention_from_empty_state(
        &self,
        normalized_input: &MlxArray,
        token_count: i32,
        full_attention_weights: &Qwen3_5FullAttentionWeights,
    ) -> Result<MlxArray, TextExecutionError> {
        let RotatedAttentionInputs {
            rotated_queries,
            rotated_keys,
            transposed_values,
            output_gate,
        } = self.rotated_attention_inputs(
            normalized_input,
            token_count,
            0,
            full_attention_weights,
            None,
        )?;
        let attention_head_dimension = self.config.head_dimension() as i32;
        self.run_full_attention_computation(
            &rotated_queries,
            &rotated_keys,
            &transposed_values,
            &output_gate,
            (attention_head_dimension as f32).sqrt().recip(),
            &full_attention_weights.output_projection,
        )
    }
}
