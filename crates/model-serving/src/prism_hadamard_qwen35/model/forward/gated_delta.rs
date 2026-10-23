//! The gated-delta (linear-attention) decoder layer of the straight forward.
//!
//! The normalized layer input is rotated once and feeds one fused packed
//! matmul whose output splits into the mixed query/key/value projection and
//! the output-gate projection; the two float rate projections run as one
//! fused matmul too. The arithmetic mirrors the qwen3_5 core's validated GDN
//! composition exactly.

use mlx_c_rust::MlxArray;

use crate::qwen3_5_core::model_math::gated_delta_sequence::qwen3_5_gated_delta_sequence;

use super::super::PrismHadamardModelError;
use super::super::weights::LinearLayerWeights;
use super::StraightGenerationState;

impl super::super::PrismHadamardTextModel {
    /// Runs one linear-attention layer through the selected shared gated-delta route.
    pub(super) fn forward_linear_attention_layer(
        &self,
        normalized_input: &MlxArray,
        token_count: i32,
        layer: &LinearLayerWeights,
        state_slot: usize,
        state: &mut StraightGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let text = self.pack.config().text();
        let rms_norm_epsilon = f32::from_bits(text.rms_norm_epsilon_bits);
        let linear_value_head_count = text.linear_num_value_heads as i32;
        let linear_value_dimension =
            (text.linear_num_value_heads * text.linear_value_head_dim) as i32;

        // One rotation of the layer input feeds the one fused launch that
        // produces both rotated projections.
        let rotated_input = self
            .rotate(normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let [mixed_queries_keys_values, output_gate] = layer
            .query_key_value_and_gate
            .forward(&self.runtime, &rotated_input)
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
        let rate_logits = layer
            .rates
            .forward(&self.runtime, normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        // One dual-output replay replaces the separate sigmoid and decay
        // launches; both consume this same rate projection (issue #1366).
        let (update_rates, decays) = self
            .runtime
            .apply_compiled_gated_delta_rate_activation(
                &self.compiled_gated_delta_rate_activation,
                &rate_logits.update_logits,
                &rate_logits.decay_inputs,
                &layer.decay_rate_logarithm,
                &layer.decay_interval_bias,
            )
            .map_err(PrismHadamardModelError::Forward)?;

        let convolution_input = state.convolution_states[state_slot]
            .update(&self.runtime, &mixed_queries_keys_values, token_count)
            .map_err(PrismHadamardModelError::ForwardRuntime)?;
        let head_queries_keys_values =
            self.convolution_to_normalized_heads(&convolution_input, token_count, layer)?;
        let recurrent_state = &mut state.recurrent_states[state_slot];
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
        let rotated_gated_output = self
            .rotate(&gated_output)
            .map_err(PrismHadamardModelError::Forward)?;
        layer
            .output
            .forward(&self.runtime, &rotated_gated_output)
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

        let queries = super::super::weights::slice_last_dimension(
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
        let keys = super::super::weights::slice_last_dimension(
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
        let values = super::super::weights::slice_last_dimension(
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

        // One dual-output replay replaces the two separate head-norm
        // launches (issue #1366); each output is the same fast-rms kernel
        // over its own input.
        let (queries, keys) = self
            .runtime
            .apply_compiled_dual_head_norm(
                &self.compiled_dual_head_norm,
                &queries,
                &keys,
                &self.gdn_query_scale_weight,
                &self.gdn_key_scale_weight,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        Ok((queries, keys, values))
    }
}
