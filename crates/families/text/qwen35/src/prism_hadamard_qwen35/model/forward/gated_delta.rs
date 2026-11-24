//! The gated-delta (linear-attention) decoder layer of the straight forward.
//!
//! The normalized layer input is rotated once and feeds one fused packed
//! matmul whose output splits into the mixed query/key/value projection and
//! the output-gate projection; the two float rate projections run as one
//! fused matmul too. The arithmetic mirrors the qwen3_5 core's validated GDN
//! composition exactly.

use mlx_bridge::MlxArray;

use crate::qwen3_5_core::model_math::gated_delta_boundary_checkpoints::qwen3_5_gated_delta_sequence_with_boundary_checkpoints;
use crate::qwen3_5_core::model_math::gated_delta_sequence::qwen3_5_gated_delta_sequence;

use crate::{
    ConvolutionState, GatedDeltaRecurrentState,
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector,
};

/// One gated-delta layer's live state for a forward, and, when the prompt
/// chunk crosses an `astro_ssd_prompt_cache` block boundary before its end,
/// the collector that keeps this layer's state at that boundary.
pub(in crate::prism_hadamard_qwen35::model) struct LinearLayerState<'a> {
    pub(in crate::prism_hadamard_qwen35::model) convolution_state: &'a mut ConvolutionState,
    pub(in crate::prism_hadamard_qwen35::model) recurrent_state: &'a mut GatedDeltaRecurrentState,
    pub(in crate::prism_hadamard_qwen35::model) boundary_checkpoints:
        Option<LayerBoundaryCheckpoints<'a>>,
}

/// Where one layer records its boundary states.
pub(in crate::prism_hadamard_qwen35::model) struct LayerBoundaryCheckpoints<'a> {
    pub(in crate::prism_hadamard_qwen35::model) collector:
        &'a mut Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector,
    pub(in crate::prism_hadamard_qwen35::model) decoder_layer_index: usize,
}

/// The two rate activations every gated-delta recurrence consumes.
struct GatedDeltaRates<'a> {
    update_rates: &'a MlxArray,
    decays: &'a MlxArray,
}

use super::super::PrismHadamardModelError;
use super::super::weights::LinearLayerWeights;

impl super::super::PrismHadamardTextModel {
    /// Runs one linear-attention layer through the selected shared gated-delta
    /// route, advancing this layer's convolution and recurrent state.
    pub(super) fn forward_linear_attention_layer(
        &self,
        normalized_input: &MlxArray,
        token_count: i32,
        layer: &LinearLayerWeights,
        linear_layer_state: LinearLayerState<'_>,
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

        let gated_delta_rates = GatedDeltaRates {
            update_rates: &update_rates,
            decays: &decays,
        };
        let recurrent_output = if linear_layer_state.boundary_checkpoints.is_some() {
            self.recur_recording_boundary_states(
                &mixed_queries_keys_values,
                token_count,
                layer,
                linear_layer_state,
                gated_delta_rates,
            )?
        } else {
            self.recur_over_chunk(
                &mixed_queries_keys_values,
                token_count,
                layer,
                linear_layer_state,
                gated_delta_rates,
            )?
        };

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

    /// The convolution window and gated-delta recurrence over a chunk, the
    /// route every forward takes unless the chunk holds a block boundary
    /// before its end. Returns the recurrence output and moves both states on.
    fn recur_over_chunk(
        &self,
        mixed_queries_keys_values: &MlxArray,
        token_count: i32,
        layer: &LinearLayerWeights,
        linear_layer_state: LinearLayerState<'_>,
        gated_delta_rates: GatedDeltaRates<'_>,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let convolution_input = linear_layer_state
            .convolution_state
            .update(&self.runtime, mixed_queries_keys_values, token_count)
            .map_err(PrismHadamardModelError::ForwardRuntime)?;
        let head_queries_keys_values =
            self.convolution_to_normalized_heads(&convolution_input, token_count, layer)?;
        let current_recurrent_state = linear_layer_state
            .recurrent_state
            .current_or_zero(&self.runtime)
            .map_err(PrismHadamardModelError::ForwardRuntime)?;
        let (recurrent_output, next_recurrent_state) = qwen3_5_gated_delta_sequence(
            &self.runtime,
            self.gated_delta_kernel.as_ref(),
            &head_queries_keys_values.0,
            &head_queries_keys_values.1,
            &head_queries_keys_values.2,
            gated_delta_rates.decays,
            gated_delta_rates.update_rates,
            &current_recurrent_state,
        )
        .map_err(PrismHadamardModelError::ForwardRuntime)?;
        linear_layer_state
            .recurrent_state
            .set_next(next_recurrent_state);
        Ok(recurrent_output)
    }

    /// The same convolution and recurrence through the shared boundary
    /// routes, which also return this layer's convolution and recurrent state
    /// at every block boundary inside the chunk, as the Qwen 3.5 layers do;
    /// the collector keeps them for the blocks the chunk completes.
    fn recur_recording_boundary_states(
        &self,
        mixed_queries_keys_values: &MlxArray,
        token_count: i32,
        layer: &LinearLayerWeights,
        linear_layer_state: LinearLayerState<'_>,
        gated_delta_rates: GatedDeltaRates<'_>,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let LinearLayerState {
            convolution_state,
            recurrent_state,
            boundary_checkpoints,
        } = linear_layer_state;
        let LayerBoundaryCheckpoints {
            collector,
            decoder_layer_index,
        } = boundary_checkpoints.ok_or_else(missing_boundary_collector)?;
        let convolution_update = convolution_state
            .update_with_boundary_checkpoints(
                &self.runtime,
                mixed_queries_keys_values,
                token_count,
                collector.completed_prefill_chunk_tokens(),
            )
            .map_err(PrismHadamardModelError::ForwardRuntime)?;
        let head_queries_keys_values = self.convolution_to_normalized_heads(
            &convolution_update.convolution_input,
            token_count,
            layer,
        )?;
        let current_recurrent_state = recurrent_state
            .current_or_zero(&self.runtime)
            .map_err(PrismHadamardModelError::ForwardRuntime)?;
        let checkpointed_recurrence = qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
            &self.runtime,
            self.gated_delta_checkpoint_kernel.as_ref(),
            &head_queries_keys_values.0,
            &head_queries_keys_values.1,
            &head_queries_keys_values.2,
            gated_delta_rates.decays,
            gated_delta_rates.update_rates,
            &current_recurrent_state,
            collector.completed_prefill_chunk_tokens(),
            collector.checkpoint_interval_token_count(),
        )
        .map_err(PrismHadamardModelError::ForwardRuntime)?;
        recurrent_state.set_next(checkpointed_recurrence.next_recurrent_state);
        collector
            .record_linear_attention_layer(
                decoder_layer_index,
                convolution_update.boundary_convolution_states,
                checkpointed_recurrence.recurrent_boundary_states,
            )
            .map_err(PrismHadamardModelError::ForwardRuntime)?;
        Ok(checkpointed_recurrence.sequence_outputs)
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

fn missing_boundary_collector() -> PrismHadamardModelError {
    PrismHadamardModelError::ForwardRuntime(
        runtime_integration::MlxRuntimeError::RuntimeOperation {
            operation: "record folded-ternary astro-ssd-prompt-cache boundary states",
            description: "the boundary route ran without its collector".to_owned(),
        },
    )
}
