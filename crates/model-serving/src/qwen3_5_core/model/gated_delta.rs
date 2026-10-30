use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::MlxArray;

use super::base::Qwen3_5ModelBase;
use crate::decoder_cache::{ConvolutionState, GatedDeltaRecurrentState};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};
use crate::qwen3_5_core::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5LinearAttentionWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::gated_delta_boundary_checkpoints;
use crate::qwen3_5_core::model_math::gated_delta_sequence;
use crate::qwen3_5_core::model_math::gated_delta_step::{
    evaluate_linear_attention_section, gated_delta_error,
};
use crate::qwen3_5_core::model_math::gdn_decode_prework_kernel::{
    is_gdn_decode_prework_eligible, qwen3_5_gdn_decode_prework,
};
use crate::qwen3_5_core::model_math::tensor_slicing;

#[cfg(test)]
mod tests;

impl Qwen3_5ModelBase {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_linear_attention(
        &self,
        hidden_states: &MlxArray,
        token_count: i32,
        decoder_layer_index: usize,
        linear_attention_weights: &Qwen3_5LinearAttentionWeights,
        convolution_state: &mut ConvolutionState,
        recurrent_state: &mut GatedDeltaRecurrentState,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let linear_value_dimension = self.config.linear_value_dimension() as i32;
        let (mixed_queries_keys_values, output_gate, update_logits, decay_inputs) = self
            .project_linear_attention_inputs(
                hidden_states,
                linear_attention_weights,
                token_count,
                performance_attribution,
            )?;
        let completed_prefill_chunk_tokens = boundary_checkpoint_collector
            .as_ref()
            .map(|collector| collector.completed_prefill_chunk_tokens().to_vec());
        let (queries, keys, values, boundary_convolution_states) = self.run_convolution_phase(
            convolution_state,
            &mixed_queries_keys_values,
            token_count,
            linear_attention_weights,
            completed_prefill_chunk_tokens.as_deref(),
            performance_attribution,
        )?;
        let update_rates = self.runtime.sigmoid(&update_logits)?;
        // The compiled decay graph is shapeless and serves every token count:
        // the historical one-token hand-composed branch paid seven kernel
        // launches per layer per decode step for the same formula.
        let decays = self.runtime.apply_compiled_gated_delta_decay(
            &self.compiled_elementwise_graphs,
            &linear_attention_weights.decay_rate_logarithm,
            &decay_inputs,
            &linear_attention_weights.decay_interval_bias,
        )?;
        evaluate_linear_attention_section(
            &self.runtime,
            performance_attribution,
            PerformanceOperation::PrefillLinearAttentionNormalizationGraphicsProcessorCompletionWait,
            &[&queries, &keys, &values, &decays, &update_rates],
            token_count,
        )?;
        let current_recurrent_state = recurrent_state.current_or_zero(&self.runtime)?;
        // Each dispatch entry owns its capability routing: a retained kernel
        // takes the fused Metal route; a demoted kernel falls back to the
        // ops-based public MLX route inside the dispatch, and the checkpoint
        // fallback preserves the prompt-cache boundary snapshot positions.
        let (recurrent_output, next_recurrent_state, boundary_recurrent_states) = self
            .run_recurrence_phase(
                boundary_checkpoint_collector.as_deref(),
                &queries,
                &keys,
                &values,
                &decays,
                &update_rates,
                &current_recurrent_state,
                completed_prefill_chunk_tokens.as_deref(),
            )?;
        evaluate_linear_attention_section(
            &self.runtime,
            performance_attribution,
            PerformanceOperation::PrefillLinearAttentionRecurrenceGraphicsProcessorCompletionWait,
            &[&recurrent_output, &next_recurrent_state],
            token_count,
        )?;
        if let Some(boundary_checkpoint_collector) = boundary_checkpoint_collector {
            boundary_checkpoint_collector.record_linear_attention_layer(
                decoder_layer_index,
                boundary_convolution_states,
                boundary_recurrent_states,
            )?;
        }
        let normalized_output = self.runtime.rms_norm(
            &recurrent_output,
            &linear_attention_weights.normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let gated_output = self.runtime.apply_compiled_precise_swiglu(
            &self.compiled_elementwise_graphs,
            &normalized_output,
            &output_gate,
        )?;
        let gated_output = self
            .runtime
            .reshape(&gated_output, &[1, token_count, linear_value_dimension])?;
        recurrent_state.set_next(next_recurrent_state);
        let projected_output =
            self.quantized_linear(&gated_output, &linear_attention_weights.output_projection)?;
        evaluate_linear_attention_section(
            &self.runtime,
            performance_attribution,
            PerformanceOperation::PrefillLinearAttentionEpilogueGraphicsProcessorCompletionWait,
            &[&projected_output],
            token_count,
        )?;
        Ok(projected_output)
    }

    /// Runs the four input projections under one evaluation section and
    /// reshapes the output gate to per-head layout.
    fn project_linear_attention_inputs(
        &self,
        hidden_states: &MlxArray,
        linear_attention_weights: &Qwen3_5LinearAttentionWeights,
        token_count: i32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, MlxArray, MlxArray, MlxArray), Qwen3_5ExecutionError> {
        let linear_value_head_count = self.config.linear_value_head_count() as i32;
        let linear_head_dimension = self.config.linear_key_head_dimension() as i32;
        let mixed_queries_keys_values = self.quantized_linear(
            hidden_states,
            &linear_attention_weights.input_queries_keys_values_projection,
        )?;
        let output_gate = self.quantized_linear(
            hidden_states,
            &linear_attention_weights.output_gate_projection,
        )?;
        let output_gate = self.runtime.reshape(
            &output_gate,
            &[
                1,
                token_count,
                linear_value_head_count,
                linear_head_dimension,
            ],
        )?;
        let update_logits = self.quantized_linear(
            hidden_states,
            &linear_attention_weights.update_rate_projection,
        )?;
        let decay_inputs = self.quantized_linear(
            hidden_states,
            &linear_attention_weights.decay_interval_projection,
        )?;
        evaluate_linear_attention_section(
            &self.runtime,
            performance_attribution,
            PerformanceOperation::PrefillLinearAttentionProjectionsGraphicsProcessorCompletionWait,
            &[
                &mixed_queries_keys_values,
                &output_gate,
                &update_logits,
                &decay_inputs,
            ],
            token_count,
        )?;
        Ok((
            mixed_queries_keys_values,
            output_gate,
            update_logits,
            decay_inputs,
        ))
    }

    /// Runs the convolution phase: the checkpoint path updates the rolling
    /// buffer with boundary snapshots; launch-bound decode steps take the
    /// fused prework kernel; everything else takes the composed fallback
    /// that must stay bit-identical to the fused kernel.
    #[allow(clippy::too_many_arguments)]
    fn run_convolution_phase(
        &self,
        convolution_state: &mut ConvolutionState,
        mixed_queries_keys_values: &MlxArray,
        token_count: i32,
        linear_attention_weights: &Qwen3_5LinearAttentionWeights,
        completed_prefill_chunk_tokens: Option<&[i32]>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, MlxArray, MlxArray, Vec<MlxArray>), Qwen3_5ExecutionError> {
        let linear_key_head_count = self.config.linear_key_head_count() as i32;
        let linear_value_head_count = self.config.linear_value_head_count() as i32;
        let linear_head_dimension = self.config.linear_key_head_dimension() as i32;
        let convolution_weight = normalize_linear_attention_convolution_weight(
            &self.runtime,
            &linear_attention_weights.convolution_weight,
            mixed_queries_keys_values.shape()[2],
            i32::try_from(self.config.linear_convolution_kernel_dimension()).map_err(|_| {
                Qwen3_5ExecutionError::InvalidInput {
                    description: "linear convolution kernel width exceeds the MLX shape range",
                }
            })?,
        )?;
        match completed_prefill_chunk_tokens {
            Some(completed_prefill_chunk_tokens) => {
                // The pre-window rolling buffer feeds the checkpoint
                // update, which replaces the state itself.
                let checkpoint_update = convolution_state.update_with_boundary_checkpoints(
                    &self.runtime,
                    mixed_queries_keys_values,
                    token_count,
                    completed_prefill_chunk_tokens,
                )?;
                let (queries, keys, values) = self.convolution_to_normalized_heads(
                    &checkpoint_update.convolution_input,
                    token_count,
                    &convolution_weight,
                    performance_attribution,
                )?;
                Ok((
                    queries,
                    keys,
                    values,
                    checkpoint_update.boundary_convolution_states,
                ))
            }
            None => {
                // Launch-bound decode steps take the fused prework kernel:
                // one Metal launch for the rolling window, conv1d, SiLU,
                // q/k/v split, both RMS norms, both scales, and the next
                // rolling state.
                if is_gdn_decode_prework_eligible(
                    self.gdn_decode_prework_kernel.as_ref(),
                    token_count,
                    mixed_queries_keys_values.dtype(),
                    linear_head_dimension,
                ) {
                    let convolution_state_array = convolution_state
                        .current_or_zero(&self.runtime, mixed_queries_keys_values.dtype())?;
                    let fused = qwen3_5_gdn_decode_prework(
                        &self.runtime,
                        self.gdn_decode_prework_kernel
                            .as_ref()
                            .expect("prework eligibility just proved the kernel is retained"),
                        linear_key_head_count,
                        linear_value_head_count,
                        linear_head_dimension,
                        mixed_queries_keys_values,
                        &convolution_state_array,
                        &convolution_weight,
                        &self.inverse_linear_head_dimension_scale,
                        &self.inverse_square_root_linear_head_dimension_scale,
                    )?;
                    convolution_state.replace_state(fused.next_convolution_state)?;
                    Ok((fused.queries, fused.keys, fused.values, Vec::new()))
                } else {
                    let convolution_input = convolution_state.update(
                        &self.runtime,
                        mixed_queries_keys_values,
                        token_count,
                    )?;
                    let (queries, keys, values) = self.convolution_to_normalized_heads(
                        &convolution_input,
                        token_count,
                        &convolution_weight,
                        performance_attribution,
                    )?;
                    Ok((queries, keys, values, Vec::new()))
                }
            }
        }
    }

    /// Runs the gated-delta recurrence: the checkpoint path snapshots
    /// recurrent states at their intervals; the plain path runs the
    /// sequence with capability-routed kernels.
    #[allow(clippy::too_many_arguments)]
    fn run_recurrence_phase(
        &self,
        boundary_checkpoint_collector: Option<
            &Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        decays: &MlxArray,
        update_rates: &MlxArray,
        current_recurrent_state: &MlxArray,
        completed_prefill_chunk_tokens: Option<&[i32]>,
    ) -> Result<(MlxArray, MlxArray, Vec<MlxArray>), Qwen3_5ExecutionError> {
        match completed_prefill_chunk_tokens {
            Some(completed_prefill_chunk_tokens) => {
                let checkpoint_interval_token_count = boundary_checkpoint_collector
                    .map(|collector| collector.checkpoint_interval_token_count())
                    .ok_or_else(|| {
                        gated_delta_error("gated-delta checkpoint collector disappeared")
                    })?;
                let checkpoint_result = gated_delta_boundary_checkpoints::qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
                    &self.runtime,
                    self.gated_delta_checkpoint_kernel.as_ref(),
                    queries,
                    keys,
                    values,
                    decays,
                    update_rates,
                    current_recurrent_state,
                    completed_prefill_chunk_tokens,
                    checkpoint_interval_token_count,
                )?;
                Ok((
                    checkpoint_result.sequence_outputs,
                    checkpoint_result.next_recurrent_state,
                    checkpoint_result.recurrent_boundary_states,
                ))
            }
            None => {
                let (recurrent_output, next_recurrent_state) =
                    gated_delta_sequence::qwen3_5_gated_delta_sequence(
                        &self.runtime,
                        self.gated_delta_kernel.as_ref(),
                        queries,
                        keys,
                        values,
                        decays,
                        update_rates,
                        current_recurrent_state,
                    )?;
                Ok((recurrent_output, next_recurrent_state, Vec::new()))
            }
        }
    }

    /// The composed prefill-path arithmetic the fused decode prework kernel
    /// replaces at launch-bound shapes: depthwise conv1d, SiLU, q/k/v split,
    /// both ones-weight RMS normalizations, and both scalar scales folded
    /// into the norms' per-channel weights. The fused prework engagement
    /// check and the bit-exactness contract in the direct-MLX numerics test
    /// keep this the fallback that must stay identical, never a divergent
    /// second formula.
    fn convolution_to_normalized_heads(
        &self,
        convolution_input: &MlxArray,
        token_count: i32,
        convolution_weight: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, MlxArray, MlxArray), Qwen3_5ExecutionError> {
        let linear_key_head_count = self.config.linear_key_head_count() as i32;
        let linear_value_head_count = self.config.linear_value_head_count() as i32;
        let linear_head_dimension = self.config.linear_key_head_dimension() as i32;
        let linear_key_dimension = self.config.linear_key_dimension() as i32;
        let linear_convolution_dimension = self.config.linear_convolution_dimension() as i32;
        let rms_norm_epsilon = f32::from_bits(self.config.rms_norm_epsilon_bits());
        let convolution_output = self.runtime.conv1d(
            convolution_input,
            convolution_weight,
            1,
            0,
            1,
            linear_convolution_dimension,
        )?;
        // The shapeless fused-silu compilation matches stock MLX `nn.silu`:
        // one kernel reading the input once, instead of a sigmoid kernel plus
        // a multiply kernel with roughly 2.5x the memory traffic.
        let convolution_output = self
            .runtime
            .apply_compiled_silu(&self.compiled_elementwise_graphs, &convolution_output)?;
        evaluate_linear_attention_section(
            &self.runtime,
            performance_attribution,
            PerformanceOperation::PrefillLinearAttentionConvolutionGraphicsProcessorCompletionWait,
            &[&convolution_output],
            token_count,
        )?;
        let queries = tensor_slicing::slice_last_dimension(
            &self.runtime,
            &convolution_output,
            0,
            linear_key_dimension,
        )?;
        let queries = self.runtime.reshape(
            &queries,
            &[1, token_count, linear_key_head_count, linear_head_dimension],
        )?;
        let keys = tensor_slicing::slice_last_dimension(
            &self.runtime,
            &convolution_output,
            linear_key_dimension,
            linear_key_dimension * 2,
        )?;
        let keys = self.runtime.reshape(
            &keys,
            &[1, token_count, linear_key_head_count, linear_head_dimension],
        )?;
        let values = tensor_slicing::slice_last_dimension(
            &self.runtime,
            &convolution_output,
            linear_key_dimension * 2,
            linear_convolution_dimension,
        )?;
        let values = self.runtime.reshape(
            &values,
            &[
                1,
                token_count,
                linear_value_head_count,
                linear_head_dimension,
            ],
        )?;
        // Each scale folds into `fast_rms_norm`'s per-channel weight: one
        // fused launch per tensor replaces the norm kernel plus the separate
        // broadcast-multiply kernel and drops the normalized intermediate
        // round-trip (issue #915 item 5). MLX's own reference computes
        // `round(normalize(x)) * weight`, identical to the former norm-then-
        // multiply pair; the direct-MLX numerics contract pins bit-for-bit
        // parity before this dispatch engages.
        let queries = self.runtime.rms_norm(
            &queries,
            &self.query_normalization_scale_weight,
            rms_norm_epsilon,
        )?;
        let keys = self.runtime.rms_norm(
            &keys,
            &self.key_normalization_scale_weight,
            rms_norm_epsilon,
        )?;
        Ok((queries, keys, values))
    }
}

fn normalize_linear_attention_convolution_weight(
    runtime: &MlxRuntime,
    convolution_weight: &MlxArray,
    output_channel_count: i32,
    convolution_kernel_width: i32,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let execution_shape = [output_channel_count, convolution_kernel_width, 1];
    let published_pytorch_shape = [output_channel_count, 1, convolution_kernel_width];
    let actual_weight_shape = convolution_weight.shape();
    if actual_weight_shape == execution_shape {
        return Ok(convolution_weight.retain()?);
    }
    if actual_weight_shape == published_pytorch_shape {
        // MLX stores Conv1d weights as [output, kernel, input]; the published
        // PyTorch layout is [output, input, kernel]. This axis permutation
        // preserves every value and restores the layout consumed by both the
        // standard and fused convolution paths.
        return Ok(runtime.transpose_axes(convolution_weight, &[0, 2, 1])?);
    }
    Err(Qwen3_5ExecutionError::InvalidTensor {
        tensor_name: "linear_attention.conv1d.weight".to_owned(),
        description: "convolution weight shape is neither the MLX nor published PyTorch layout",
    })
}
