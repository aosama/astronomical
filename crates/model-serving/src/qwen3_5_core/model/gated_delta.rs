use astronomical_mlx_c_rust::MlxArray;

use super::base::Qwen3_5ModelBase;
use crate::decoder_cache::{ConvolutionState, GatedDeltaRecurrentState};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};
use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;
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
        mut boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let linear_key_head_count = self.config.linear_key_head_count() as i32;
        let linear_value_head_count = self.config.linear_value_head_count() as i32;
        let linear_head_dimension = self.config.linear_key_head_dimension() as i32;
        let linear_value_dimension = self.config.linear_value_dimension() as i32;
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
        let completed_prefill_chunk_tokens = boundary_checkpoint_collector
            .as_ref()
            .map(|collector| collector.completed_prefill_chunk_tokens().to_vec());
        let (queries, keys, values, boundary_convolution_states) =
            match completed_prefill_chunk_tokens.as_deref() {
                Some(completed_prefill_chunk_tokens) => {
                    // The pre-window rolling buffer feeds the checkpoint
                    // update, which replaces the state itself.
                    let checkpoint_update = convolution_state.update_with_boundary_checkpoints(
                        &self.runtime,
                        &mixed_queries_keys_values,
                        token_count,
                        completed_prefill_chunk_tokens,
                    )?;
                    let (queries, keys, values) = self.convolution_to_normalized_heads(
                        &checkpoint_update.convolution_input,
                        token_count,
                        linear_attention_weights,
                        performance_attribution,
                    )?;
                    (
                        queries,
                        keys,
                        values,
                        checkpoint_update.boundary_convolution_states,
                    )
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
                            &mixed_queries_keys_values,
                            &convolution_state_array,
                            &linear_attention_weights.convolution_weight,
                            &self.inverse_linear_head_dimension_scale,
                            &self.inverse_square_root_linear_head_dimension_scale,
                        )?;
                        convolution_state.replace_state(fused.next_convolution_state)?;
                        (fused.queries, fused.keys, fused.values, Vec::new())
                    } else {
                        let convolution_input = convolution_state.update(
                            &self.runtime,
                            &mixed_queries_keys_values,
                            token_count,
                        )?;
                        let (queries, keys, values) = self.convolution_to_normalized_heads(
                            &convolution_input,
                            token_count,
                            linear_attention_weights,
                            performance_attribution,
                        )?;
                        (queries, keys, values, Vec::new())
                    }
                }
            };
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
        let (recurrent_output, next_recurrent_state, boundary_recurrent_states) =
            match completed_prefill_chunk_tokens.as_deref() {
                Some(completed_prefill_chunk_tokens) => {
                    let checkpoint_interval_token_count = boundary_checkpoint_collector
                        .as_ref()
                        .map(|collector| collector.checkpoint_interval_token_count())
                        .ok_or_else(|| {
                            gated_delta_error("gated-delta checkpoint collector disappeared")
                        })?;
                    let checkpoint_result = gated_delta_boundary_checkpoints::qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
                        &self.runtime,
                        self.gated_delta_checkpoint_kernel.as_ref(),
                        &queries,
                        &keys,
                        &values,
                        &decays,
                        &update_rates,
                        &current_recurrent_state,
                        completed_prefill_chunk_tokens,
                        checkpoint_interval_token_count,
                    )?;
                    (
                        checkpoint_result.sequence_outputs,
                        checkpoint_result.next_recurrent_state,
                        checkpoint_result.recurrent_boundary_states,
                    )
                }
                None => {
                    let (recurrent_output, next_recurrent_state) =
                        gated_delta_sequence::qwen3_5_gated_delta_sequence(
                            &self.runtime,
                            self.gated_delta_kernel.as_ref(),
                            &queries,
                            &keys,
                            &values,
                            &decays,
                            &update_rates,
                            &current_recurrent_state,
                        )?;
                    (recurrent_output, next_recurrent_state, Vec::new())
                }
            };
        evaluate_linear_attention_section(
            &self.runtime,
            performance_attribution,
            PerformanceOperation::PrefillLinearAttentionRecurrenceGraphicsProcessorCompletionWait,
            &[&recurrent_output, &next_recurrent_state],
            token_count,
        )?;
        if let Some(boundary_checkpoint_collector) = boundary_checkpoint_collector.as_deref_mut() {
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
        linear_attention_weights: &Qwen3_5LinearAttentionWeights,
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
            &linear_attention_weights.convolution_weight,
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
