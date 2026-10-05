use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::Qwen3_5ExecutionError;
use super::decoder_layer_weights::Qwen3_5LinearAttentionWeights;
use super::gated_delta_boundary_checkpoints;
use super::gated_delta_sequence;
use super::gdn_decode_prework_kernel::{
    is_gdn_decode_prework_eligible, qwen3_5_gdn_decode_prework,
};
use super::model::Qwen3_5Model;
use super::tensor_slicing;
use crate::decoder_cache::{ConvolutionState, GatedDeltaRecurrentState};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};
use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;
use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};

const GATED_DELTA_STEP_OPERATION: &str = "apply one Qwen3.5 gated-delta recurrent step";

/// Forces one gated-delta section's graphics-processor work so its GPU time is
/// attributed to its own operation instead of folding into the family wait or
/// the chunk-terminal wait. Multi-token prefill only: one-token decode keeps
/// the latency-sensitive step free of host synchronization.
fn evaluate_linear_attention_section(
    runtime: &MlxRuntime,
    performance_attribution: &mut PerformanceAttribution,
    operation: PerformanceOperation,
    arrays: &[&MlxArray],
    token_count: i32,
    is_verification_window: bool,
) -> Result<(), Qwen3_5ExecutionError> {
    // The MTP verification window is a latency-sensitive decode pass: injected
    // evaluation boundaries serialize it like one-token decode, and its phase
    // costs are attributed at the attempt boundary instead.
    if !performance_attribution.is_enabled() || token_count <= 1 || is_verification_window {
        return Ok(());
    }
    performance_attribution.measure_operation(operation, |_performance_attribution| {
        runtime.evaluate_arrays(arrays)
    })?;
    Ok(())
}

/// Applies one ops-based Qwen3.5 gated-delta recurrence while retaining state in float32.
pub fn qwen3_5_gated_delta_step(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
    let repeat_factor =
        validate_gated_delta_shapes(queries, keys, values, decays, update_rates, recurrent_state)?;
    let output_dtype = queries.dtype();
    let float32_queries = runtime.astype(queries, MlxDtype::Float32)?;
    let float32_keys = runtime.astype(keys, MlxDtype::Float32)?;
    let float32_values = runtime.astype(values, MlxDtype::Float32)?;
    let float32_decays = runtime.astype(decays, MlxDtype::Float32)?;
    let float32_update_rates = runtime.astype(update_rates, MlxDtype::Float32)?;
    let repeated_queries = runtime.repeat_axis(&float32_queries, repeat_factor, 1)?;
    let repeated_keys = runtime.repeat_axis(&float32_keys, repeat_factor, 1)?;

    let decay_rows = runtime.expand_dims(&float32_decays, -1)?;
    let decay_matrices = runtime.expand_dims(&decay_rows, -1)?;
    let decayed_state = runtime.multiply(recurrent_state, &decay_matrices)?;
    let expanded_keys = runtime.expand_dims(&repeated_keys, 2)?;
    let state_key_products = runtime.multiply(&decayed_state, &expanded_keys)?;
    let remembered_values = runtime.sum_axis(&state_key_products, -1, false)?;
    let value_differences = runtime.subtract(&float32_values, &remembered_values)?;
    let expanded_update_rates = runtime.expand_dims(&float32_update_rates, -1)?;
    let value_updates = runtime.multiply(&value_differences, &expanded_update_rates)?;
    let expanded_value_updates = runtime.expand_dims(&value_updates, -1)?;
    let state_updates = runtime.multiply(&expanded_keys, &expanded_value_updates)?;
    let next_recurrent_state = runtime.add(&decayed_state, &state_updates)?;

    let expanded_queries = runtime.expand_dims(&repeated_queries, 2)?;
    let state_query_products = runtime.multiply(&next_recurrent_state, &expanded_queries)?;
    let float32_output = runtime.sum_axis(&state_query_products, -1, false)?;
    let output = runtime.astype(&float32_output, output_dtype)?;
    Ok((output, next_recurrent_state))
}

fn validate_gated_delta_shapes(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
) -> Result<i32, MlxRuntimeError> {
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    let value_shape = values.shape();
    let decay_shape = decays.shape();
    let update_rate_shape = update_rates.shape();
    let recurrent_state_shape = recurrent_state.shape();
    if query_shape.len() != 3
        || value_shape.len() != 3
        || decay_shape.len() != 2
        || recurrent_state_shape.len() != 4
    {
        return Err(gated_delta_error(
            "queries, values, decays, and recurrent state must have ranks three, three, two, and four",
        ));
    }
    if key_shape != query_shape {
        return Err(gated_delta_error(
            "gated-delta queries and keys must have identical shapes",
        ));
    }
    if update_rate_shape != decay_shape {
        return Err(gated_delta_error(
            "gated-delta update rates and decays must have identical shapes",
        ));
    }
    let batch_size = query_shape[0];
    let key_head_count = query_shape[1];
    let key_dimension = query_shape[2];
    let value_head_count = value_shape[1];
    let value_dimension = value_shape[2];
    if batch_size <= 0
        || key_head_count <= 0
        || key_dimension <= 0
        || value_head_count <= 0
        || value_dimension <= 0
        || value_head_count % key_head_count != 0
    {
        return Err(gated_delta_error(
            "gated-delta dimensions must be positive and value heads must divide evenly by key heads",
        ));
    }
    if value_shape[0] != batch_size
        || decay_shape != [batch_size, value_head_count]
        || recurrent_state_shape != [batch_size, value_head_count, value_dimension, key_dimension]
    {
        return Err(gated_delta_error(
            "gated-delta value, decay, and recurrent-state dimensions are incompatible",
        ));
    }
    if recurrent_state.dtype() != MlxDtype::Float32 {
        return Err(gated_delta_error(
            "gated-delta recurrent state must use float32",
        ));
    }
    if !is_supported_activation_dtype(queries.dtype())
        || !is_supported_activation_dtype(keys.dtype())
        || !is_supported_activation_dtype(values.dtype())
        || !is_supported_activation_dtype(decays.dtype())
        || !is_supported_activation_dtype(update_rates.dtype())
    {
        return Err(gated_delta_error(
            "gated-delta inputs must use float16, bfloat16, or float32",
        ));
    }
    Ok(value_head_count / key_head_count)
}

fn is_supported_activation_dtype(dtype: MlxDtype) -> bool {
    matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
    )
}

fn gated_delta_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: GATED_DELTA_STEP_OPERATION,
        description: description.to_owned(),
    }
}

impl Qwen3_5Model {
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
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let linear_key_head_count = self.config.linear_key_head_count() as i32;
        let linear_value_head_count = self.config.linear_value_head_count() as i32;
        let linear_head_dimension = self.config.linear_key_head_dimension() as i32;
        let linear_value_dimension = self.config.linear_value_dimension() as i32;
        let is_verification_window = paged_prefill_execution_mode
            == crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow;
        let mixed_queries_keys_values = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &linear_attention_weights.input_queries_keys_values_projection,
            paged_prefill_execution_mode,
        )?;
        let output_gate = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &linear_attention_weights.output_gate_projection,
            paged_prefill_execution_mode,
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
        let update_logits = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &linear_attention_weights.update_rate_projection,
            paged_prefill_execution_mode,
        )?;
        let decay_inputs = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &linear_attention_weights.decay_interval_projection,
            paged_prefill_execution_mode,
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
            is_verification_window,
        )?;
        let completed_prefill_chunk_tokens = boundary_checkpoint_collector
            .as_ref()
            .map(|collector| collector.completed_prefill_chunk_tokens().to_vec());
        let (queries, keys, values, boundary_convolution_states) =
            match completed_prefill_chunk_tokens.as_deref() {
                Some(completed_prefill_chunk_tokens) => {
                    // The pre-window rolling buffer feeds the fused prework
                    // below; the checkpoint update replaces the state itself.
                    let pre_window_convolution_state = convolution_state
                        .current_or_zero(&self.runtime, mixed_queries_keys_values.dtype())?;
                    let checkpoint_update = convolution_state.update_with_boundary_checkpoints(
                        &self.runtime,
                        &mixed_queries_keys_values,
                        token_count,
                        completed_prefill_chunk_tokens,
                    )?;
                    // Verification rows are launch-bound decode work: the fused
                    // prework kernel serves them exactly as one-token decode
                    // (the kernel is numerics-contracted at two and four tokens),
                    // while plain prefill keeps the composed chain.
                    if is_verification_window
                        && is_gdn_decode_prework_eligible(
                            self.gdn_decode_prework_kernel.as_ref(),
                            token_count,
                            mixed_queries_keys_values.dtype(),
                            linear_head_dimension,
                        )
                    {
                        let fused = qwen3_5_gdn_decode_prework(
                            &self.runtime,
                            self.gdn_decode_prework_kernel
                                .as_ref()
                                .expect("prework eligibility just proved the kernel is retained"),
                            linear_key_head_count,
                            linear_value_head_count,
                            linear_head_dimension,
                            &mixed_queries_keys_values,
                            &pre_window_convolution_state,
                            &linear_attention_weights.convolution_weight,
                            &self.inverse_linear_head_dimension_scale,
                            &self.inverse_square_root_linear_head_dimension_scale,
                        )?;
                        (
                            fused.queries,
                            fused.keys,
                            fused.values,
                            checkpoint_update.boundary_convolution_states,
                        )
                    } else {
                        let (queries, keys, values) = self.convolution_to_normalized_heads(
                            &checkpoint_update.convolution_input,
                            token_count,
                            linear_attention_weights,
                            is_verification_window,
                            performance_attribution,
                        )?;
                        (
                            queries,
                            keys,
                            values,
                            checkpoint_update.boundary_convolution_states,
                        )
                    }
                }
                None => {
                    // Launch-bound decode steps and verification rows take the
                    // fused prework kernel: one Metal launch for the rolling
                    // window, conv1d, SiLU, q/k/v split, both RMS norms, both
                    // scales, and the next rolling state.
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
                            is_verification_window,
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
            is_verification_window,
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
            is_verification_window,
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
        let projected_output = self.quantized_linear_for_paged_prefill_execution_mode(
            &gated_output,
            &linear_attention_weights.output_projection,
            paged_prefill_execution_mode,
        )?;
        evaluate_linear_attention_section(
            &self.runtime,
            performance_attribution,
            PerformanceOperation::PrefillLinearAttentionEpilogueGraphicsProcessorCompletionWait,
            &[&projected_output],
            token_count,
            is_verification_window,
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
        is_verification_window: bool,
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
            is_verification_window,
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
