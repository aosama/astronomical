use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

use super::Qwen3_5ExecutionError;
use super::decoder_layer_weights::Qwen3_5LinearAttentionWeights;
use super::gated_delta_boundary_checkpoints::qwen3_5_gated_delta_sequence_with_boundary_checkpoints;
use super::gated_delta_sequence::qwen3_5_gated_delta_sequence;
use super::model::Qwen3_5Model;
use super::tensor_slicing::slice_last_dimension;
use crate::decoder_cache::{ConvolutionState, GatedDeltaRecurrentState};
use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;
use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;

const GATED_DELTA_STEP_OPERATION: &str = "apply one Qwen3.5 gated-delta recurrent step";

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
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let linear_key_head_count = self.config.linear_key_head_count() as i32;
        let linear_value_head_count = self.config.linear_value_head_count() as i32;
        let linear_head_dimension = self.config.linear_key_head_dimension() as i32;
        let linear_key_dimension = self.config.linear_key_dimension() as i32;
        let linear_value_dimension = self.config.linear_value_dimension() as i32;
        let linear_convolution_dimension = self.config.linear_convolution_dimension() as i32;
        let rms_norm_epsilon = f32::from_bits(self.config.rms_norm_epsilon_bits());
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
        let completed_prefill_chunk_tokens = boundary_checkpoint_collector
            .as_ref()
            .map(|collector| collector.completed_prefill_chunk_tokens().to_vec());
        let (convolution_input, boundary_convolution_states) = match completed_prefill_chunk_tokens
            .as_deref()
        {
            Some(completed_prefill_chunk_tokens) => {
                let checkpoint_update = convolution_state.update_with_boundary_checkpoints(
                    &self.runtime,
                    &mixed_queries_keys_values,
                    token_count,
                    completed_prefill_chunk_tokens,
                )?;
                (
                    checkpoint_update.convolution_input,
                    checkpoint_update.boundary_convolution_states,
                )
            }
            None => (
                convolution_state.update(&self.runtime, &mixed_queries_keys_values, token_count)?,
                Vec::new(),
            ),
        };
        let convolution_output = self.runtime.conv1d(
            &convolution_input,
            &linear_attention_weights.convolution_weight,
            1,
            0,
            1,
            linear_convolution_dimension,
        )?;
        let convolution_output = self.runtime.silu(&convolution_output)?;
        let queries =
            slice_last_dimension(&self.runtime, &convolution_output, 0, linear_key_dimension)?;
        let queries = self.runtime.reshape(
            &queries,
            &[1, token_count, linear_key_head_count, linear_head_dimension],
        )?;
        let keys = slice_last_dimension(
            &self.runtime,
            &convolution_output,
            linear_key_dimension,
            linear_key_dimension * 2,
        )?;
        let keys = self.runtime.reshape(
            &keys,
            &[1, token_count, linear_key_head_count, linear_head_dimension],
        )?;
        let values = slice_last_dimension(
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
        let queries = self
            .runtime
            .rms_norm_without_weight(&queries, rms_norm_epsilon)?;
        let queries = self
            .runtime
            .multiply(&queries, &self.inverse_linear_head_dimension_scale)?;
        let keys = self
            .runtime
            .rms_norm_without_weight(&keys, rms_norm_epsilon)?;
        let keys = self
            .runtime
            .multiply(&keys, &self.inverse_square_root_linear_head_dimension_scale)?;
        let update_rates = self.runtime.sigmoid(&update_logits)?;
        let decays = if token_count > 1 {
            self.runtime.apply_compiled_gated_delta_decay(
                &self.compiled_elementwise_graphs,
                &linear_attention_weights.decay_rate_logarithm,
                &decay_inputs,
                &linear_attention_weights.decay_interval_bias,
            )?
        } else {
            let decay_bias_inputs = self
                .runtime
                .add(&decay_inputs, &linear_attention_weights.decay_interval_bias)?;
            let decay_intervals = self.runtime.softplus(&decay_bias_inputs)?;
            let float32_decay_logs = self.runtime.astype(
                &linear_attention_weights.decay_rate_logarithm,
                MlxDtype::Float32,
            )?;
            let decay_rates = self.runtime.exp(&float32_decay_logs)?;
            let decay_products = self.runtime.multiply(&decay_rates, &decay_intervals)?;
            self.runtime.exp(&self.runtime.negative(&decay_products)?)?
        };
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
                    let checkpoint_result = qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
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
                    let (recurrent_output, next_recurrent_state) = qwen3_5_gated_delta_sequence(
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
        self.quantized_linear_for_paged_prefill_execution_mode(
            &gated_output,
            &linear_attention_weights.output_projection,
            paged_prefill_execution_mode,
        )
    }
}
