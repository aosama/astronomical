//! One ModernBERT embedding forward pass as a single lazy MLX graph.
//!
//! Layout contract for the mlx-community ModernBERT affine artifact:
//! fused `Wqkv` produces query, key, then value; global layers (index divisible
//! by `global_attn_every_n_layers`) attend bidirectionally without a mask;
//! local layers use a symmetric bidirectional window of `local_attention // 2`
//! positions on each side; every layer ends with a gated MLP whose activation
//! is GELU. Pooling and normalization happen on the host after one evaluation.

use astronomical_ipc_protocol::EmbeddingsFailureReason;
use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, MlxSafetensors,
};

use crate::modernbert::configuration::ModernBertConfiguration;

const QUANTIZED_WEIGHTS_TAIL: &str = ".weight";
const QUANTIZED_SCALES_TAIL: &str = ".scales";
const QUANTIZED_BIASES_TAIL: &str = ".biases";

/// Computes one unit-norm pooled vector for one token sequence.
pub(super) fn embed_token_ids(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    configuration: &ModernBertConfiguration,
    token_ids: &[u32],
    requested_dimensions: Option<u32>,
) -> Result<Vec<f32>, EmbeddingsFailureReason> {
    let sequence_length = token_ids.len();
    let token_id_array = runtime
        .array_from_i32(
            &token_ids
                .iter()
                .map(|token_id| i32::try_from(*token_id).unwrap_or(0))
                .collect::<Vec<_>>(),
            &[1, sequence_length as i32],
        )
        .map_err(runtime_failure)?;
    let hidden_states = embedding_layer_norm(runtime, tensors, configuration, &token_id_array)?;
    let final_states = encode_all_layers(
        runtime,
        tensors,
        configuration,
        hidden_states,
        sequence_length,
    )?;
    let normalized_states = layer_norm(
        runtime,
        tensors,
        "model.final_norm.weight",
        &final_states,
        configuration,
    )?;
    let pooled = mean_pool(runtime, configuration, &normalized_states, token_ids)?;
    let normalized = l2_normalize(runtime, &pooled)?;
    let width_limited = match requested_dimensions {
        Some(requested_width) if requested_width < configuration.hidden_size => {
            slice_width(runtime, &normalized, requested_width)?
        }
        _ => normalized,
    };
    let output_vector = match requested_dimensions {
        Some(requested_width) if requested_width < configuration.hidden_size => {
            l2_normalize(runtime, &width_limited)?
        }
        _ => width_limited,
    };
    output_vector.to_vec_f32().map_err(runtime_failure)
}

fn runtime_failure(runtime_error: MlxRuntimeError) -> EmbeddingsFailureReason {
    EmbeddingsFailureReason::FatalExecution {
        reason: format!("embedding forward pass failed: {runtime_error}")
            .chars()
            .take(256)
            .collect(),
    }
}

fn embedding_layer_norm(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    configuration: &ModernBertConfiguration,
    token_id_array: &MlxArray,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let embedded = quantized_gather(
        runtime,
        tensors,
        "model.embeddings.tok_embeddings",
        token_id_array,
        configuration,
    )?;
    layer_norm(
        runtime,
        tensors,
        "model.embeddings.norm.weight",
        &embedded,
        configuration,
    )
}

fn encode_all_layers(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    configuration: &ModernBertConfiguration,
    embedding_states: MlxArray,
    sequence_length: usize,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let local_window_half = i64::from(configuration.local_attention_window) / 2;
    // Within window_half + 1 positions every pair is inside the window, so the
    // additive mask is mathematically identical to unmasked attention.
    //
    // For sequences longer than the window the mask is a dense [1,1,seq,seq] additive
    // tensor. At the ModernBERT max (8192 tokens) this is ~256 MiB for f32; the
    // first-slice implementation accepts the cost for correctness. A banded or
    // block-sparse representation would reduce memory for long inputs in a later slice.
    let local_attention_mask = if sequence_length as i64 > local_window_half + 1 {
        Some(build_local_attention_mask(
            runtime,
            sequence_length,
            local_window_half,
        )?)
    } else {
        None
    };
    let mut layer_states = embedding_states;
    for layer_index in 0..configuration.layer_count {
        layer_states = encode_one_layer(
            runtime,
            tensors,
            configuration,
            layer_states,
            layer_index,
            sequence_length,
            local_attention_mask.as_ref(),
        )?;
    }
    Ok(layer_states)
}

fn encode_one_layer(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    configuration: &ModernBertConfiguration,
    layer_states: MlxArray,
    layer_index: u32,
    sequence_length: usize,
    local_attention_mask: Option<&MlxArray>,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let layer_prefix = format!("model.layers.{layer_index}");
    let residual_states = if layer_index == 0 {
        // ModernBERT applies no attention norm on layer zero (Identity).
        let attention_output = encode_attention(
            runtime,
            tensors,
            configuration,
            &layer_states,
            &format!("{layer_prefix}.attn"),
            layer_index,
            sequence_length,
            local_attention_mask,
        )?;
        runtime
            .add(&layer_states, &attention_output)
            .map_err(runtime_failure)?
    } else {
        let normalized = layer_norm(
            runtime,
            tensors,
            &format!("{layer_prefix}.attn_norm.weight"),
            &layer_states,
            configuration,
        )?;
        let attention_output = encode_attention(
            runtime,
            tensors,
            configuration,
            &normalized,
            &format!("{layer_prefix}.attn"),
            layer_index,
            sequence_length,
            local_attention_mask,
        )?;
        runtime
            .add(&layer_states, &attention_output)
            .map_err(runtime_failure)?
    };
    let mlp_normalized = layer_norm(
        runtime,
        tensors,
        &format!("{layer_prefix}.mlp_norm.weight"),
        &residual_states,
        configuration,
    )?;
    let mlp_output = encode_mlp(
        runtime,
        tensors,
        configuration,
        &mlp_normalized,
        &format!("{layer_prefix}.mlp"),
    )?;
    runtime
        .add(&residual_states, &mlp_output)
        .map_err(runtime_failure)
}

fn encode_attention(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    configuration: &ModernBertConfiguration,
    normalized_states: &MlxArray,
    attention_prefix: &str,
    layer_index: u32,
    sequence_length: usize,
    local_attention_mask: Option<&MlxArray>,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let head_dimension = configuration.head_dimension();
    let head_count = configuration.attention_head_count;
    let fused_qkv = quantized_matmul(
        runtime,
        tensors,
        &format!("{attention_prefix}.Wqkv"),
        normalized_states,
        configuration,
    )?;
    let sequence_length_i32 = i32::try_from(sequence_length).unwrap_or(i32::MAX);
    let split_qkv = runtime
        .reshape(
            &fused_qkv,
            &[
                1,
                sequence_length_i32,
                3,
                head_count as i32,
                head_dimension as i32,
            ],
        )
        .map_err(runtime_failure)?;
    let reordered = runtime
        .transpose_axes(&split_qkv, &[0, 3, 2, 1, 4])
        .map_err(runtime_failure)?;
    let queries = slice_head(runtime, &reordered, 0)?;
    let keys = slice_head(runtime, &reordered, 1)?;
    let values = slice_head(runtime, &reordered, 2)?;
    let rope_theta = if is_global_layer(configuration, layer_index) {
        configuration.global_rope_theta
    } else {
        configuration.local_rope_theta
    };
    let rope_dimension = i32::try_from(head_dimension).unwrap_or(i32::MAX);
    let rotated_queries = runtime
        .rope(&queries, rope_dimension, rope_theta, 0)
        .map_err(runtime_failure)?;
    let rotated_keys = runtime
        .rope(&keys, rope_dimension, rope_theta, 0)
        .map_err(runtime_failure)?;
    let attention_scale = f32::from(head_dimension as u16).powf(-0.5);
    let attended = if is_global_layer(configuration, layer_index) {
        runtime
            .scaled_dot_product_attention(&rotated_queries, &rotated_keys, &values, attention_scale)
            .map_err(runtime_failure)?
    } else if let Some(attention_mask) = local_attention_mask {
        runtime
            .masked_scaled_dot_product_attention(
                &rotated_queries,
                &rotated_keys,
                &values,
                attention_scale,
                attention_mask,
            )
            .map_err(runtime_failure)?
    } else {
        runtime
            .scaled_dot_product_attention(&rotated_queries, &rotated_keys, &values, attention_scale)
            .map_err(runtime_failure)?
    };
    let sequence_ordered = runtime
        .transpose_axes(&attended, &[0, 2, 1, 3])
        .map_err(runtime_failure)?;
    let flat_attention = runtime
        .reshape(
            &sequence_ordered,
            &[1, sequence_length_i32, configuration.hidden_size as i32],
        )
        .map_err(runtime_failure)?;
    quantized_matmul(
        runtime,
        tensors,
        &format!("{attention_prefix}.Wo"),
        &flat_attention,
        configuration,
    )
}

fn encode_mlp(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    configuration: &ModernBertConfiguration,
    mlp_normalized_states: &MlxArray,
    mlp_prefix: &str,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let fused_wi = quantized_matmul(
        runtime,
        tensors,
        &format!("{mlp_prefix}.Wi"),
        mlp_normalized_states,
        configuration,
    )?;
    let intermediate_width = fused_wi.shape().last().copied().unwrap_or(0) / 2;
    let input_states = slice_tail(runtime, &fused_wi, 0, intermediate_width)?;
    let gate_states = slice_tail(
        runtime,
        &fused_wi,
        intermediate_width,
        intermediate_width * 2,
    )?;
    let activated_gate = runtime.gelu(&gate_states).map_err(runtime_failure)?;
    let gated_input = runtime
        .multiply(&activated_gate, &input_states)
        .map_err(runtime_failure)?;
    quantized_matmul(
        runtime,
        tensors,
        &format!("{mlp_prefix}.Wo"),
        &gated_input,
        configuration,
    )
}

fn quantized_gather(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    tensor_prefix: &str,
    token_id_array: &MlxArray,
    configuration: &ModernBertConfiguration,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let quantized_weights = tensor(tensors, &format!("{tensor_prefix}{QUANTIZED_WEIGHTS_TAIL}"))?;
    let scales = tensor(tensors, &format!("{tensor_prefix}{QUANTIZED_SCALES_TAIL}"))?;
    let biases = tensor(tensors, &format!("{tensor_prefix}{QUANTIZED_BIASES_TAIL}"))?;
    let selected_weights = runtime
        .take_axis(&quantized_weights, token_id_array, 0)
        .map_err(runtime_failure)?;
    let selected_scales = runtime
        .take_axis(&scales, token_id_array, 0)
        .map_err(runtime_failure)?;
    let selected_biases = runtime
        .take_axis(&biases, token_id_array, 0)
        .map_err(runtime_failure)?;
    runtime
        .dequantize_affine(
            &selected_weights,
            &selected_scales,
            &selected_biases,
            configuration.quantization_group_size as i32,
            configuration.quantization_bits as i32,
        )
        .map_err(runtime_failure)
}

fn quantized_matmul(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    tensor_prefix: &str,
    activations: &MlxArray,
    configuration: &ModernBertConfiguration,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let quantized_weights = tensor(tensors, &format!("{tensor_prefix}{QUANTIZED_WEIGHTS_TAIL}"))?;
    let scales = tensor(tensors, &format!("{tensor_prefix}{QUANTIZED_SCALES_TAIL}"))?;
    let biases = tensor(tensors, &format!("{tensor_prefix}{QUANTIZED_BIASES_TAIL}"))?;
    runtime
        .quantized_matmul_affine(
            activations,
            &quantized_weights,
            &scales,
            &biases,
            true,
            configuration.quantization_group_size as i32,
            configuration.quantization_bits as i32,
        )
        .map_err(runtime_failure)
}

fn layer_norm(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    weight_tensor_name: &str,
    input_states: &MlxArray,
    configuration: &ModernBertConfiguration,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let weight_f16 = tensor(tensors, weight_tensor_name)?;
    let weight = runtime
        .astype(&weight_f16, MlxDtype::Float32)
        .map_err(runtime_failure)?;
    let hidden_size_i32 = configuration.hidden_size as i32;
    let zero_bias = runtime
        .zeros(&[hidden_size_i32], MlxDtype::Float32)
        .map_err(runtime_failure)?;
    runtime
        .layer_norm(
            input_states,
            &weight,
            &zero_bias,
            configuration.layer_norm_epsilon,
        )
        .map_err(runtime_failure)
}

fn mean_pool(
    runtime: &MlxRuntime,
    configuration: &ModernBertConfiguration,
    final_states: &MlxArray,
    token_ids: &[u32],
) -> Result<MlxArray, EmbeddingsFailureReason> {
    // CLS/SEP/PAD stay in the encoder so attention can use them, but averaging
    // them into the returned vector collapsed short-text discrimination on GPU
    // (Romeo lines vs finance). Pool only content tokens.
    let content_mask: Vec<f32> = token_ids
        .iter()
        .map(|token_id| {
            if configuration.is_excluded_from_pooled_mean(*token_id) {
                0.0
            } else {
                1.0
            }
        })
        .collect();
    let content_token_count: f32 = content_mask.iter().sum();
    if content_token_count < 1.0 {
        return Err(EmbeddingsFailureReason::InvalidRequest {
            reason: "embedding input encoded to zero content tokens".to_owned(),
        });
    }
    let sequence_length_i32 = i32::try_from(token_ids.len()).unwrap_or(i32::MAX);
    let mask_array = runtime
        .array_from_f32(&content_mask, &[1, sequence_length_i32, 1])
        .map_err(runtime_failure)?;
    let masked_states = runtime
        .multiply(final_states, &mask_array)
        .map_err(runtime_failure)?;
    let token_sum = runtime
        .sum_axis(&masked_states, 1, true)
        .map_err(runtime_failure)?;
    runtime
        .multiply_scalar(&token_sum, 1.0 / content_token_count)
        .map_err(runtime_failure)
}

fn l2_normalize(
    runtime: &MlxRuntime,
    pooled: &MlxArray,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let squared = runtime.multiply(pooled, pooled).map_err(runtime_failure)?;
    let squared_sum = runtime
        .sum_axis(&squared, 2, true)
        .map_err(runtime_failure)?;
    let norm = runtime.sqrt(&squared_sum).map_err(runtime_failure)?;
    runtime.divide(pooled, &norm).map_err(runtime_failure)
}

fn slice_width(
    runtime: &MlxRuntime,
    normalized: &MlxArray,
    requested_width: u32,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let requested_width_i32 = i32::try_from(requested_width).unwrap_or(i32::MAX);
    runtime
        .slice(
            normalized,
            &[0, 0, 0],
            &[1, 1, requested_width_i32],
            &[1, 1, 1],
        )
        .map_err(runtime_failure)
}

fn slice_head(
    runtime: &MlxRuntime,
    reordered_qkv: &MlxArray,
    qkv_index: i32,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let qkv_start = qkv_index;
    runtime
        .slice(
            reordered_qkv,
            &[0, 0, qkv_start, 0, 0],
            &[
                1,
                reordered_qkv.shape()[1],
                qkv_start + 1,
                reordered_qkv.shape()[3],
                reordered_qkv.shape()[4],
            ],
            &[1, 1, 1, 1, 1],
        )
        .map_err(runtime_failure)
        .and_then(|sliced| runtime.squeeze_axis(&sliced, 2).map_err(runtime_failure))
}

fn slice_tail(
    runtime: &MlxRuntime,
    fused: &MlxArray,
    start: i32,
    stop: i32,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let width = fused.shape().last().copied().unwrap_or(0);
    runtime
        .slice(
            fused,
            &[0, 0, start],
            &[1, fused.shape()[1], stop.min(width)],
            &[1, 1, 1],
        )
        .map_err(runtime_failure)
}

fn build_local_attention_mask(
    runtime: &MlxRuntime,
    sequence_length: usize,
    local_window_half: i64,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    let sequence_length_i32 =
        i32::try_from(sequence_length).map_err(|_| EmbeddingsFailureReason::FatalExecution {
            reason: "embedding sequence exceeds the mask index range".to_owned(),
        })?;
    let mut mask_values = vec![f32::NEG_INFINITY; sequence_length * sequence_length];
    for row_index in 0..sequence_length {
        for column_index in 0..sequence_length {
            let position_distance = i64::from(row_index as i32 - column_index as i32).abs();
            if position_distance <= local_window_half {
                mask_values[row_index * sequence_length + column_index] = 0.0;
            }
        }
    }
    runtime
        .array_from_f32(
            &mask_values,
            &[1, 1, sequence_length_i32, sequence_length_i32],
        )
        .map_err(runtime_failure)
}

fn is_global_layer(configuration: &ModernBertConfiguration, layer_index: u32) -> bool {
    layer_index % configuration.global_attn_every_n_layers == 0
}

fn tensor(
    tensors: &MlxSafetensors,
    tensor_name: &str,
) -> Result<MlxArray, EmbeddingsFailureReason> {
    tensors
        .tensor(tensor_name)
        .map_err(|tensor_error| EmbeddingsFailureReason::FatalExecution {
            reason: format!("embedding tensor {tensor_name} is unavailable: {tensor_error}")
                .chars()
                .take(256)
                .collect(),
        })
}
