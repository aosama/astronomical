use astronomical_runtime_integration::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxRuntime, MlxRuntimeError,
};

use super::Qwen3_5MoEExecutionError;
use super::decoder_layer_weights::Qwen3_5MoEFullAttentionWeights;
use super::model::Qwen3_5MoEModel;
use super::tensor_slicing::slice_last_dimension;
use crate::decoder_cache::FullAttentionKeyValueState;

const FULL_ATTENTION_OPERATION: &str = "apply one Qwen3.5-MoE full-attention step";

/// Applies the post-projection Qwen3.5-MoE full-attention formula over the
/// already-fetched active keys and values. The KV state owner is responsible
/// for capacity growth and concatenation; this function runs only the
/// scaled dot-product attention and the output gate.
///
/// `active_keys` and `active_values` are the views over `[0..offset]` returned
/// by `FullAttentionKeyValueState::update_and_fetch`. They already contain the
/// newly rotated keys and transposed values appended to the cached prefix.
/// `rotated_queries` are the post-RoPE queries ready for the attention call.
/// Multi-token queries use causal attention; one-token decode does not need a
/// mask because its active key/value view contains only preceding and current tokens.
pub fn qwen3_5_moe_full_attention_step(
    runtime: &MlxRuntime,
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    rotated_queries: &MlxArray,
    active_keys: &MlxArray,
    active_values: &MlxArray,
    output_gate: &MlxArray,
    attention_scale: f32,
) -> Result<MlxArray, MlxRuntimeError> {
    let attention_shape = validate_full_attention_arguments(
        rotated_queries,
        active_keys,
        active_values,
        output_gate,
        attention_scale,
    )?;
    let is_causal = attention_shape.query_token_count > 1;
    let attention_output = if is_causal {
        runtime.causal_scaled_dot_product_attention(
            rotated_queries,
            active_keys,
            active_values,
            attention_scale,
        )?
    } else {
        runtime.scaled_dot_product_attention(
            rotated_queries,
            active_keys,
            active_values,
            attention_scale,
        )?
    };
    let attention_output = runtime.transpose_axes(&attention_output, &[0, 2, 1, 3])?;
    let attention_output = runtime.reshape(
        &attention_output,
        &[
            attention_shape.batch_size,
            attention_shape.query_token_count,
            attention_shape.output_dimension,
        ],
    )?;
    if is_causal {
        runtime.apply_compiled_attention_output_gate(
            compiled_elementwise_graphs,
            &attention_output,
            output_gate,
        )
    } else {
        let gate_weights = runtime.sigmoid(output_gate)?;
        runtime.multiply(&attention_output, &gate_weights)
    }
}

#[derive(Clone, Copy)]
struct FullAttentionShape {
    batch_size: i32,
    query_token_count: i32,
    output_dimension: i32,
}

fn validate_full_attention_arguments(
    rotated_queries: &MlxArray,
    active_keys: &MlxArray,
    active_values: &MlxArray,
    output_gate: &MlxArray,
    attention_scale: f32,
) -> Result<FullAttentionShape, MlxRuntimeError> {
    let query_shape = rotated_queries.shape();
    let key_shape = active_keys.shape();
    let value_shape = active_values.shape();
    if query_shape.len() != 4 || key_shape.len() != 4 || value_shape.len() != 4 {
        return Err(full_attention_error(
            "rotated queries, active keys, and active values must have rank four",
        ));
    }
    if key_shape != value_shape {
        return Err(full_attention_error(
            "full-attention active keys and values must have identical shapes",
        ));
    }
    let batch_size = query_shape[0];
    let query_head_count = query_shape[1];
    let query_token_count = query_shape[2];
    let head_dimension = query_shape[3];
    let key_value_head_count = key_shape[1];
    let active_key_value_token_count = key_shape[2];
    if batch_size <= 0
        || query_token_count <= 0
        || query_head_count <= 0
        || head_dimension <= 0
        || key_value_head_count <= 0
        || key_shape[0] != batch_size
        || key_shape[3] != head_dimension
        || query_head_count % key_value_head_count != 0
        || active_key_value_token_count < query_token_count
    {
        return Err(full_attention_error(
            "full-attention dimensions must be positive, grouped-query heads must divide evenly, \
             and the active KV view must cover the new query tokens",
        ));
    }
    let output_dimension = query_head_count
        .checked_mul(head_dimension)
        .ok_or_else(|| full_attention_error("full-attention output dimension overflowed"))?;
    if output_gate.shape() != [batch_size, query_token_count, output_dimension] {
        return Err(full_attention_error(
            "full-attention output gate shape is incompatible with query heads",
        ));
    }
    if !attention_scale.is_finite() || attention_scale <= 0.0 {
        return Err(full_attention_error("full-attention scale is invalid"));
    }
    Ok(FullAttentionShape {
        batch_size,
        query_token_count,
        output_dimension,
    })
}

fn full_attention_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: FULL_ATTENTION_OPERATION,
        description: description.to_owned(),
    }
}

impl Qwen3_5MoEModel {
    /// Runs one full-attention layer, threading the in-memory KV state through
    /// the single `FullAttentionKeyValueState` owner. The owner grows capacity
    /// in 256-token steps, appends the rotated keys and transposed values, and
    /// returns active views for the attention call.
    pub(super) fn forward_full_attention(
        &self,
        hidden_states: &MlxArray,
        token_count: i32,
        rope_offset_tokens: i32,
        full_attention_weights: &Qwen3_5MoEFullAttentionWeights,
        kv_state: &mut FullAttentionKeyValueState,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let query_head_count = self.config.query_head_count() as i32;
        let key_value_head_count = self.config.key_value_head_count() as i32;
        let attention_head_dimension = self.config.head_dimension() as i32;
        let attention_projection_head_dimension = attention_head_dimension * 2;
        let rotary_dimension = self.config.rotary_dimension() as i32;
        let rms_norm_epsilon = f32::from_bits(self.config.rms_norm_epsilon_bits());
        let query_projection =
            self.quantized_linear(hidden_states, &full_attention_weights.query_projection)?;
        let query_projection = self.runtime.reshape(
            &query_projection,
            &[
                1,
                token_count,
                query_head_count,
                attention_projection_head_dimension,
            ],
        )?;
        let queries = slice_last_dimension(
            &self.runtime,
            &query_projection,
            0,
            attention_head_dimension,
        )?;
        let output_gate = slice_last_dimension(
            &self.runtime,
            &query_projection,
            attention_head_dimension,
            attention_projection_head_dimension,
        )?;
        let output_gate = self.runtime.reshape(
            &output_gate,
            &[1, token_count, query_head_count * attention_head_dimension],
        )?;
        let keys = self.quantized_linear(hidden_states, &full_attention_weights.key_projection)?;
        let keys = self.runtime.reshape(
            &keys,
            &[
                1,
                token_count,
                key_value_head_count,
                attention_head_dimension,
            ],
        )?;
        let values =
            self.quantized_linear(hidden_states, &full_attention_weights.value_projection)?;
        let values = self.runtime.reshape(
            &values,
            &[
                1,
                token_count,
                key_value_head_count,
                attention_head_dimension,
            ],
        )?;

        // RoPE and RMS-norm are applied to the new keys before concatenation,
        // exactly as the model expects. The owner then appends these rotated
        // tensors to the cached slab and returns views over the full prefix.
        let normalized_queries = self.runtime.rms_norm(
            &queries,
            &full_attention_weights.query_normalization_weight,
            rms_norm_epsilon,
        )?;
        let normalized_keys = self.runtime.rms_norm(
            &keys,
            &full_attention_weights.key_normalization_weight,
            rms_norm_epsilon,
        )?;
        let transposed_queries = self
            .runtime
            .transpose_axes(&normalized_queries, &[0, 2, 1, 3])?;
        let transposed_keys = self
            .runtime
            .transpose_axes(&normalized_keys, &[0, 2, 1, 3])?;
        let transposed_values = self.runtime.transpose_axes(&values, &[0, 2, 1, 3])?;
        let rotated_queries = self.runtime.rope(
            &transposed_queries,
            rotary_dimension,
            f32::from_bits(self.config.rope_theta_bits()),
            rope_offset_tokens,
        )?;
        let rotated_keys = self.runtime.rope(
            &transposed_keys,
            rotary_dimension,
            f32::from_bits(self.config.rope_theta_bits()),
            rope_offset_tokens,
        )?;

        let (active_keys, active_values) = kv_state.update_and_fetch(
            &self.runtime,
            &rotated_keys,
            &transposed_values,
            rope_offset_tokens,
        )?;

        let gated_output = qwen3_5_moe_full_attention_step(
            &self.runtime,
            &self.compiled_elementwise_graphs,
            &rotated_queries,
            &active_keys,
            &active_values,
            &output_gate,
            (attention_head_dimension as f32).sqrt().recip(),
        )?;
        self.quantized_linear(&gated_output, &full_attention_weights.output_projection)
    }
}
