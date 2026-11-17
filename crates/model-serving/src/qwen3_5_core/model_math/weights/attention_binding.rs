//! Binds one decoder layer's attention projections from the loaded tensors:
//! every one a dense projection, bound as stored until a drafting engine
//! converts it.

use std::collections::HashMap;

use mlx_bridge::MlxArray;

use super::{take_dense_projection, take_tensor};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5FullAttentionWeights, Qwen3_5LinearAttentionWeights,
};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

pub(super) fn take_linear_attention_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
    decoder_layer_prefix: &str,
) -> Result<Qwen3_5LinearAttentionWeights, Qwen3_5ExecutionError> {
    let linear_attention_prefix = format!("{decoder_layer_prefix}.linear_attn");
    Ok(Qwen3_5LinearAttentionWeights {
        input_queries_keys_values_projection: take_dense_projection(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.in_proj_qkv"),
        )?,
        output_gate_projection: take_dense_projection(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.in_proj_z"),
        )?,
        update_rate_projection: take_dense_projection(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.in_proj_b"),
        )?,
        decay_interval_projection: take_dense_projection(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.in_proj_a"),
        )?,
        convolution_weight: take_tensor(
            bound_tensors,
            format!("{linear_attention_prefix}.conv1d.weight"),
        )?,
        decay_interval_bias: take_tensor(
            bound_tensors,
            format!("{linear_attention_prefix}.dt_bias"),
        )?,
        decay_rate_logarithm: take_tensor(
            bound_tensors,
            format!("{linear_attention_prefix}.A_log"),
        )?,
        normalization_weight: take_tensor(
            bound_tensors,
            format!("{linear_attention_prefix}.norm.weight"),
        )?,
        output_projection: take_dense_projection(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.out_proj"),
        )?,
    })
}

pub(super) fn take_full_attention_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
    decoder_layer_prefix: &str,
) -> Result<Qwen3_5FullAttentionWeights, Qwen3_5ExecutionError> {
    let full_attention_prefix = format!("{decoder_layer_prefix}.self_attn");
    Ok(Qwen3_5FullAttentionWeights {
        query_projection: take_dense_projection(
            bound_tensors,
            qwen3_5_config,
            &format!("{full_attention_prefix}.q_proj"),
        )?,
        key_projection: take_dense_projection(
            bound_tensors,
            qwen3_5_config,
            &format!("{full_attention_prefix}.k_proj"),
        )?,
        value_projection: take_dense_projection(
            bound_tensors,
            qwen3_5_config,
            &format!("{full_attention_prefix}.v_proj"),
        )?,
        output_projection: take_dense_projection(
            bound_tensors,
            qwen3_5_config,
            &format!("{full_attention_prefix}.o_proj"),
        )?,
        query_normalization_weight: take_tensor(
            bound_tensors,
            format!("{full_attention_prefix}.q_norm.weight"),
        )?,
        key_normalization_weight: take_tensor(
            bound_tensors,
            format!("{full_attention_prefix}.k_norm.weight"),
        )?,
    })
}
