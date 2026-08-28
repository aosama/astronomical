//! Exact decoder-state dtype derivation from bound model weights.
//!
//! A configuration activation label is insufficient for persistent state: MLX
//! promotes each operation from live activation, normalization, affine-scale,
//! and affine-bias dtypes. This module replays only that type flow, without
//! evaluating arrays, so cache serialization and memory admission describe the
//! tensors the production graph actually creates.

use astronomical_runtime_integration::MlxDtype;

use crate::decoder_cache::DecoderCacheTensorDtype;
use crate::expert_paging::{QuantizationMode, QuantizedExpertLayerPlan, SafetensorsDtype};
use crate::qwen3_5::decoder::Qwen3_5DecoderLayerCacheDtypes;
use crate::qwen3_5_moe::Qwen3_5ExpertPager;

use super::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
};
use super::{Qwen3_5ExecutionError, Qwen3_5Weights};

/// Replays MLX's floating dtype propagation over the bound decoder graph without
/// evaluating model tensors. The resulting contract therefore follows each
/// artifact's actual affine scale and bias dtypes instead of its configured
/// activation dtype alone.
pub(super) fn derive_decoder_layer_cache_dtypes(
    weights: &Qwen3_5Weights,
    expert_pager: Option<&Qwen3_5ExpertPager>,
) -> Result<Vec<Qwen3_5DecoderLayerCacheDtypes>, Qwen3_5ExecutionError> {
    let mut hidden_state_dtype = embedding_output_dtype(&weights.embedding_weights)?;
    let mut decoder_layer_cache_dtypes = Vec::with_capacity(weights.decoder_layer_weights.len());

    for (decoder_layer_index, decoder_layer_weights) in
        weights.decoder_layer_weights.iter().enumerate()
    {
        let normalized_input_dtype = promote_floating_dtypes(
            hidden_state_dtype,
            decoder_layer_weights.input_normalization_weight.dtype(),
        )?;
        let (decoder_layer_cache_dtype, attention_output_dtype) = match &decoder_layer_weights
            .attention_weights
        {
            Qwen3_5AttentionWeights::Linear(linear_attention_weights) => {
                let convolution_state_dtype = affine_output_dtype(
                    normalized_input_dtype,
                    &linear_attention_weights.input_queries_keys_values_projection,
                )?;
                let convolution_output_dtype = promote_floating_dtypes(
                    convolution_state_dtype,
                    linear_attention_weights.convolution_weight.dtype(),
                )?;
                // Qwen multiplies normalized linear-attention queries and keys by retained
                // BF16 scales before the gated-delta kernel. The kernel declares its sequence
                // output with the promoted query dtype, independently from its FP32 state.
                let recurrent_output_dtype =
                    promote_floating_dtypes(convolution_output_dtype, MlxDtype::BFloat16)?;
                let normalized_recurrent_output_dtype = promote_floating_dtypes(
                    recurrent_output_dtype,
                    linear_attention_weights.normalization_weight.dtype(),
                )?;
                let output_gate_dtype = affine_output_dtype(
                    normalized_input_dtype,
                    &linear_attention_weights.output_gate_projection,
                )?;
                let gated_output_dtype =
                    promote_floating_dtypes(normalized_recurrent_output_dtype, output_gate_dtype)?;
                (
                    Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
                        convolution: decoder_cache_tensor_dtype(convolution_state_dtype)?,
                    },
                    affine_output_dtype(
                        gated_output_dtype,
                        &linear_attention_weights.output_projection,
                    )?,
                )
            }
            Qwen3_5AttentionWeights::Full(full_attention_weights) => {
                let query_projection_dtype = affine_output_dtype(
                    normalized_input_dtype,
                    &full_attention_weights.query_projection,
                )?;
                let normalized_query_dtype = promote_floating_dtypes(
                    query_projection_dtype,
                    full_attention_weights.query_normalization_weight.dtype(),
                )?;
                let key_projection_dtype = affine_output_dtype(
                    normalized_input_dtype,
                    &full_attention_weights.key_projection,
                )?;
                let normalized_key_dtype = promote_floating_dtypes(
                    key_projection_dtype,
                    full_attention_weights.key_normalization_weight.dtype(),
                )?;
                let value_projection_dtype = affine_output_dtype(
                    normalized_input_dtype,
                    &full_attention_weights.value_projection,
                )?;
                let attention_dtype = promote_floating_dtypes(
                    promote_floating_dtypes(normalized_query_dtype, normalized_key_dtype)?,
                    value_projection_dtype,
                )?;
                let gated_attention_dtype =
                    promote_floating_dtypes(attention_dtype, query_projection_dtype)?;
                (
                    Qwen3_5DecoderLayerCacheDtypes::FullAttention {
                        keys: decoder_cache_tensor_dtype(normalized_key_dtype)?,
                        values: decoder_cache_tensor_dtype(value_projection_dtype)?,
                    },
                    affine_output_dtype(
                        gated_attention_dtype,
                        &full_attention_weights.output_projection,
                    )?,
                )
            }
        };
        decoder_layer_cache_dtypes.push(decoder_layer_cache_dtype);

        // Residual promotion becomes the input dtype for the following MLP and
        // ultimately for the next decoder layer. A single FP32 affine companion
        // can therefore influence cache state beyond the operation that owns it.
        let attention_residual_dtype =
            promote_floating_dtypes(hidden_state_dtype, attention_output_dtype)?;
        let normalized_attention_dtype = promote_floating_dtypes(
            attention_residual_dtype,
            decoder_layer_weights
                .post_attention_normalization_weight
                .dtype(),
        )?;
        let feed_forward_output_dtype = match &decoder_layer_weights.mlp_weights {
            Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) => {
                let gate_dtype = affine_output_dtype(
                    normalized_attention_dtype,
                    &dense_mlp_weights.gate_projection,
                )?;
                let up_dtype = affine_output_dtype(
                    normalized_attention_dtype,
                    &dense_mlp_weights.up_projection,
                )?;
                let activated_dtype = promote_floating_dtypes(gate_dtype, up_dtype)?;
                affine_output_dtype(activated_dtype, &dense_mlp_weights.down_projection)?
            }
            Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(mixture_of_experts_weights) => {
                let expert_pager = expert_pager.ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "sparse decoder-cache dtype derivation requires an expert pager",
                })?;
                let expert_layer_plan = expert_pager.layer_plan(decoder_layer_index)?;
                let sparse_gate_dtype = expert_projection_output_dtype(
                    normalized_attention_dtype,
                    expert_layer_plan,
                    "gate_proj",
                )?;
                let sparse_up_dtype = expert_projection_output_dtype(
                    normalized_attention_dtype,
                    expert_layer_plan,
                    "up_proj",
                )?;
                let sparse_activated_dtype =
                    promote_floating_dtypes(sparse_gate_dtype, sparse_up_dtype)?;
                let sparse_output_dtype = expert_projection_output_dtype(
                    sparse_activated_dtype,
                    expert_layer_plan,
                    "down_proj",
                )?;

                let shared_gate_dtype = affine_output_dtype(
                    normalized_attention_dtype,
                    &mixture_of_experts_weights.shared_expert_gate_projection,
                )?;
                let shared_up_dtype = affine_output_dtype(
                    normalized_attention_dtype,
                    &mixture_of_experts_weights.shared_expert_up_projection,
                )?;
                let shared_activated_dtype =
                    promote_floating_dtypes(shared_gate_dtype, shared_up_dtype)?;
                let shared_output_dtype = affine_output_dtype(
                    shared_activated_dtype,
                    &mixture_of_experts_weights.shared_expert_down_projection,
                )?;
                let shared_output_gate_dtype = affine_output_dtype(
                    normalized_attention_dtype,
                    &mixture_of_experts_weights.shared_expert_output_gate_projection,
                )?;
                promote_floating_dtypes(
                    sparse_output_dtype,
                    promote_floating_dtypes(shared_output_dtype, shared_output_gate_dtype)?,
                )?
            }
        };
        hidden_state_dtype =
            promote_floating_dtypes(attention_residual_dtype, feed_forward_output_dtype)?;
    }

    Ok(decoder_layer_cache_dtypes)
}

fn embedding_output_dtype(
    embedding_weights: &Qwen3_5AffineWeights,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
    match embedding_weights {
        Qwen3_5AffineWeights::NativeBfloat16 { .. } => Ok(MlxDtype::BFloat16),
        // MLX's graphics-processor affine dequantize primitive declares its output with the
        // scale dtype. Bias promotion occurs inside that declared output contract.
        Qwen3_5AffineWeights::Quantized {
            quantization_scales,
            ..
        } => Ok(quantization_scales.dtype()),
    }
}

fn affine_output_dtype(
    activation_dtype: MlxDtype,
    affine_weights: &Qwen3_5AffineWeights,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
    let affine_parameter_dtype = match affine_weights {
        Qwen3_5AffineWeights::NativeBfloat16 { .. } => MlxDtype::BFloat16,
        Qwen3_5AffineWeights::Quantized {
            quantization_scales,
            quantization_biases,
            ..
        } => promote_floating_dtypes(quantization_scales.dtype(), quantization_biases.dtype())?,
    };
    promote_floating_dtypes(activation_dtype, affine_parameter_dtype)
}

fn expert_projection_output_dtype(
    activation_dtype: MlxDtype,
    expert_layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
    if expert_layer_plan.quantization_mode_for_projection(projection_name)
        == QuantizationMode::NativeBfloat16
    {
        return promote_floating_dtypes(activation_dtype, MlxDtype::BFloat16);
    }
    let scale_dtype = expert_parameter_dtype(expert_layer_plan, projection_name, "scales")?;
    let bias_dtype = expert_parameter_dtype(expert_layer_plan, projection_name, "biases")?;
    promote_floating_dtypes(
        activation_dtype,
        promote_floating_dtypes(scale_dtype, bias_dtype)?,
    )
}

fn expert_parameter_dtype(
    expert_layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
    parameter_name: &str,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
    let tensor_source = expert_layer_plan
        .tensor_sources
        .iter()
        .find(|tensor_source| {
            tensor_source.projection_name == projection_name
                && tensor_source.parameter_name == parameter_name
        })
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "expert layer plan is missing an affine parameter dtype",
        })?;
    match tensor_source.dtype {
        SafetensorsDtype::Float16 => Ok(MlxDtype::Float16),
        SafetensorsDtype::BFloat16 => Ok(MlxDtype::BFloat16),
        SafetensorsDtype::Float32 => Ok(MlxDtype::Float32),
        _ => Err(Qwen3_5ExecutionError::InvalidInput {
            description: "expert affine parameter has a non-floating dtype",
        }),
    }
}

fn promote_floating_dtypes(
    left_dtype: MlxDtype,
    right_dtype: MlxDtype,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
    // MLX promotes mixed FP16/BF16 to FP32 because neither 16-bit format can
    // represent the other's complete range and precision contract.
    match (left_dtype, right_dtype) {
        (MlxDtype::Float32, _) | (_, MlxDtype::Float32) => Ok(MlxDtype::Float32),
        (MlxDtype::Float16, MlxDtype::BFloat16) | (MlxDtype::BFloat16, MlxDtype::Float16) => {
            Ok(MlxDtype::Float32)
        }
        (MlxDtype::Float16, MlxDtype::Float16) => Ok(MlxDtype::Float16),
        (MlxDtype::BFloat16, MlxDtype::BFloat16) => Ok(MlxDtype::BFloat16),
        _ => Err(Qwen3_5ExecutionError::InvalidInput {
            description: "decoder dtype flow encountered an unsupported MLX dtype",
        }),
    }
}

fn decoder_cache_tensor_dtype(
    mlx_dtype: MlxDtype,
) -> Result<DecoderCacheTensorDtype, Qwen3_5ExecutionError> {
    match mlx_dtype {
        MlxDtype::Float16 => Ok(DecoderCacheTensorDtype::Float16),
        MlxDtype::BFloat16 => Ok(DecoderCacheTensorDtype::BFloat16),
        MlxDtype::Float32 => Ok(DecoderCacheTensorDtype::Float32),
        _ => Err(Qwen3_5ExecutionError::InvalidInput {
            description: "decoder cache state has an unsupported MLX dtype",
        }),
    }
}
