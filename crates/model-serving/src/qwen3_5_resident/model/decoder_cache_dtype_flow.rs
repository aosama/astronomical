//! Exact decoder-state dtype derivation from bound model weights.
//!
//! A configuration activation label is insufficient for persistent state: MLX
//! promotes each operation from live activation, normalization, affine-scale,
//! and affine-bias dtypes. This module replays only that type flow, without
//! evaluating arrays, so cache serialization and memory admission describe the
//! tensors the production graph actually creates.
//!
//! The promotion replay itself is shared with the streaming engine in
//! `qwen3_5_core::decoder::decoder_cache_dtype_flow`; this module reduces the
//! resident engine's weight types to that replay's dtype contracts and
//! resolves each sparse layer's expert plan from the prebuilt plan slice.

use crate::qwen3_5_core::artifacts::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::decoder::Qwen3_5DecoderLayerCacheDtypes;
use crate::qwen3_5_core::decoder::decoder_cache_dtype_flow::{
    AffineDtypeContract, DenseMlpDtypeFlow, FullAttentionDtypeFlow, LinearAttentionDtypeFlow,
    MixtureOfExpertsDtypeFlow, dense_mlp_output_dtype, embedding_output_dtype,
    full_attention_dtypes, linear_attention_dtypes, mixture_of_experts_output_dtype,
    promote_floating_dtypes,
};

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
    expert_layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<Vec<Qwen3_5DecoderLayerCacheDtypes>, Qwen3_5ExecutionError> {
    let mut hidden_state_dtype =
        embedding_output_dtype(&affine_contract(&weights.embedding_weights))?;
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
                linear_attention_dtypes(&LinearAttentionDtypeFlow {
                    normalized_input_dtype,
                    input_queries_keys_values_projection: affine_contract(
                        &linear_attention_weights.input_queries_keys_values_projection,
                    ),
                    convolution_weight_dtype: linear_attention_weights.convolution_weight.dtype(),
                    normalization_weight_dtype: linear_attention_weights
                        .normalization_weight
                        .dtype(),
                    output_gate_projection: affine_contract(
                        &linear_attention_weights.output_gate_projection,
                    ),
                    output_projection: affine_contract(&linear_attention_weights.output_projection),
                })?
            }
            Qwen3_5AttentionWeights::Full(full_attention_weights) => {
                full_attention_dtypes(&FullAttentionDtypeFlow {
                    normalized_input_dtype,
                    query_projection: affine_contract(&full_attention_weights.query_projection),
                    query_normalization_weight_dtype: full_attention_weights
                        .query_normalization_weight
                        .dtype(),
                    key_projection: affine_contract(&full_attention_weights.key_projection),
                    key_normalization_weight_dtype: full_attention_weights
                        .key_normalization_weight
                        .dtype(),
                    value_projection: affine_contract(&full_attention_weights.value_projection),
                    output_projection: affine_contract(&full_attention_weights.output_projection),
                })?
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
                dense_mlp_output_dtype(&DenseMlpDtypeFlow {
                    normalized_attention_dtype,
                    gate_projection: affine_contract(&dense_mlp_weights.gate_projection),
                    up_projection: affine_contract(&dense_mlp_weights.up_projection),
                    down_projection: affine_contract(&dense_mlp_weights.down_projection),
                })?
            }
            Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(mixture_of_experts_weights) => {
                let expert_layer_plan = expert_layer_plans.get(decoder_layer_index).ok_or(
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "sparse decoder-cache dtype derivation has no layer plan",
                    },
                )?;
                mixture_of_experts_output_dtype(
                    &MixtureOfExpertsDtypeFlow {
                        normalized_attention_dtype,
                        shared_expert_gate_projection: affine_contract(
                            &mixture_of_experts_weights.shared_expert_gate_projection,
                        ),
                        shared_expert_up_projection: affine_contract(
                            &mixture_of_experts_weights.shared_expert_up_projection,
                        ),
                        shared_expert_down_projection: affine_contract(
                            &mixture_of_experts_weights.shared_expert_down_projection,
                        ),
                        shared_expert_output_gate_projection: affine_contract(
                            &mixture_of_experts_weights.shared_expert_output_gate_projection,
                        ),
                    },
                    expert_layer_plan,
                )?
            }
        };
        hidden_state_dtype =
            promote_floating_dtypes(attention_residual_dtype, feed_forward_output_dtype)?;
    }

    Ok(decoder_layer_cache_dtypes)
}

/// Reduces one resident affine weight to the shared replay's dtype contract.
fn affine_contract(affine_weights: &Qwen3_5AffineWeights) -> AffineDtypeContract {
    match affine_weights {
        Qwen3_5AffineWeights::NativeBfloat16 { .. } => AffineDtypeContract::NativeBfloat16,
        Qwen3_5AffineWeights::Jangtq2Codebook { .. } => AffineDtypeContract::Jangtq2Codebook,
        Qwen3_5AffineWeights::Quantized {
            quantization_scales,
            quantization_biases,
            ..
        } => AffineDtypeContract::Quantized {
            scales: quantization_scales.dtype(),
            biases: quantization_biases.dtype(),
        },
    }
}
