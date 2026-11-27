//! Exact decoder-state dtype derivation from bound model weights.
//!
//! A configuration activation label is insufficient for persistent state: MLX
//! promotes each operation from live activation, normalization, affine-scale,
//! and affine-bias dtypes. This module replays only that type flow, without
//! evaluating arrays, so astro decoder state cache serialization and memory admission describe the
//! tensors the production graph actually creates.
//!
//! The promotion replay itself lives in
//! `qwen3_5_core::decoder::astro_decoder_state_cache_dtype_flow`; this module reduces the
//! bound weight types to that replay's dtype contracts.
//! Each sparse layer's expert plan comes from the slice the caller holds,
//! such as the pager's plans.

use crate::qwen3_5_core::artifacts::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::decoder::Qwen3_5AstroDecoderStateCacheLayerDtypes;
use crate::qwen3_5_core::decoder::astro_decoder_state_cache_dtype_flow::{
    AffineDtypeContract, DenseMlpDtypeFlow, FullAttentionDtypeFlow, LinearAttentionDtypeFlow,
    MixtureOfExpertsDtypeFlow, dense_mlp_output_dtype, embedding_output_dtype,
    full_attention_dtypes, linear_attention_dtypes, mixture_of_experts_output_dtype,
    promote_floating_dtypes,
};
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
};
use crate::qwen3_5_core::model_math::dense_projection::Qwen3_5DenseProjection;
use crate::qwen3_5_core::model_math::weights::Qwen3_5Weights;
use text_generation::execution_error::TextExecutionError;

/// The dtype of the hidden states the embedding lookup hands the first
/// decoder layer, which every later layer's input inherits.
pub fn embedded_hidden_state_dtype(
    weights: &Qwen3_5Weights,
) -> Result<mlx_bridge::MlxDtype, TextExecutionError> {
    embedding_output_dtype(&affine_contract(&weights.embedding_weights))
}

/// Replays MLX's floating dtype propagation over the bound decoder graph without
/// evaluating model tensors. The resulting contract therefore follows each
/// artifact's actual affine scale and bias dtypes instead of its configured
/// activation dtype alone.
pub fn derive_astro_decoder_state_cache_layer_dtypes(
    weights: &Qwen3_5Weights,
    expert_layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<Vec<Qwen3_5AstroDecoderStateCacheLayerDtypes>, TextExecutionError> {
    if weights.final_lane_mixer.is_some() {
        return derive_lane_mixing_layer_dtypes(weights, expert_layer_plans);
    }
    derive_normalized_layer_dtypes(weights, expert_layer_plans)
}

/// Qwen 3.5's flow: each norm promotes with its weight's dtype.
fn derive_normalized_layer_dtypes(
    weights: &Qwen3_5Weights,
    expert_layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<Vec<Qwen3_5AstroDecoderStateCacheLayerDtypes>, TextExecutionError> {
    let mut hidden_state_dtype = embedded_hidden_state_dtype(weights)?;
    let mut astro_decoder_state_cache_layer_dtypes =
        Vec::with_capacity(weights.decoder_layer_weights.len());

    for (decoder_layer_index, decoder_layer_weights) in
        weights.decoder_layer_weights.iter().enumerate()
    {
        let normalized_input_dtype = promote_floating_dtypes(
            hidden_state_dtype,
            decoder_layer_weights.input_normalization_weight.dtype(),
        )?;
        let (astro_decoder_state_cache_layer_dtype, attention_output_dtype) = attention_dtypes(
            normalized_input_dtype,
            &decoder_layer_weights.attention_weights,
        )?;
        astro_decoder_state_cache_layer_dtypes.push(astro_decoder_state_cache_layer_dtype);

        // Residual promotion becomes the input dtype for the following MLP and
        // ultimately for the next decoder layer. A single FP32 affine companion
        // can therefore influence astro decoder state cache state beyond the operation that owns it.
        let attention_residual_dtype =
            promote_floating_dtypes(hidden_state_dtype, attention_output_dtype)?;
        let normalized_attention_dtype = promote_floating_dtypes(
            attention_residual_dtype,
            decoder_layer_weights
                .post_attention_normalization_weight
                .dtype(),
        )?;
        let feed_forward_output_dtype = feed_forward_output_dtype(
            normalized_attention_dtype,
            &decoder_layer_weights.mlp_weights,
            expert_layer_plans.get(decoder_layer_index),
        )?;
        hidden_state_dtype =
            promote_floating_dtypes(attention_residual_dtype, feed_forward_output_dtype)?;
    }

    Ok(astro_decoder_state_cache_layer_dtypes)
}

/// Flash-Next's flow: its norms compute in float32 and cast back, so each
/// block reads the residual lanes' own dtype, and its query and key norms
/// leave the attention inputs in that dtype too.
fn derive_lane_mixing_layer_dtypes(
    weights: &Qwen3_5Weights,
    expert_layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<Vec<Qwen3_5AstroDecoderStateCacheLayerDtypes>, TextExecutionError> {
    let mut hidden_state_dtype = embedded_hidden_state_dtype(weights)?;
    let mut layer_dtypes = Vec::with_capacity(weights.decoder_layer_weights.len());
    for (decoder_layer_index, decoder_layer_weights) in
        weights.decoder_layer_weights.iter().enumerate()
    {
        let (layer_dtype, attention_output_dtype) = attention_dtypes_with_norm_dtype(
            hidden_state_dtype,
            &decoder_layer_weights.attention_weights,
            Some(hidden_state_dtype),
        )?;
        layer_dtypes.push(layer_dtype);
        let attention_residual_dtype =
            promote_floating_dtypes(hidden_state_dtype, attention_output_dtype)?;
        let feed_forward_output_dtype = feed_forward_output_dtype(
            attention_residual_dtype,
            &decoder_layer_weights.mlp_weights,
            expert_layer_plans.get(decoder_layer_index),
        )?;
        hidden_state_dtype =
            promote_floating_dtypes(attention_residual_dtype, feed_forward_output_dtype)?;
    }
    Ok(layer_dtypes)
}

/// The astro decoder state cache dtypes one attention layer stores and the dtype it outputs.
fn attention_dtypes(
    normalized_input_dtype: mlx_bridge::MlxDtype,
    attention_weights: &Qwen3_5AttentionWeights,
) -> Result<
    (
        Qwen3_5AstroDecoderStateCacheLayerDtypes,
        mlx_bridge::MlxDtype,
    ),
    TextExecutionError,
> {
    Ok(match attention_weights {
        Qwen3_5AttentionWeights::Linear(linear_attention_weights) => {
            linear_attention_dtypes(&LinearAttentionDtypeFlow {
                normalized_input_dtype,
                input_queries_keys_values_projection: dense_contract(
                    &linear_attention_weights.input_queries_keys_values_projection,
                ),
                convolution_weight_dtype: linear_attention_weights.convolution_weight.dtype(),
                normalization_weight_dtype: linear_attention_weights.normalization_weight.dtype(),
                output_gate_projection: dense_contract(
                    &linear_attention_weights.output_gate_projection,
                ),
                output_projection: dense_contract(&linear_attention_weights.output_projection),
            })?
        }
        Qwen3_5AttentionWeights::Full(full_attention_weights) => full_attention_dtypes(
            &full_attention_dtype_flow(normalized_input_dtype, full_attention_weights, None),
        )?,
    })
}

/// [`attention_dtypes`] where the query and key norms return
/// `norm_output_dtype` rather than promoting with their weights.
fn attention_dtypes_with_norm_dtype(
    normalized_input_dtype: mlx_bridge::MlxDtype,
    attention_weights: &Qwen3_5AttentionWeights,
    norm_output_dtype: Option<mlx_bridge::MlxDtype>,
) -> Result<
    (
        Qwen3_5AstroDecoderStateCacheLayerDtypes,
        mlx_bridge::MlxDtype,
    ),
    TextExecutionError,
> {
    match attention_weights {
        Qwen3_5AttentionWeights::Full(full_attention_weights) => {
            full_attention_dtypes(&full_attention_dtype_flow(
                normalized_input_dtype,
                full_attention_weights,
                norm_output_dtype,
            ))
        }
        Qwen3_5AttentionWeights::Linear(_) => {
            attention_dtypes(normalized_input_dtype, attention_weights)
        }
    }
}

/// One full-attention layer's dtype flow; `norm_output_dtype` overrides the
/// query and key norm weights' dtypes when the norms cast back.
fn full_attention_dtype_flow(
    normalized_input_dtype: mlx_bridge::MlxDtype,
    full_attention_weights: &crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5FullAttentionWeights,
    norm_output_dtype: Option<mlx_bridge::MlxDtype>,
) -> FullAttentionDtypeFlow {
    FullAttentionDtypeFlow {
        normalized_input_dtype,
        query_projection: dense_contract(&full_attention_weights.query_projection),
        query_normalization_weight_dtype: norm_output_dtype
            .unwrap_or(full_attention_weights.query_normalization_weight.dtype()),
        key_projection: dense_contract(&full_attention_weights.key_projection),
        key_normalization_weight_dtype: norm_output_dtype
            .unwrap_or(full_attention_weights.key_normalization_weight.dtype()),
        value_projection: dense_contract(&full_attention_weights.value_projection),
        output_projection: dense_contract(&full_attention_weights.output_projection),
    }
}

/// The dtype one feed-forward block outputs. A sparse block needs its expert
/// plan; the caller passes `None` when the plan list does not cover it.
fn feed_forward_output_dtype(
    normalized_attention_dtype: mlx_bridge::MlxDtype,
    mlp_weights: &Qwen3_5DecoderFeedForwardWeights,
    expert_layer_plan: Option<&QuantizedExpertLayerPlan>,
) -> Result<mlx_bridge::MlxDtype, TextExecutionError> {
    Ok(match mlp_weights {
        Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) => {
            dense_mlp_output_dtype(&DenseMlpDtypeFlow {
                normalized_attention_dtype,
                gate_projection: dense_contract(&dense_mlp_weights.gate_projection),
                up_projection: dense_contract(&dense_mlp_weights.up_projection),
                down_projection: dense_contract(&dense_mlp_weights.down_projection),
            })?
        }
        Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(mixture_of_experts_weights) => {
            let expert_layer_plan = expert_layer_plan.ok_or(TextExecutionError::InvalidInput {
                description: "sparse astro-decoder-state-cache dtype derivation has no layer plan",
            })?;
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
    })
}

/// A dense projection's dtype contract, from its weights as stored (a
/// rotated module's included: a folded pack's float16 scales make its
/// products float16). A projection converted into register order holds bf16
/// scales and biases, so its products promote exactly as a bf16 weight's do.
fn dense_contract(projection: &Qwen3_5DenseProjection) -> AffineDtypeContract {
    match projection.stored_affine_weights() {
        Some(bound_weights) => affine_contract(bound_weights),
        None => AffineDtypeContract::Native {
            weight: mlx_bridge::MlxDtype::BFloat16,
        },
    }
}

/// Reduces one bound affine weight to the shared replay's dtype contract.
fn affine_contract(affine_weights: &Qwen3_5AffineWeights) -> AffineDtypeContract {
    match affine_weights {
        Qwen3_5AffineWeights::NativeBfloat16 { weight } => AffineDtypeContract::Native {
            weight: weight.dtype(),
        },
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
