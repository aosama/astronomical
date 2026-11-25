//! Exact decoder-state dtype derivation, shared by the streaming and resident
//! engines.
//!
//! A configuration activation label is insufficient for persistent state: MLX
//! promotes each operation from live activation, normalization, affine-scale,
//! and affine-bias dtypes. This module replays only that type flow, without
//! evaluating arrays, so astro decoder state cache serialization and memory admission describe the
//! tensors the production graph actually creates.
//!
//! The replay lives here, in `qwen3_5_core`, because the streaming and resident
//! engines previously carried two byte-identical copies of it: every
//! promotion-rule correction had to land twice, and a correction that landed
//! once left the copies silently divergent. The engines now reduce their
//! per-crate weight types to the small dtype contracts below and share this
//! single replay.

use mlx_bridge::MlxDtype;

use crate::astro_decoder_state_cache::AstroDecoderStateCacheTensorDtype;
use crate::qwen3_5_core::artifacts::{
    QuantizationMode, QuantizedExpertLayerPlan, SafetensorsDtype,
};
use crate::qwen3_5_core::decoder::Qwen3_5AstroDecoderStateCacheLayerDtypes;
use text_generation::execution_error::TextExecutionError;

/// The dtype contract one affine weight carries into the type flow. Each
/// engine's weight enum reduces to this before the shared replay runs, so the
/// replay stays independent of how an engine binds its tensors.
#[derive(Debug, Clone, Copy)]
pub enum AffineDtypeContract {
    /// An unquantized weight, in the dtype it was bound in: BF16 as stored,
    /// or float32 where the engine binds for a CPU compute device (#2441).
    Native {
        weight: MlxDtype,
    },
    Jangtq2Codebook,
    Quantized {
        scales: MlxDtype,
        biases: MlxDtype,
    },
}

/// The normalized input activation plus the linear-attention projections and
/// normalizations one layer needs for its astro decoder state cache and output dtype derivation.
#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionDtypeFlow {
    pub normalized_input_dtype: MlxDtype,
    pub input_queries_keys_values_projection: AffineDtypeContract,
    pub convolution_weight_dtype: MlxDtype,
    pub normalization_weight_dtype: MlxDtype,
    pub output_gate_projection: AffineDtypeContract,
    pub output_projection: AffineDtypeContract,
}

/// The normalized input activation plus the full-attention projections and
/// normalizations one layer needs for its astro decoder state cache and output dtype derivation.
#[derive(Debug, Clone, Copy)]
pub struct FullAttentionDtypeFlow {
    pub normalized_input_dtype: MlxDtype,
    pub query_projection: AffineDtypeContract,
    pub query_normalization_weight_dtype: MlxDtype,
    pub key_projection: AffineDtypeContract,
    pub key_normalization_weight_dtype: MlxDtype,
    pub value_projection: AffineDtypeContract,
    pub output_projection: AffineDtypeContract,
}

/// The normalized post-attention activation plus the dense MLP projections one
/// layer needs for its output dtype derivation.
#[derive(Debug, Clone, Copy)]
pub struct DenseMlpDtypeFlow {
    pub normalized_attention_dtype: MlxDtype,
    pub gate_projection: AffineDtypeContract,
    pub up_projection: AffineDtypeContract,
    pub down_projection: AffineDtypeContract,
}

/// The normalized post-attention activation plus the shared-expert projections
/// of a mixture-of-experts layer. The routed experts' dtype contract arrives
/// separately as the layer's quantized expert plan, because each engine
/// resolves that plan differently.
#[derive(Debug, Clone, Copy)]
pub struct MixtureOfExpertsDtypeFlow {
    pub normalized_attention_dtype: MlxDtype,
    pub shared_expert_gate_projection: AffineDtypeContract,
    pub shared_expert_up_projection: AffineDtypeContract,
    pub shared_expert_down_projection: AffineDtypeContract,
    pub shared_expert_output_gate_projection: AffineDtypeContract,
}

/// Derives the linear-attention astro decoder state cache dtype (convolution state) and the
/// attention output dtype for one layer.
pub fn linear_attention_dtypes(
    flow: &LinearAttentionDtypeFlow,
) -> Result<(Qwen3_5AstroDecoderStateCacheLayerDtypes, MlxDtype), TextExecutionError> {
    let convolution_state_dtype = affine_output_dtype(
        flow.normalized_input_dtype,
        &flow.input_queries_keys_values_projection,
    )?;
    let convolution_output_dtype =
        promote_floating_dtypes(convolution_state_dtype, flow.convolution_weight_dtype)?;
    // Qwen multiplies normalized linear-attention queries and keys by retained
    // BF16 scales before the gated-delta kernel. The kernel declares its sequence
    // output with the promoted query dtype, independently from its FP32 state.
    let recurrent_output_dtype =
        promote_floating_dtypes(convolution_output_dtype, MlxDtype::BFloat16)?;
    let normalized_recurrent_output_dtype =
        promote_floating_dtypes(recurrent_output_dtype, flow.normalization_weight_dtype)?;
    let output_gate_dtype =
        affine_output_dtype(flow.normalized_input_dtype, &flow.output_gate_projection)?;
    let gated_output_dtype =
        promote_floating_dtypes(normalized_recurrent_output_dtype, output_gate_dtype)?;
    Ok((
        Qwen3_5AstroDecoderStateCacheLayerDtypes::LinearAttention {
            convolution: astro_decoder_state_cache_tensor_dtype(convolution_state_dtype)?,
        },
        affine_output_dtype(gated_output_dtype, &flow.output_projection)?,
    ))
}

/// Derives the full-attention astro decoder state cache dtypes (keys and values) and the attention
/// output dtype for one layer.
pub fn full_attention_dtypes(
    flow: &FullAttentionDtypeFlow,
) -> Result<(Qwen3_5AstroDecoderStateCacheLayerDtypes, MlxDtype), TextExecutionError> {
    let query_projection_dtype =
        affine_output_dtype(flow.normalized_input_dtype, &flow.query_projection)?;
    let normalized_query_dtype = promote_floating_dtypes(
        query_projection_dtype,
        flow.query_normalization_weight_dtype,
    )?;
    let key_projection_dtype =
        affine_output_dtype(flow.normalized_input_dtype, &flow.key_projection)?;
    let normalized_key_dtype =
        promote_floating_dtypes(key_projection_dtype, flow.key_normalization_weight_dtype)?;
    let value_projection_dtype =
        affine_output_dtype(flow.normalized_input_dtype, &flow.value_projection)?;
    let attention_dtype = promote_floating_dtypes(
        promote_floating_dtypes(normalized_query_dtype, normalized_key_dtype)?,
        value_projection_dtype,
    )?;
    let gated_attention_dtype = promote_floating_dtypes(attention_dtype, query_projection_dtype)?;
    Ok((
        Qwen3_5AstroDecoderStateCacheLayerDtypes::FullAttention {
            keys: astro_decoder_state_cache_tensor_dtype(normalized_key_dtype)?,
            values: astro_decoder_state_cache_tensor_dtype(value_projection_dtype)?,
        },
        affine_output_dtype(gated_attention_dtype, &flow.output_projection)?,
    ))
}

/// Derives the output dtype of one dense MLP.
pub fn dense_mlp_output_dtype(flow: &DenseMlpDtypeFlow) -> Result<MlxDtype, TextExecutionError> {
    let gate_dtype = affine_output_dtype(flow.normalized_attention_dtype, &flow.gate_projection)?;
    let up_dtype = affine_output_dtype(flow.normalized_attention_dtype, &flow.up_projection)?;
    let activated_dtype = promote_floating_dtypes(gate_dtype, up_dtype)?;
    affine_output_dtype(activated_dtype, &flow.down_projection)
}

/// Derives the combined sparse-plus-shared output dtype of one
/// mixture-of-experts MLP. The routed-expert leg resolves first, then the
/// shared expert, then the final promotion — the order the production graph
/// evaluates them in.
pub fn mixture_of_experts_output_dtype(
    flow: &MixtureOfExpertsDtypeFlow,
    expert_layer_plan: &QuantizedExpertLayerPlan,
) -> Result<MlxDtype, TextExecutionError> {
    let sparse_gate_dtype = expert_projection_output_dtype(
        flow.normalized_attention_dtype,
        expert_layer_plan,
        "gate_proj",
    )?;
    let sparse_up_dtype = expert_projection_output_dtype(
        flow.normalized_attention_dtype,
        expert_layer_plan,
        "up_proj",
    )?;
    let sparse_activated_dtype = promote_floating_dtypes(sparse_gate_dtype, sparse_up_dtype)?;
    let sparse_output_dtype =
        expert_projection_output_dtype(sparse_activated_dtype, expert_layer_plan, "down_proj")?;

    let shared_gate_dtype = affine_output_dtype(
        flow.normalized_attention_dtype,
        &flow.shared_expert_gate_projection,
    )?;
    let shared_up_dtype = affine_output_dtype(
        flow.normalized_attention_dtype,
        &flow.shared_expert_up_projection,
    )?;
    let shared_activated_dtype = promote_floating_dtypes(shared_gate_dtype, shared_up_dtype)?;
    let shared_output_dtype =
        affine_output_dtype(shared_activated_dtype, &flow.shared_expert_down_projection)?;
    let shared_output_gate_dtype = affine_output_dtype(
        flow.normalized_attention_dtype,
        &flow.shared_expert_output_gate_projection,
    )?;
    promote_floating_dtypes(
        sparse_output_dtype,
        promote_floating_dtypes(shared_output_dtype, shared_output_gate_dtype)?,
    )
}

/// Derives the embedding output dtype from the embedding table's affine
/// contract.
pub fn embedding_output_dtype(
    embedding_contract: &AffineDtypeContract,
) -> Result<MlxDtype, TextExecutionError> {
    match embedding_contract {
        AffineDtypeContract::Native { weight } => Ok(*weight),
        // A codebook page is never an embedding table; it only appears on the
        // routed-expert path, whose products carry the activation dtype.
        AffineDtypeContract::Jangtq2Codebook => Ok(MlxDtype::BFloat16),
        // MLX's graphics-processor affine dequantize primitive declares its output with the
        // scale dtype. Bias promotion occurs inside that declared output contract.
        AffineDtypeContract::Quantized { scales, .. } => Ok(*scales),
    }
}

/// Derives the output dtype of one affine (activation promoted over the
/// weight's parameter-dtype contract).
pub fn affine_output_dtype(
    activation_dtype: MlxDtype,
    affine_contract: &AffineDtypeContract,
) -> Result<MlxDtype, TextExecutionError> {
    let affine_parameter_dtype = match affine_contract {
        AffineDtypeContract::Native { weight } => *weight,
        // Codebook levels decode to bfloat16 weights on the materializing path
        // and the fused kernel writes the activation dtype, so the product
        // never widens past the bfloat16 contract of a native page.
        AffineDtypeContract::Jangtq2Codebook => MlxDtype::BFloat16,
        AffineDtypeContract::Quantized { scales, biases } => {
            promote_floating_dtypes(*scales, *biases)?
        }
    };
    promote_floating_dtypes(activation_dtype, affine_parameter_dtype)
}

/// Derives the output dtype of one routed-expert projection from the layer's
/// quantized plan.
pub fn expert_projection_output_dtype(
    activation_dtype: MlxDtype,
    expert_layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
) -> Result<MlxDtype, TextExecutionError> {
    if matches!(
        expert_layer_plan.quantization_mode_for_projection(projection_name),
        QuantizationMode::NativeBfloat16 | QuantizationMode::Jangtq2Codebook
    ) {
        return promote_floating_dtypes(activation_dtype, MlxDtype::BFloat16);
    }
    let scale_dtype = expert_parameter_dtype(expert_layer_plan, projection_name, "scales")?;
    let bias_dtype = expert_parameter_dtype(expert_layer_plan, projection_name, "biases")?;
    promote_floating_dtypes(
        activation_dtype,
        promote_floating_dtypes(scale_dtype, bias_dtype)?,
    )
}

/// Looks up one routed-expert affine parameter dtype in the layer's plan.
pub fn expert_parameter_dtype(
    expert_layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
    parameter_name: &str,
) -> Result<MlxDtype, TextExecutionError> {
    let tensor_source = expert_layer_plan
        .tensor_sources
        .iter()
        .find(|tensor_source| {
            tensor_source.projection_name == projection_name
                && tensor_source.parameter_name == parameter_name
        })
        .ok_or(TextExecutionError::InvalidInput {
            description: "expert layer plan is missing an affine parameter dtype",
        })?;
    match tensor_source.dtype {
        SafetensorsDtype::Float16 => Ok(MlxDtype::Float16),
        SafetensorsDtype::BFloat16 => Ok(MlxDtype::BFloat16),
        SafetensorsDtype::Float32 => Ok(MlxDtype::Float32),
        _ => Err(TextExecutionError::InvalidInput {
            description: "expert affine parameter has a non-floating dtype",
        }),
    }
}

/// MLX's floating dtype promotion, as the decoder graph performs it.
pub fn promote_floating_dtypes(
    left_dtype: MlxDtype,
    right_dtype: MlxDtype,
) -> Result<MlxDtype, TextExecutionError> {
    // MLX promotes mixed FP16/BF16 to FP32 because neither 16-bit format can
    // represent the other's complete range and precision contract.
    match (left_dtype, right_dtype) {
        (MlxDtype::Float32, _) | (_, MlxDtype::Float32) => Ok(MlxDtype::Float32),
        (MlxDtype::Float16, MlxDtype::BFloat16) | (MlxDtype::BFloat16, MlxDtype::Float16) => {
            Ok(MlxDtype::Float32)
        }
        (MlxDtype::Float16, MlxDtype::Float16) => Ok(MlxDtype::Float16),
        (MlxDtype::BFloat16, MlxDtype::BFloat16) => Ok(MlxDtype::BFloat16),
        _ => Err(TextExecutionError::InvalidInput {
            description: "decoder dtype flow encountered an unsupported MLX dtype",
        }),
    }
}

/// Narrows one MLX dtype to the dtype set the `astro_decoder_state_cache` can serialize.
pub fn astro_decoder_state_cache_tensor_dtype(
    mlx_dtype: MlxDtype,
) -> Result<AstroDecoderStateCacheTensorDtype, TextExecutionError> {
    match mlx_dtype {
        MlxDtype::Float16 => Ok(AstroDecoderStateCacheTensorDtype::Float16),
        MlxDtype::BFloat16 => Ok(AstroDecoderStateCacheTensorDtype::BFloat16),
        MlxDtype::Float32 => Ok(AstroDecoderStateCacheTensorDtype::Float32),
        _ => Err(TextExecutionError::InvalidInput {
            description: "astro decoder state cache state has an unsupported MLX dtype",
        }),
    }
}
