//! Exact decoder-state dtype derivation, shared by the streaming and resident
//! engines.
//!
//! A configuration activation label is insufficient for persistent state: MLX
//! promotes each operation from live activation, normalization, affine-scale,
//! and affine-bias dtypes. This module replays only that type flow, without
//! evaluating arrays, so cache serialization and memory admission describe the
//! tensors the production graph actually creates.
//!
//! The replay lives here, in `qwen3_5_core`, because the streaming and resident
//! engines previously carried two byte-identical copies of it: every
//! promotion-rule correction had to land twice, and a correction that landed
//! once left the copies silently divergent. The engines now reduce their
//! per-crate weight types to the small dtype contracts below and share this
//! single replay.

use mlx_c_rust::MlxDtype;

use crate::decoder_cache::DecoderCacheTensorDtype;
use crate::qwen3_5_core::artifacts::{
    QuantizationMode, QuantizedExpertLayerPlan, SafetensorsDtype,
};
use crate::qwen3_5_core::decoder::Qwen3_5DecoderLayerCacheDtypes;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

/// The dtype contract one affine weight carries into the type flow. Each
/// engine's weight enum reduces to this before the shared replay runs, so the
/// replay stays independent of how an engine binds its tensors.
#[derive(Debug, Clone, Copy)]
pub(crate) enum AffineDtypeContract {
    NativeBfloat16,
    Jangtq2Codebook,
    Quantized { scales: MlxDtype, biases: MlxDtype },
}

/// The normalized input activation plus the linear-attention projections and
/// normalizations one layer needs for its cache and output dtype derivation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LinearAttentionDtypeFlow {
    pub(crate) normalized_input_dtype: MlxDtype,
    pub(crate) input_queries_keys_values_projection: AffineDtypeContract,
    pub(crate) convolution_weight_dtype: MlxDtype,
    pub(crate) normalization_weight_dtype: MlxDtype,
    pub(crate) output_gate_projection: AffineDtypeContract,
    pub(crate) output_projection: AffineDtypeContract,
}

/// The normalized input activation plus the full-attention projections and
/// normalizations one layer needs for its cache and output dtype derivation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FullAttentionDtypeFlow {
    pub(crate) normalized_input_dtype: MlxDtype,
    pub(crate) query_projection: AffineDtypeContract,
    pub(crate) query_normalization_weight_dtype: MlxDtype,
    pub(crate) key_projection: AffineDtypeContract,
    pub(crate) key_normalization_weight_dtype: MlxDtype,
    pub(crate) value_projection: AffineDtypeContract,
    pub(crate) output_projection: AffineDtypeContract,
}

/// The normalized post-attention activation plus the dense MLP projections one
/// layer needs for its output dtype derivation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseMlpDtypeFlow {
    pub(crate) normalized_attention_dtype: MlxDtype,
    pub(crate) gate_projection: AffineDtypeContract,
    pub(crate) up_projection: AffineDtypeContract,
    pub(crate) down_projection: AffineDtypeContract,
}

/// The normalized post-attention activation plus the shared-expert projections
/// of a mixture-of-experts layer. The routed experts' dtype contract arrives
/// separately as the layer's quantized expert plan, because each engine
/// resolves that plan differently.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MixtureOfExpertsDtypeFlow {
    pub(crate) normalized_attention_dtype: MlxDtype,
    pub(crate) shared_expert_gate_projection: AffineDtypeContract,
    pub(crate) shared_expert_up_projection: AffineDtypeContract,
    pub(crate) shared_expert_down_projection: AffineDtypeContract,
    pub(crate) shared_expert_output_gate_projection: AffineDtypeContract,
}

/// Derives the linear-attention cache dtype (convolution state) and the
/// attention output dtype for one layer.
pub(crate) fn linear_attention_dtypes(
    flow: &LinearAttentionDtypeFlow,
) -> Result<(Qwen3_5DecoderLayerCacheDtypes, MlxDtype), Qwen3_5ExecutionError> {
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
        Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
            convolution: decoder_cache_tensor_dtype(convolution_state_dtype)?,
        },
        affine_output_dtype(gated_output_dtype, &flow.output_projection)?,
    ))
}

/// Derives the full-attention cache dtypes (keys and values) and the attention
/// output dtype for one layer.
pub(crate) fn full_attention_dtypes(
    flow: &FullAttentionDtypeFlow,
) -> Result<(Qwen3_5DecoderLayerCacheDtypes, MlxDtype), Qwen3_5ExecutionError> {
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
        Qwen3_5DecoderLayerCacheDtypes::FullAttention {
            keys: decoder_cache_tensor_dtype(normalized_key_dtype)?,
            values: decoder_cache_tensor_dtype(value_projection_dtype)?,
        },
        affine_output_dtype(gated_attention_dtype, &flow.output_projection)?,
    ))
}

/// Derives the output dtype of one dense MLP.
pub(crate) fn dense_mlp_output_dtype(
    flow: &DenseMlpDtypeFlow,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
    let gate_dtype = affine_output_dtype(flow.normalized_attention_dtype, &flow.gate_projection)?;
    let up_dtype = affine_output_dtype(flow.normalized_attention_dtype, &flow.up_projection)?;
    let activated_dtype = promote_floating_dtypes(gate_dtype, up_dtype)?;
    affine_output_dtype(activated_dtype, &flow.down_projection)
}

/// Derives the combined sparse-plus-shared output dtype of one
/// mixture-of-experts MLP. The routed-expert leg resolves first, then the
/// shared expert, then the final promotion — the order the production graph
/// evaluates them in.
pub(crate) fn mixture_of_experts_output_dtype(
    flow: &MixtureOfExpertsDtypeFlow,
    expert_layer_plan: &QuantizedExpertLayerPlan,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
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
pub(crate) fn embedding_output_dtype(
    embedding_contract: &AffineDtypeContract,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
    match embedding_contract {
        AffineDtypeContract::NativeBfloat16 => Ok(MlxDtype::BFloat16),
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
pub(crate) fn affine_output_dtype(
    activation_dtype: MlxDtype,
    affine_contract: &AffineDtypeContract,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
    let affine_parameter_dtype = match affine_contract {
        AffineDtypeContract::NativeBfloat16 => MlxDtype::BFloat16,
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
pub(crate) fn expert_projection_output_dtype(
    activation_dtype: MlxDtype,
    expert_layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
) -> Result<MlxDtype, Qwen3_5ExecutionError> {
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
pub(crate) fn expert_parameter_dtype(
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

/// MLX's floating dtype promotion, as the decoder graph performs it.
pub(crate) fn promote_floating_dtypes(
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

/// Narrows one MLX dtype to the cache-serializable dtype set.
pub(crate) fn decoder_cache_tensor_dtype(
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
