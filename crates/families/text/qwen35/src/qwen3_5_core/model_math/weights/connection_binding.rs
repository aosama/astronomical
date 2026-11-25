//! Where a decoder layer's and the model's norms come from, and the
//! Flash-Next hyper-connection projections that follow them (issue #2533).
//!
//! Flash-Next replaces Qwen 3.5's two per-layer norms and its final norm
//! with hyper-connections over its residual lanes. Each hyper-connection
//! begins with a norm over all lanes, so that norm is bound in the slot of
//! the Qwen 3.5 norm it stands in for, and only the mixing projections are
//! Flash-Next's own.

use std::collections::HashMap;

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen4_exp::hyper_connection::HyperConnectionWeights;
use crate::qwen4_exp::layer_weights::FlashNextLayerWeights;

use super::take_tensor;

/// The tensor names of a layer's norm before attention and before its
/// feed-forward block.
fn layer_norm_tensor_names(
    qwen3_5_config: &Qwen3_5Config,
    decoder_layer_prefix: &str,
) -> (String, String) {
    if qwen3_5_config.qwen4_exp_architecture().is_some() {
        return (
            format!("{decoder_layer_prefix}.attn_hyper_connection.hc_norm.weight"),
            format!("{decoder_layer_prefix}.mlp_hyper_connection.hc_norm.weight"),
        );
    }
    (
        format!("{decoder_layer_prefix}.input_layernorm.weight"),
        format!("{decoder_layer_prefix}.post_attention_layernorm.weight"),
    )
}

/// The tensor name of the norm the vocabulary head's input passes through.
fn final_norm_tensor_name(qwen3_5_config: &Qwen3_5Config) -> String {
    if qwen3_5_config.qwen4_exp_architecture().is_some() {
        return "language_model.model.hyper_connection_mixer.hc_norm.weight".to_owned();
    }
    "language_model.model.norm.weight".to_owned()
}

/// A layer's two norms, and Flash-Next's additions after them.
pub(super) struct LayerNormsAndAdditions {
    pub input_normalization_weight: MlxArray,
    pub post_attention_normalization_weight: MlxArray,
    pub flash_next: Option<Box<FlashNextLayerWeights>>,
}

pub(super) fn take_layer_norms_and_additions(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
    decoder_layer_index: usize,
    decoder_layer_prefix: &str,
) -> Result<LayerNormsAndAdditions, TextExecutionError> {
    let (input_name, post_attention_name) =
        layer_norm_tensor_names(qwen3_5_config, decoder_layer_prefix);
    let flash_next = match qwen3_5_config.qwen4_exp_architecture() {
        Some(architecture) => Some(Box::new(FlashNextLayerWeights::take(
            bound_tensors,
            qwen3_5_config,
            &architecture,
            decoder_layer_index,
        )?)),
        None => None,
    };
    Ok(LayerNormsAndAdditions {
        input_normalization_weight: take_tensor(bound_tensors, input_name)?,
        post_attention_normalization_weight: take_tensor(bound_tensors, post_attention_name)?,
        flash_next,
    })
}

/// The norm before the vocabulary head and, for Flash-Next, the lane mixer
/// after it.
pub(super) fn take_model_output_norm(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<(MlxArray, Option<Box<HyperConnectionWeights>>), TextExecutionError> {
    let final_normalization_weight =
        take_tensor(bound_tensors, final_norm_tensor_name(qwen3_5_config))?;
    Ok((
        final_normalization_weight,
        take_final_lane_mixer(bound_tensors, qwen3_5_config)?,
    ))
}

/// Flash-Next's final lane mixer; nothing for Qwen 3.5.
fn take_final_lane_mixer(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<Option<Box<HyperConnectionWeights>>, TextExecutionError> {
    if qwen3_5_config.qwen4_exp_architecture().is_none() {
        return Ok(None);
    }
    Ok(Some(Box::new(HyperConnectionWeights::take(
        bound_tensors,
        qwen3_5_config,
        "language_model.model.hyper_connection_mixer",
        false,
    )?)))
}

/// Flash-Next norms that store `scale - 1`. Its gated linear-attention
/// output norm stores the scale itself and is not one of them.
pub(super) fn is_qwen4_exp_centered_norm_weight_tensor(tensor_name: &str) -> bool {
    [
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
        ".hc_norm.weight",
        ".ple.norm_key.weight",
        ".ple.norm_query.weight",
        ".ple.norm_conv.weight",
        ".indexer.q_layernorm.weight",
        ".indexer.k_layernorm.weight",
    ]
    .iter()
    .any(|suffix| tensor_name.ends_with(suffix))
}

/// Flash-Next's zero-centred norms. The reference forms `1 + deviation`
/// and the whole norm in float32; a bfloat16 sum would round the scale to
/// 8 bits before it is used.
pub(super) fn apply_centered_norm_storage_convention(
    runtime: &MlxRuntime,
    qwen3_5_config: &Qwen3_5Config,
    tensor_name: &str,
    tensor: MlxArray,
) -> Result<MlxArray, TextExecutionError> {
    if qwen3_5_config.qwen4_exp_architecture().is_none()
        || !is_qwen4_exp_centered_norm_weight_tensor(tensor_name)
    {
        return Ok(tensor);
    }
    let widened = runtime.astype(&tensor, mlx_bridge::MlxDtype::Float32)?;
    super::add_unit_scale(runtime, widened)
}

pub(super) fn append_final_lane_mixer_arrays<'weights>(
    final_lane_mixer: Option<&'weights HyperConnectionWeights>,
    arrays: &mut Vec<&'weights MlxArray>,
) {
    if let Some(final_lane_mixer) = final_lane_mixer {
        final_lane_mixer.append_array_references(arrays);
    }
}
