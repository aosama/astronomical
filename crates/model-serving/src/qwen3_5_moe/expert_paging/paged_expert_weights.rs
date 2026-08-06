//! Converts one bounded safetensors tensor map into paged MoE projection owners.

use std::collections::HashMap;

use astronomical_runtime_integration::MlxArray;

use super::expert_pager::{ExpertPagingError, Qwen3_5PagedExpertWeights};
use crate::expert_paging::{QuantizationMode, QuantizedExpertLayerPlan};
use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;

pub(super) fn build_paged_expert_weights(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    layer_plan: &QuantizedExpertLayerPlan,
) -> Result<Qwen3_5PagedExpertWeights, ExpertPagingError> {
    build_prefixed_paged_expert_weights(loaded_tensors, layer_plan, "")
}

pub(super) fn build_prefixed_paged_expert_weights(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    layer_plan: &QuantizedExpertLayerPlan,
    tensor_name_prefix: &str,
) -> Result<Qwen3_5PagedExpertWeights, ExpertPagingError> {
    let gate_projection_profile = projection_quantization_profile(layer_plan, "gate_proj")?;
    let up_projection_profile = projection_quantization_profile(layer_plan, "up_proj")?;
    let down_projection_profile = projection_quantization_profile(layer_plan, "down_proj")?;
    let gate_projection = take_affine_weights_from_tensors(
        loaded_tensors,
        layer_plan,
        tensor_name_prefix,
        "gate_proj",
        gate_projection_profile.quantization_bits,
        gate_projection_profile.quantization_group_size,
    )?;
    let up_projection = take_affine_weights_from_tensors(
        loaded_tensors,
        layer_plan,
        tensor_name_prefix,
        "up_proj",
        up_projection_profile.quantization_bits,
        up_projection_profile.quantization_group_size,
    )?;
    let down_projection = take_affine_weights_from_tensors(
        loaded_tensors,
        layer_plan,
        tensor_name_prefix,
        "down_proj",
        down_projection_profile.quantization_bits,
        down_projection_profile.quantization_group_size,
    )?;
    Ok(Qwen3_5PagedExpertWeights {
        gate_projection,
        up_projection,
        down_projection,
    })
}

struct ProjectionQuantizationProfile {
    quantization_bits: i32,
    quantization_group_size: i32,
}

fn projection_quantization_profile(
    layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
) -> Result<ProjectionQuantizationProfile, ExpertPagingError> {
    let tensor_source = layer_plan
        .tensor_sources
        .iter()
        .find(|source| {
            source.projection_name == projection_name && source.parameter_name == "weight"
        })
        .ok_or_else(|| ExpertPagingError::Runtime {
            description: format!("layer plan missing quantization profile for {projection_name}"),
        })?;
    Ok(ProjectionQuantizationProfile {
        quantization_bits: tensor_source.quantization_bits,
        quantization_group_size: tensor_source.quantization_group_size,
    })
}

fn take_affine_weights_from_tensors(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    layer_plan: &QuantizedExpertLayerPlan,
    tensor_name_prefix: &str,
    projection_name: &str,
    quantization_bits: i32,
    quantization_group_size: i32,
) -> Result<Qwen3_5AffineWeights, ExpertPagingError> {
    let weight_tensor_name = short_tensor_name(tensor_name_prefix, projection_name, "weight");
    if matches!(
        layer_plan.quantization_mode,
        QuantizationMode::NativeBfloat16
    ) {
        let weight = loaded_tensors.remove(&weight_tensor_name).ok_or_else(|| {
            ExpertPagingError::Runtime {
                description: format!("paged expert weights missing {weight_tensor_name}"),
            }
        })?;
        return Ok(Qwen3_5AffineWeights::NativeBfloat16 { weight });
    }
    let scales_tensor_name = short_tensor_name(tensor_name_prefix, projection_name, "scales");
    let biases_tensor_name = short_tensor_name(tensor_name_prefix, projection_name, "biases");
    let packed_weight =
        loaded_tensors
            .remove(&weight_tensor_name)
            .ok_or_else(|| ExpertPagingError::Runtime {
                description: format!("paged expert weights missing {weight_tensor_name}"),
            })?;
    let quantization_scales =
        loaded_tensors
            .remove(&scales_tensor_name)
            .ok_or_else(|| ExpertPagingError::Runtime {
                description: format!("paged expert weights missing {scales_tensor_name}"),
            })?;
    let quantization_biases =
        loaded_tensors
            .remove(&biases_tensor_name)
            .ok_or_else(|| ExpertPagingError::Runtime {
                description: format!("paged expert weights missing {biases_tensor_name}"),
            })?;
    Ok(Qwen3_5AffineWeights::Quantized {
        packed_weight,
        quantization_scales,
        quantization_biases,
        quantization_bits,
        quantization_group_size,
    })
}

fn short_tensor_name(
    tensor_name_prefix: &str,
    projection_name: &str,
    parameter_name: &str,
) -> String {
    if tensor_name_prefix.is_empty() {
        format!("{projection_name}.{parameter_name}")
    } else {
        format!("{tensor_name_prefix}.{projection_name}.{parameter_name}")
    }
}
