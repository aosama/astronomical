//! Converts bounded SafeTensors tensors into one streamed expert-layer owner.
//!
//! The generic bounded reader returns arrays by manifest-local tensor name. This
//! module is where those names acquire Qwen semantics. It consumes every required
//! gate, up, and down projection component into one owner that can be temporary
//! for a forward or retained as a complete warm layer.
//!
//! Native BFloat16 and affine-quantized layers deliberately take different
//! branches. Quantized execution requires the packed weight *and* its exact scale,
//! bias, bit width, and group size; dropping or synthesizing any component would
//! alter model precision.

use std::collections::HashMap;

use astronomical_runtime_integration::MlxArray;

use super::expert_pager::{ExpertPagingError, Qwen3_5PagedExpertWeights};
use crate::expert_paging::{QuantizationMode, QuantizedExpertLayerPlan};
use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;

pub(super) fn build_paged_expert_weights(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    layer_plan: &QuantizedExpertLayerPlan,
) -> Result<Qwen3_5PagedExpertWeights, ExpertPagingError> {
    // `take_projection` removes ownership from the map. Construction therefore
    // either returns one self-contained page or drops all partially taken arrays
    // on error; no incomplete page can escape.
    Ok(Qwen3_5PagedExpertWeights {
        gate_projection: take_projection(loaded_tensors, layer_plan, "gate_proj")?,
        up_projection: take_projection(loaded_tensors, layer_plan, "up_proj")?,
        down_projection: take_projection(loaded_tensors, layer_plan, "down_proj")?,
    })
}

fn take_projection(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
) -> Result<Qwen3_5AffineWeights, ExpertPagingError> {
    let weight_name = format!("{projection_name}.weight");
    let weight = take_tensor(loaded_tensors, &weight_name)?;
    if layer_plan.quantization_mode_for_projection(projection_name)
        == QuantizationMode::NativeBfloat16
    {
        // Native projections have no affine side tensors. Looking for scales or
        // biases here would reject a valid mixed OptiQ source representation.
        return Ok(Qwen3_5AffineWeights::NativeBfloat16 { weight });
    }
    // The weight's validated source profile is authoritative for bit width and
    // group size. Never infer these values from filenames or model identity.
    let weight_source = layer_plan
        .tensor_sources
        .iter()
        .find(|tensor_source| {
            tensor_source.projection_name == projection_name
                && tensor_source.parameter_name == "weight"
        })
        .ok_or_else(|| ExpertPagingError::Runtime {
            description: format!("layer plan is missing {projection_name} quantization metadata"),
        })?;
    Ok(Qwen3_5AffineWeights::Quantized {
        packed_weight: weight,
        quantization_scales: take_tensor(loaded_tensors, &format!("{projection_name}.scales"))?,
        quantization_biases: take_tensor(loaded_tensors, &format!("{projection_name}.biases"))?,
        quantization_bits: weight_source.quantization_bits,
        quantization_group_size: weight_source.quantization_group_size,
    })
}

fn take_tensor(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    tensor_name: &str,
) -> Result<MlxArray, ExpertPagingError> {
    loaded_tensors
        .remove(tensor_name)
        .ok_or_else(|| ExpertPagingError::Runtime {
            description: format!("streamed expert layer is missing {tensor_name}"),
        })
}
