//! Converts bounded SafeTensors tensors into one streamed expert-layer owner.
//!
//! The generic bounded reader returns arrays by manifest-local tensor name. This
//! module is where those names acquire Qwen semantics. It consumes every required
//! gate, up, and down projection component into one owner that can be temporary
//! for a forward or retained as a complete warm layer.
//!
//! Native BFloat16, affine-quantized, and jangtq2 codebook layers deliberately
//! take different branches. Quantized execution requires the packed weight
//! *and* its exact scale, bias, bit width, and group size; dropping or
//! synthesizing any component would alter model precision. Codebook pages are
//! kept packed with their row scales and bit width for the same reason.

use std::collections::HashMap;

use astronomical_mlx_c_rust::MlxArray;

use super::expert_pager::{ExpertPagingError, Qwen3_5PagedExpertWeights};
use crate::expert_paging::{QuantizationMode, QuantizedExpertLayerPlan};
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;

pub(super) fn build_paged_expert_weights(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    layer_plan: &QuantizedExpertLayerPlan,
) -> Result<Qwen3_5PagedExpertWeights, ExpertPagingError> {
    // `take_projection` removes ownership from the map. Construction therefore
    // either returns one self-contained page or drops all partially taken arrays
    // on error; no incomplete page can escape.
    let (gate_projection, gate_activation_rotation) =
        take_projection(loaded_tensors, layer_plan, "gate_proj")?;
    let (up_projection, up_activation_rotation) =
        take_projection(loaded_tensors, layer_plan, "up_proj")?;
    let (down_projection, down_activation_rotation) =
        take_projection(loaded_tensors, layer_plan, "down_proj")?;
    Ok(Qwen3_5PagedExpertWeights {
        gate_projection,
        up_projection,
        down_projection,
        gate_activation_rotation,
        up_activation_rotation,
        down_activation_rotation,
    })
}

fn take_projection(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
) -> Result<
    (
        Qwen3_5AffineWeights,
        crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
    ),
    ExpertPagingError,
> {
    let quantization_mode = layer_plan.quantization_mode_for_projection(projection_name);
    match quantization_mode {
        QuantizationMode::NativeBfloat16 => {
            if layer_plan
                .jangtq2_rotation_for_projection(projection_name)
                .is_some()
            {
                return Err(ExpertPagingError::InvalidPagingPlan {
                    description: format!(
                        "native {projection_name} projection unexpectedly declares a jangtq2 rotation"
                    ),
                });
            }
            // Native projections have no affine side tensors. Looking for scales or
            // biases here would reject a valid mixed OptiQ source representation.
            let weight = take_tensor(loaded_tensors, &format!("{projection_name}.weight"))?;
            Ok((
                Qwen3_5AffineWeights::NativeBfloat16 { weight },
                crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation,
            ))
        }
        QuantizationMode::Jangtq2Codebook => {
            let activation_rotation = layer_plan
                .jangtq2_rotation_for_projection(projection_name)
                .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
                    description: format!(
                        "codebook {projection_name} projection is missing its rotation descriptor"
                    ),
                })?;
            let packed_code_words =
                take_tensor(loaded_tensors, &format!("{projection_name}.tq2_packed"))?;
            let row_scales = take_tensor(loaded_tensors, &format!("{projection_name}.tq2_scales"))?;
            let packed_tensor_source = layer_plan
                .tensor_sources
                .iter()
                .find(|tensor_source| {
                    tensor_source.projection_name == projection_name
                        && tensor_source.parameter_name == "tq2_packed"
                })
                .ok_or_else(|| ExpertPagingError::Runtime {
                    description: format!(
                        "layer plan is missing {projection_name} codebook metadata"
                    ),
                })?;
            let codebook_bit_width = u32::try_from(packed_tensor_source.quantization_bits)
                .map_err(|_| ExpertPagingError::Runtime {
                    description: format!("layer plan has invalid {projection_name} codebook bits"),
                })?;
            // The page stays packed: retaining decoded bfloat16 would cost
            // eight times the bytes and is exactly what kept this family's
            // expert set from fitting in RAM. Execution decides per forward
            // whether to multiply straight from the codes (decode tokens) or
            // to decode a transient copy (multi-token prefill).
            Ok((
                Qwen3_5AffineWeights::Jangtq2Codebook {
                    packed_code_words,
                    row_scales,
                    codebook_bit_width,
                },
                activation_rotation,
            ))
        }
        QuantizationMode::Affine => {
            if layer_plan
                .jangtq2_rotation_for_projection(projection_name)
                .is_some()
            {
                return Err(ExpertPagingError::InvalidPagingPlan {
                    description: format!(
                        "affine {projection_name} projection unexpectedly declares a jangtq2 rotation"
                    ),
                });
            }
            // The weight's validated source profile is authoritative for bit width and
            // group size. Never infer these values from filenames or model identity.
            let affine_weight_source = layer_plan
                .tensor_sources
                .iter()
                .find(|tensor_source| {
                    tensor_source.projection_name == projection_name
                        && tensor_source.parameter_name == "weight"
                })
                .ok_or_else(|| ExpertPagingError::Runtime {
                    description: format!(
                        "layer plan is missing {projection_name} quantization metadata"
                    ),
                })?;
            Ok((
                Qwen3_5AffineWeights::Quantized {
                    packed_weight: take_tensor(
                        loaded_tensors,
                        &format!("{projection_name}.weight"),
                    )?,
                    quantization_scales: take_tensor(
                        loaded_tensors,
                        &format!("{projection_name}.scales"),
                    )?,
                    quantization_biases: take_tensor(
                        loaded_tensors,
                        &format!("{projection_name}.biases"),
                    )?,
                    quantization_bits: affine_weight_source.quantization_bits,
                    quantization_group_size: affine_weight_source.quantization_group_size,
                },
                crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation,
            ))
        }
    }
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
