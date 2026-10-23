//! Shared startup-validated expert layer plan construction from SafeTensors headers.
//!
//! Complete pager construction shares one SafeTensors header cache across every layer and validates
//! all tensor geometry. Standalone callers retain a one-layer cache. `build_source_manifests()`
//! groups selected expert intervals by shard file for bounded native reads.
//! `contiguous_selected_runs()` groups adjacent selected experts for efficient
//! sequential I/O.

#[cfg(test)]
mod tests;

use std::collections::{HashMap, hash_map::Entry};
use std::path::{Path, PathBuf};

use crate::expert_paging::quantized_expert_manifest::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource,
};
use crate::expert_paging::quantized_expert_validation;
use crate::expert_paging::safetensors_header::{
    SafetensorsDtype, SafetensorsHeader, TensorHeaderEntry,
};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::quantizations::optiq::{ModuleWeightCodec, OptiQQuantizationProfile};

/// The three MoE projection names in an expert's SwitchMLP block.
pub(crate) const PROJECTION_NAMES: &[&str] = &["gate_proj", "up_proj", "down_proj"];

/// The three quantized parameter names for each projection.
const PARAMETER_NAMES: &[&str] = &["weight", "scales", "biases"];

/// Build startup-validated layer metadata by reading safetensors headers.
///
/// `weight_map` maps tensor names (e.g. `language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight`)
/// to shard file names (e.g. `model-00001-of-00005.safetensors`). The model directory
/// contains the actual shard files.
pub fn build_quantized_expert_layer_plan(
    model_dir: &std::path::Path,
    weight_map: &HashMap<String, String>,
    layer_prefix: &str,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<QuantizedExpertLayerPlan, ExpertManifestError> {
    build_quantized_expert_layer_plan_with_stored_names(
        model_dir,
        weight_map,
        &HashMap::new(),
        layer_prefix,
        qwen3_5_config,
    )
}

/// Builds every decoder layer plan while parsing each referenced shard header once.
pub fn build_quantized_expert_layer_plans(
    model_dir: &Path,
    weight_map: &HashMap<String, String>,
    stored_tensor_name_by_canonical_name: &HashMap<String, String>,
    config: &Qwen3_5Config,
    decoder_layer_count: usize,
) -> Result<Vec<QuantizedExpertLayerPlan>, ExpertManifestError> {
    let mut layer_plans = Vec::with_capacity(decoder_layer_count);
    let mut header_cache = HashMap::new();
    for decoder_layer_index in 0..decoder_layer_count {
        let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}.mlp");
        layer_plans.push(
            build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
                model_dir,
                weight_map,
                stored_tensor_name_by_canonical_name,
                &layer_prefix,
                config,
                &mut header_cache,
            )?,
        );
    }
    Ok(layer_plans)
}

pub(crate) fn build_quantized_expert_layer_plan_with_stored_names(
    model_dir: &std::path::Path,
    weight_map: &HashMap<String, String>,
    stored_tensor_name_by_canonical_name: &HashMap<String, String>,
    layer_prefix: &str,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<QuantizedExpertLayerPlan, ExpertManifestError> {
    let mut header_cache = HashMap::new();
    build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
        model_dir,
        weight_map,
        stored_tensor_name_by_canonical_name,
        layer_prefix,
        qwen3_5_config,
        &mut header_cache,
    )
}

/// Builds one layer while reusing headers retained by the complete pager construction pass.
///
/// Qwen sparse layers commonly share a small set of SafeTensors files. Keeping this cache outside
/// the per-layer loop guarantees each unique header is parsed once instead of once per layer.
pub(crate) fn build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
    model_dir: &Path,
    weight_map: &HashMap<String, String>,
    stored_tensor_name_by_canonical_name: &HashMap<String, String>,
    layer_prefix: &str,
    qwen3_5_config: &Qwen3_5Config,
    header_cache: &mut HashMap<PathBuf, SafetensorsHeader>,
) -> Result<QuantizedExpertLayerPlan, ExpertManifestError> {
    let mut tensor_sources = Vec::new();
    let mut projection_quantization_modes = Vec::new();
    let mut jangtq2_projection_rotations = Vec::new();

    for projection_name in PROJECTION_NAMES {
        let projection_module_name = format!("{layer_prefix}.switch_mlp.{projection_name}");
        let projection_quantization_profile =
            qwen3_5_config.quantization_profile_for_module(&projection_module_name);
        // Storage is a module fact. A model-wide affine document does not make a
        // native expert projection into zero-bit affine.
        let projection_storage =
            projection_storage_contract(&projection_module_name, projection_quantization_profile)?;
        if let ModuleWeightCodec::Jangtq2CodebookQuantized { rotation } =
            projection_quantization_profile.weight_codec
        {
            jangtq2_projection_rotations.push(((*projection_name).to_owned(), rotation));
        }
        projection_quantization_modes.push(projection_storage.quantization_mode);
        let mut projection_sources: HashMap<&str, Option<QuantizedTensorSource>> = HashMap::new();
        for parameter_name in projection_storage.parameter_names {
            let tensor_name =
                format!("{layer_prefix}.switch_mlp.{projection_name}.{parameter_name}");
            let source_file_name = weight_map.get(&tensor_name).ok_or_else(|| {
                ExpertManifestError::MissingTensorEntry {
                    tensor_name: tensor_name.clone(),
                }
            })?;
            let source_file = model_dir.join(source_file_name);
            let stored_tensor_name = stored_tensor_name_by_canonical_name
                .get(&tensor_name)
                .map_or(tensor_name.as_str(), String::as_str);
            let header = match header_cache.entry(source_file.clone()) {
                Entry::Occupied(cached_header_entry) => cached_header_entry.into_mut(),
                Entry::Vacant(vacant_header_entry) => vacant_header_entry.insert(
                    crate::expert_paging::safetensors_header::parse_safetensors_header(
                        &source_file,
                    )?,
                ),
            };
            let tensor_entry = header
                .tensor_entry_for_name(stored_tensor_name)
                .ok_or_else(|| ExpertManifestError::MissingShardTensor {
                    tensor_name: tensor_name.clone(),
                })?;
            let source = validate_quantized_tensor_source(
                &tensor_name,
                projection_name,
                parameter_name,
                projection_storage.quantization_bits,
                projection_storage.quantization_group_size,
                tensor_entry,
                &source_file,
                header.total_file_size_bytes,
            )?;
            projection_sources.insert(parameter_name, Some(source));
        }
        // Validate the source geometry specific to the storage format.
        let packed_parameter_name = match projection_storage.quantization_mode {
            QuantizationMode::Jangtq2Codebook => "tq2_packed",
            QuantizationMode::Affine | QuantizationMode::NativeBfloat16 => "weight",
        };
        let packed_source = projection_sources
            .get(packed_parameter_name)
            .and_then(Option::as_ref)
            .ok_or_else(|| ExpertManifestError::MissingShardTensor {
                tensor_name: format!(
                    "{layer_prefix}.switch_mlp.{projection_name}.{packed_parameter_name}"
                ),
            })?;
        match projection_storage.quantization_mode {
            QuantizationMode::Affine => {
                let scales_source = projection_sources
                    .get("scales")
                    .and_then(Option::as_ref)
                    .ok_or_else(|| ExpertManifestError::MissingShardTensor {
                        tensor_name: format!("{layer_prefix}.switch_mlp.{projection_name}.scales"),
                    })?;
                let biases_source = projection_sources
                    .get("biases")
                    .and_then(Option::as_ref)
                    .ok_or_else(|| ExpertManifestError::MissingShardTensor {
                        tensor_name: format!("{layer_prefix}.switch_mlp.{projection_name}.biases"),
                    })?;
                validate_projection_quantization_geometry(
                    projection_name,
                    packed_source,
                    scales_source,
                    biases_source,
                    projection_storage.quantization_bits,
                    projection_storage.quantization_group_size,
                )?;
            }
            QuantizationMode::Jangtq2Codebook => {
                let scales_source = projection_sources
                    .get("tq2_scales")
                    .and_then(Option::as_ref)
                    .ok_or_else(|| ExpertManifestError::MissingShardTensor {
                        tensor_name: format!(
                            "{layer_prefix}.switch_mlp.{projection_name}.tq2_scales"
                        ),
                    })?;
                validate_projection_codebook_geometry(
                    &projection_module_name,
                    projection_name,
                    packed_source,
                    scales_source,
                    projection_storage.quantization_bits,
                    projection_quantization_profile,
                    qwen3_5_config,
                )?;
            }
            QuantizationMode::NativeBfloat16 => {}
        }
        for parameter_name in projection_storage.parameter_names {
            let tensor_name =
                format!("{layer_prefix}.switch_mlp.{projection_name}.{parameter_name}");
            let tensor_source = projection_sources
                .remove(*parameter_name)
                .flatten()
                .ok_or(ExpertManifestError::MissingShardTensor { tensor_name })?;
            tensor_sources.push(tensor_source);
        }
    }

    let expert_capacities: std::collections::HashSet<usize> =
        tensor_sources.iter().map(|s| s.expert_capacity).collect();
    if expert_capacities.len() != 1 {
        return Err(ExpertManifestError::InconsistentExpertCapacity {
            found_capacities: expert_capacities.iter().copied().collect(),
        });
    }
    let expert_capacity = expert_capacities.iter().next().copied().ok_or(
        ExpertManifestError::InconsistentExpertCapacity {
            found_capacities: Vec::new(),
        },
    )?;
    let quantization_mode = layer_quantization_mode(&projection_quantization_modes);
    Ok(QuantizedExpertLayerPlan {
        layer_prefix: layer_prefix.to_owned(),
        tensor_sources,
        jangtq2_projection_rotations,
        expert_capacity,
        quantization_bits: match quantization_mode {
            QuantizationMode::Affine => i32::try_from(qwen3_5_config.default_quantization_bits())
                .map_err(|_| ExpertManifestError::InvalidBits)?,
            QuantizationMode::NativeBfloat16 | QuantizationMode::Jangtq2Codebook => 0,
        },
        quantization_group_size: match quantization_mode {
            QuantizationMode::Affine => {
                i32::try_from(qwen3_5_config.default_quantization_group_size())
                    .map_err(|_| ExpertManifestError::InvalidGroupSize)?
            }
            QuantizationMode::NativeBfloat16 | QuantizationMode::Jangtq2Codebook => 0,
        },
        quantization_mode,
    })
}

pub(crate) struct ProjectionStorageContract {
    pub(crate) quantization_bits: i32,
    pub(crate) quantization_group_size: i32,
    pub(crate) parameter_names: &'static [&'static str],
    pub(crate) quantization_mode: QuantizationMode,
}

pub(crate) fn projection_storage_contract(
    module_name: &str,
    projection_quantization_profile: OptiQQuantizationProfile,
) -> Result<ProjectionStorageContract, ExpertManifestError> {
    if projection_quantization_profile.is_unquantized() {
        return Ok(ProjectionStorageContract {
            quantization_bits: 0,
            quantization_group_size: 0,
            parameter_names: &["weight"],
            quantization_mode: QuantizationMode::NativeBfloat16,
        });
    }
    if projection_quantization_profile.is_jangtq2_codebook() {
        // Codebook storage has no group concept and no `weight`/`scales`/`biases`
        // trio: its on-disk tensors are the family's `tq2_packed` codes plus one
        // float16 `tq2_scales` row scale. The family owns the supported widths and
        // the decode; the plan only records the descriptor here.
        let codebook_bits = i32::try_from(projection_quantization_profile.bits)
            .map_err(|_| ExpertManifestError::InvalidBits)?;
        let ModuleWeightCodec::Jangtq2CodebookQuantized { rotation } =
            projection_quantization_profile.weight_codec
        else {
            return Err(ExpertManifestError::UnsupportedQuantizationMode {
                mode: format!("{:?}", projection_quantization_profile.weight_codec),
            });
        };
        let jangtq2_codebook_entry =
            astronomical_jangtq2_qwen35::configuration::Jangtq2WeightOverrideEntry::Codebook {
                mode: astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookMode,
                bits: projection_quantization_profile.bits,
                rotation,
            };
        jangtq2_codebook_entry.validate().map_err(|source| {
            ExpertManifestError::Jangtq2Contract {
                module_name: module_name.to_owned(),
                source,
            }
        })?;
        return Ok(ProjectionStorageContract {
            quantization_bits: codebook_bits,
            quantization_group_size: 0,
            parameter_names: &["tq2_packed", "tq2_scales"],
            quantization_mode: QuantizationMode::Jangtq2Codebook,
        });
    }
    let quantization_bits = i32::try_from(projection_quantization_profile.bits)
        .map_err(|_| ExpertManifestError::InvalidBits)?;
    let quantization_group_size = i32::try_from(projection_quantization_profile.group_size)
        .map_err(|_| ExpertManifestError::InvalidGroupSize)?;
    quantized_expert_validation::validate_quantization_contract(
        quantization_bits,
        quantization_group_size,
        QuantizationMode::Affine,
    )?;
    Ok(ProjectionStorageContract {
        quantization_bits,
        quantization_group_size,
        parameter_names: PARAMETER_NAMES,
        quantization_mode: QuantizationMode::Affine,
    })
}

pub(crate) fn layer_quantization_mode(
    projection_quantization_modes: &[QuantizationMode],
) -> QuantizationMode {
    if projection_quantization_modes
        .iter()
        .all(|quantization_mode| *quantization_mode == QuantizationMode::NativeBfloat16)
    {
        QuantizationMode::NativeBfloat16
    } else if projection_quantization_modes
        .iter()
        .all(|quantization_mode| *quantization_mode == QuantizationMode::Jangtq2Codebook)
    {
        QuantizationMode::Jangtq2Codebook
    } else {
        QuantizationMode::Affine
    }
}

// Keeping tensor metadata explicit makes startup validation failures easier to trace.
#[allow(clippy::too_many_arguments)]
fn validate_quantized_tensor_source(
    tensor_name: &str,
    projection_name: &str,
    parameter_name: &str,
    quantization_bits: i32,
    quantization_group_size: i32,
    tensor_entry: &TensorHeaderEntry,
    source_file: &Path,
    source_file_size_bytes: u64,
) -> Result<QuantizedTensorSource, ExpertManifestError> {
    let expected_exact_dtype = match (quantization_bits, parameter_name) {
        (0, _) => Some(SafetensorsDtype::BFloat16),
        (_, "weight" | "tq2_packed") => Some(SafetensorsDtype::Uint32),
        (_, "tq2_scales") => Some(SafetensorsDtype::Float16),
        _ => None,
    };
    if let Some(expected_dtype) = expected_exact_dtype
        && tensor_entry.dtype != expected_dtype
    {
        return Err(ExpertManifestError::WrongDtype {
            tensor_name: tensor_name.to_owned(),
            expected_dtype,
            actual_dtype: tensor_entry.dtype,
        });
    }
    if expected_exact_dtype.is_none()
        && !matches!(
            tensor_entry.dtype,
            SafetensorsDtype::Float16 | SafetensorsDtype::BFloat16 | SafetensorsDtype::Float32
        )
    {
        return Err(ExpertManifestError::UnsupportedAffineParameterDtype {
            tensor_name: tensor_name.to_owned(),
            actual_dtype: tensor_entry.dtype,
        });
    }
    let expected_tensor_rank = if parameter_name == "tq2_scales" { 2 } else { 3 };
    if tensor_entry.shape.len() != expected_tensor_rank || tensor_entry.shape.contains(&0) {
        return Err(ExpertManifestError::InvalidShape {
            tensor_name: tensor_name.to_owned(),
        });
    }
    let full_shape = tensor_entry.shape.clone();
    let dtype_byte_width = tensor_entry.dtype.byte_width();
    let bytes_per_expert = full_shape[1..]
        .iter()
        .try_fold(dtype_byte_width, |expert_tensor_bytes, tensor_dimension| {
            expert_tensor_bytes.checked_mul(*tensor_dimension)
        })
        .ok_or_else(|| ExpertManifestError::InvalidShape {
            tensor_name: tensor_name.to_owned(),
        })?;
    let expert_capacity = full_shape[0];
    // tensor_entry.data_start_offset is already file-relative
    // (converted from payload-relative during parse_safetensors_header).
    let tensor_payload_offset = tensor_entry.data_start_offset;

    Ok(QuantizedTensorSource {
        tensor_name: tensor_name.to_owned(),
        projection_name: projection_name.to_owned(),
        parameter_name: parameter_name.to_owned(),
        quantization_bits,
        quantization_group_size,
        source_file: source_file.to_path_buf(),
        source_file_size_bytes,
        dtype: tensor_entry.dtype,
        full_shape,
        tensor_payload_offset,
        bytes_per_expert,
        expert_capacity,
    })
}

pub(crate) fn validate_projection_quantization_geometry(
    projection_name: &str,
    weight_source: &QuantizedTensorSource,
    scales_source: &QuantizedTensorSource,
    biases_source: &QuantizedTensorSource,
    bits: i32,
    group_size: i32,
) -> Result<(), ExpertManifestError> {
    if scales_source.full_shape != biases_source.full_shape {
        return Err(ExpertManifestError::ScalesBiasesShapeMismatch {
            projection_name: projection_name.to_owned(),
        });
    }
    // Weight shape[:2] must match scales shape[:2] (expert_count × output_dim).
    if weight_source.full_shape[0..2] != scales_source.full_shape[0..2] {
        return Err(ExpertManifestError::WeightScalesBatchMismatch {
            projection_name: projection_name.to_owned(),
        });
    }
    // Packed width = scales_width * group_size * bits / 32.
    let scales_width = scales_source.full_shape[2];
    let expected_packed_width = scales_width * group_size as usize * bits as usize / 32;
    if weight_source.full_shape[2] != expected_packed_width {
        return Err(ExpertManifestError::InvalidPackedWidth {
            projection_name: projection_name.to_owned(),
            expected_packed_width,
            actual_packed_width: weight_source.full_shape[2],
        });
    }
    Ok(())
}

fn validate_projection_codebook_geometry(
    module_name: &str,
    projection_name: &str,
    packed_source: &QuantizedTensorSource,
    scales_source: &QuantizedTensorSource,
    bit_width: i32,
    quantization_profile: OptiQQuantizationProfile,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<(), ExpertManifestError> {
    let ModuleWeightCodec::Jangtq2CodebookQuantized { rotation } =
        quantization_profile.weight_codec
    else {
        return Err(ExpertManifestError::UnsupportedQuantizationMode {
            mode: format!("{:?}", quantization_profile.weight_codec),
        });
    };
    let bit_width = u32::try_from(bit_width).map_err(|_| ExpertManifestError::InvalidBits)?;
    let codebook_override_entry =
        astronomical_jangtq2_qwen35::configuration::Jangtq2WeightOverrideEntry::Codebook {
            mode: astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookMode,
            bits: bit_width,
            rotation,
        };
    codebook_override_entry
        .validate()
        .map_err(|source| ExpertManifestError::Jangtq2Contract {
            module_name: module_name.to_owned(),
            source,
        })?;

    let (input_width, output_width) = if projection_name == "down_proj" {
        (
            qwen3_5_config.expert_intermediate_size(),
            qwen3_5_config.hidden_size(),
        )
    } else {
        (
            qwen3_5_config.hidden_size(),
            qwen3_5_config.expert_intermediate_size(),
        )
    };
    let module_geometry = astronomical_jangtq2_qwen35::artifacts::Jangtq2ModuleGeometry {
        input_width,
        output_width,
        expert_count: qwen3_5_config.expert_count(),
        affine_group_size: qwen3_5_config.default_quantization_group_size(),
    };
    let codebook_tensor_inventory =
        astronomical_jangtq2_qwen35::artifacts::tensor_inventory_for_module(
            module_name,
            &codebook_override_entry,
            &module_geometry,
        )
        .map_err(|source| ExpertManifestError::Jangtq2Contract {
            module_name: module_name.to_owned(),
            source,
        })?;
    for (source_tensor, expected_tensor) in [
        (
            packed_source,
            &codebook_tensor_inventory.expected_tensors[0],
        ),
        (
            scales_source,
            &codebook_tensor_inventory.expected_tensors[1],
        ),
    ] {
        if source_tensor.full_shape != expected_tensor.shape {
            return Err(ExpertManifestError::Jangtq2TensorShapeMismatch {
                module_name: module_name.to_owned(),
                tensor_name: expected_tensor.tensor_name.clone(),
                expected_shape: expected_tensor.shape.clone(),
                actual_shape: source_tensor.full_shape.clone(),
            });
        }
    }
    Ok(())
}
