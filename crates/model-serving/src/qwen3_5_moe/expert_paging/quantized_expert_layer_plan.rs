//! Startup-validated layer plan construction and source manifest building.
//!
//! Complete pager construction shares one SafeTensors header cache across every layer and validates
//! all tensor geometry. Standalone callers retain a one-layer cache. `build_source_manifests()`
//! groups selected expert intervals by shard file for bounded native reads.
//! `contiguous_selected_runs()` groups adjacent selected experts for efficient
//! sequential I/O.

use std::collections::{HashMap, hash_map::Entry};
use std::path::{Path, PathBuf};

use crate::expert_paging::quantized_expert_manifest::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource,
};
use crate::expert_paging::quantized_expert_validation::validate_quantization_contract;
use crate::expert_paging::safetensors_header::{
    SafetensorsDtype, SafetensorsHeader, TensorHeaderEntry,
};
use crate::qwen3_5::Qwen3_5Config;

/// The three MoE projection names in an expert's SwitchMLP block.
const PROJECTION_NAMES: &[&str] = &["gate_proj", "up_proj", "down_proj"];

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
    quantization_mode: QuantizationMode,
) -> Result<QuantizedExpertLayerPlan, ExpertManifestError> {
    build_quantized_expert_layer_plan_with_stored_names(
        model_dir,
        weight_map,
        &HashMap::new(),
        layer_prefix,
        qwen3_5_config,
        quantization_mode,
    )
}

pub(crate) fn build_quantized_expert_layer_plan_with_stored_names(
    model_dir: &std::path::Path,
    weight_map: &HashMap<String, String>,
    stored_tensor_name_by_canonical_name: &HashMap<String, String>,
    layer_prefix: &str,
    qwen3_5_config: &Qwen3_5Config,
    quantization_mode: QuantizationMode,
) -> Result<QuantizedExpertLayerPlan, ExpertManifestError> {
    let mut header_cache = HashMap::new();
    build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
        model_dir,
        weight_map,
        stored_tensor_name_by_canonical_name,
        layer_prefix,
        qwen3_5_config,
        quantization_mode,
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
    quantization_mode: QuantizationMode,
    header_cache: &mut HashMap<PathBuf, SafetensorsHeader>,
) -> Result<QuantizedExpertLayerPlan, ExpertManifestError> {
    let mut tensor_sources = Vec::new();

    for projection_name in PROJECTION_NAMES {
        let projection_module_name = format!("{layer_prefix}.switch_mlp.{projection_name}");
        let projection_quantization_profile =
            qwen3_5_config.quantization_profile_for_module(&projection_module_name);
        let (quantization_bits, quantization_group_size, parameter_names): (i32, i32, &[&str]) =
            match quantization_mode {
                QuantizationMode::Affine => {
                    let quantization_bits = i32::try_from(projection_quantization_profile.bits)
                        .map_err(|_| ExpertManifestError::InvalidBits)?;
                    let quantization_group_size =
                        i32::try_from(projection_quantization_profile.group_size)
                            .map_err(|_| ExpertManifestError::InvalidGroupSize)?;
                    validate_quantization_contract(
                        quantization_bits,
                        quantization_group_size,
                        quantization_mode,
                    )?;
                    (quantization_bits, quantization_group_size, PARAMETER_NAMES)
                }
                QuantizationMode::NativeBfloat16 => (0, 0, &["weight"]),
            };
        let mut projection_sources: HashMap<&str, Option<QuantizedTensorSource>> = HashMap::new();
        for parameter_name in parameter_names {
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
                quantization_bits,
                quantization_group_size,
                tensor_entry,
                &source_file,
                header.total_file_size_bytes,
            )?;
            projection_sources.insert(parameter_name, Some(source));
        }
        // Validate the source geometry specific to the storage format.
        let weight_source = projection_sources
            .get("weight")
            .and_then(Option::as_ref)
            .ok_or_else(|| ExpertManifestError::MissingShardTensor {
                tensor_name: format!("{layer_prefix}.switch_mlp.{projection_name}.weight"),
            })?;
        match quantization_mode {
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
                    weight_source,
                    scales_source,
                    biases_source,
                    quantization_bits,
                    quantization_group_size,
                )?;
            }
            QuantizationMode::NativeBfloat16 => {}
        }
        for parameter_name in parameter_names {
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
    Ok(QuantizedExpertLayerPlan {
        layer_prefix: layer_prefix.to_owned(),
        tensor_sources,
        expert_capacity,
        quantization_bits: match quantization_mode {
            QuantizationMode::Affine => i32::try_from(qwen3_5_config.default_quantization_bits())
                .map_err(|_| ExpertManifestError::InvalidBits)?,
            QuantizationMode::NativeBfloat16 => 0,
        },
        quantization_group_size: match quantization_mode {
            QuantizationMode::Affine => {
                i32::try_from(qwen3_5_config.default_quantization_group_size())
                    .map_err(|_| ExpertManifestError::InvalidGroupSize)?
            }
            QuantizationMode::NativeBfloat16 => 0,
        },
        quantization_mode,
    })
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
    let expected_exact_dtype = if quantization_bits == 0 {
        Some(SafetensorsDtype::BFloat16)
    } else if parameter_name == "weight" {
        Some(SafetensorsDtype::Uint32)
    } else {
        None
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
    if tensor_entry.shape.len() != 3 || tensor_entry.shape.contains(&0) {
        return Err(ExpertManifestError::InvalidShape {
            tensor_name: tensor_name.to_owned(),
        });
    }
    let full_shape = tensor_entry.shape.clone();
    let dtype_byte_width = tensor_entry.dtype.byte_width();
    let bytes_per_expert: usize = full_shape[1] * full_shape[2] * dtype_byte_width;
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

fn validate_projection_quantization_geometry(
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
