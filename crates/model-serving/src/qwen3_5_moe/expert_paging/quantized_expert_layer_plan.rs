//! Startup-validated layer plan construction and source manifest building.
//!
//! `build_quantized_expert_layer_plan()` reads safetensors headers at startup
//! once per layer and validates all tensor geometry. `build_source_manifests()`
//! groups selected expert intervals by shard file for bounded native reads.
//! `contiguous_selected_runs()` groups adjacent selected experts for efficient
//! sequential I/O.

use std::collections::{HashMap, hash_map::Entry};
use std::path::{Path, PathBuf};

use super::quantized_expert_manifest::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertShardManifest,
    QuantizedExpertSourceInterval, QuantizedExpertTensorRange, QuantizedTensorSource,
};
use super::quantized_expert_validation::{
    validate_quantization_contract, validate_source_intervals, validate_virtual_intervals,
};
use super::safetensors_header::{SafetensorsDtype, SafetensorsHeader, TensorHeaderEntry};
use crate::qwen3_5_moe::Qwen3_5MoEConfig;

/// The three MoE projection names in an expert's SwitchMLP block.
const PROJECTION_NAMES: &[&str] = &["gate_proj", "up_proj", "down_proj"];

/// The three quantized parameter names for each projection.
const PARAMETER_NAMES: &[&str] = &["weight", "scales", "biases"];

/// Expected dtype for each quantized parameter.
/// Weight tensors are packed U32, scales and biases are BF16.
const EXPECTED_PARAMETER_DTYPES: &[(&str, SafetensorsDtype)] = &[
    ("weight", SafetensorsDtype::Uint32),
    ("scales", SafetensorsDtype::BFloat16),
    ("biases", SafetensorsDtype::BFloat16),
];

fn expected_dtype_for_parameter(parameter_name: &str) -> Option<SafetensorsDtype> {
    EXPECTED_PARAMETER_DTYPES
        .iter()
        .find(|(name, _)| *name == parameter_name)
        .map(|(_, dtype)| *dtype)
}

/// Build startup-validated layer metadata by reading safetensors headers.
///
/// `weight_map` maps tensor names (e.g. `language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight`)
/// to shard file names (e.g. `model-00001-of-00005.safetensors`). The model directory
/// contains the actual shard files.
pub fn build_quantized_expert_layer_plan(
    model_dir: &std::path::Path,
    weight_map: &HashMap<String, String>,
    layer_prefix: &str,
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
    quantization_mode: QuantizationMode,
) -> Result<QuantizedExpertLayerPlan, ExpertManifestError> {
    let mut tensor_sources = Vec::new();
    let mut header_cache: HashMap<PathBuf, SafetensorsHeader> = HashMap::new();

    for projection_name in PROJECTION_NAMES {
        let projection_module_name = format!("{layer_prefix}.switch_mlp.{projection_name}");
        let projection_quantization_profile =
            qwen3_5_moe_config.quantization_profile_for_module(&projection_module_name);
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
            let header = match header_cache.entry(source_file.clone()) {
                Entry::Occupied(cached_header_entry) => cached_header_entry.into_mut(),
                Entry::Vacant(vacant_header_entry) => vacant_header_entry.insert(
                    super::safetensors_header::parse_safetensors_header(&source_file)?,
                ),
            };
            let tensor_entry = header.tensor_entry_for_name(&tensor_name).ok_or_else(|| {
                ExpertManifestError::MissingShardTensor {
                    tensor_name: tensor_name.clone(),
                }
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
            QuantizationMode::Affine => {
                i32::try_from(qwen3_5_moe_config.default_quantization_bits())
                    .map_err(|_| ExpertManifestError::InvalidBits)?
            }
            QuantizationMode::NativeBfloat16 => 0,
        },
        quantization_group_size: match quantization_mode {
            QuantizationMode::Affine => {
                i32::try_from(qwen3_5_moe_config.default_quantization_group_size())
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
    let expected_dtype = if quantization_bits == 0 {
        SafetensorsDtype::BFloat16
    } else {
        expected_dtype_for_parameter(parameter_name).ok_or_else(|| {
            ExpertManifestError::WrongDtype {
                tensor_name: tensor_name.to_owned(),
                expected_dtype: SafetensorsDtype::BFloat16,
                actual_dtype: tensor_entry.dtype,
            }
        })?
    };
    if tensor_entry.dtype != expected_dtype {
        return Err(ExpertManifestError::WrongDtype {
            tensor_name: tensor_name.to_owned(),
            expected_dtype,
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

/// Groups source intervals by shard file and builds compact shard manifests
/// for bounded native reads. Each manifest describes which byte ranges to
/// read from which source file, and how they map to the virtual payload.
pub fn build_source_manifests(
    tensor_sources: &[QuantizedTensorSource],
    selected_expert_ids: &[usize],
) -> Result<Vec<QuantizedExpertShardManifest>, ExpertManifestError> {
    let mut source_files_set: std::collections::BTreeSet<PathBuf> =
        std::collections::BTreeSet::new();
    for source in tensor_sources {
        source_files_set.insert(source.source_file.clone());
    }
    let source_files: Vec<PathBuf> = source_files_set.into_iter().collect();
    let mut source_manifests = Vec::with_capacity(source_files.len());

    for source_file in &source_files {
        let mut tensor_ranges = Vec::new();
        let mut source_intervals = Vec::new();
        let mut virtual_payload_offset: u64 = 0;

        for tensor_source in tensor_sources {
            if tensor_source.source_file != *source_file {
                continue;
            }
            let selected_tensor_byte_count =
                selected_expert_ids.len() * tensor_source.bytes_per_expert;
            tensor_ranges.push(QuantizedExpertTensorRange {
                tensor_name: format!(
                    "{}.{}",
                    tensor_source.projection_name, tensor_source.parameter_name
                ),
                projection_name: tensor_source.projection_name.clone(),
                parameter_name: tensor_source.parameter_name.clone(),
                dtype: tensor_source.dtype,
                shape: {
                    let mut shape = vec![selected_expert_ids.len()];
                    shape.extend(&tensor_source.full_shape[1..]);
                    shape
                },
                virtual_payload_offset,
                byte_count: selected_tensor_byte_count,
            });
            for (expert_start, expert_count, first_page_slot) in
                contiguous_selected_runs(selected_expert_ids)
            {
                source_intervals.push(QuantizedExpertSourceInterval {
                    tensor_name: tensor_source.tensor_name.clone(),
                    expert_start,
                    expert_count,
                    source_file_offset: tensor_source.tensor_payload_offset
                        + expert_start as u64 * tensor_source.bytes_per_expert as u64,
                    source_byte_count: expert_count * tensor_source.bytes_per_expert,
                    virtual_payload_offset: virtual_payload_offset
                        + first_page_slot as u64 * tensor_source.bytes_per_expert as u64,
                });
            }
            virtual_payload_offset += selected_tensor_byte_count as u64;
        }

        let mut ordered_source_intervals = source_intervals;
        ordered_source_intervals.sort_by_key(|interval| interval.source_file_offset);

        validate_source_intervals(&ordered_source_intervals, 0)?;
        validate_virtual_intervals(&ordered_source_intervals, virtual_payload_offset)?;

        source_manifests.push(QuantizedExpertShardManifest {
            source_file: source_file.clone(),
            tensor_ranges,
            source_intervals: ordered_source_intervals,
            payload_byte_count: virtual_payload_offset,
        });
    }
    Ok(source_manifests)
}

/// Groups adjacent selected experts while retaining their first page slot.
pub fn contiguous_selected_runs(expert_ids: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    if expert_ids.is_empty() {
        return runs;
    }
    let mut run_start = expert_ids[0];
    let mut first_page_slot = 0;
    for page_slot in 1..=expert_ids.len() {
        let run_is_complete = page_slot == expert_ids.len();
        if !run_is_complete && expert_ids[page_slot] == expert_ids[page_slot - 1] + 1 {
            continue;
        }
        let run_end = expert_ids[page_slot - 1];
        runs.push((run_start, run_end - run_start + 1, first_page_slot));
        if !run_is_complete {
            run_start = expert_ids[page_slot];
            first_page_slot = page_slot;
        }
    }
    runs
}
