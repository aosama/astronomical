//! Layer-plan construction for converted per-expert streaming revisions.
//!
//! Shard-backed revisions derive expert layer plans from the safetensors
//! index; a shard-less streaming revision derives them from the per-expert
//! pack headers instead. One spot-checked header per layer binds real bytes
//! to the format, every declared pack file is size-validated against the
//! plan-derived geometry, and config stays authoritative for quantization:
//! any header/plan disagreement fails the load instead of serving garbage
//! weights.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::streaming_expert_packs::{
    STREAMING_MANIFEST_FORMAT_VERSION, StreamingExpertPackError, StreamingExpertPackSources,
    StreamingManifestProbe, read_pack_header_probe, read_streaming_manifest_probe,
    validated_layer_expert_paths,
};
use crate::expert_paging::QuantizationMode;
use crate::expert_paging::{QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype};
use crate::qwen3_5::Qwen3_5Config;
use crate::qwen3_5_moe::expert_paging::quantized_expert_layer_plan::{
    PROJECTION_NAMES, layer_quantization_mode, projection_storage_contract,
    validate_projection_quantization_geometry,
};

/// Builds one layer plan per decoder layer directly from the per-expert pack
/// headers, then validates the complete manifest inventory against that
/// geometry. Config stays authoritative for quantization: every header field
/// must agree with the module profile the shard path would derive, so a
/// mis-converted pack fails the load instead of serving garbage weights.
pub fn build_streaming_expert_layer_plans(
    model_dir: &Path,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<(Vec<QuantizedExpertLayerPlan>, StreamingExpertPackSources), StreamingExpertPackError> {
    let manifest: StreamingManifestProbe =
        read_streaming_manifest_probe(&model_dir.join("manifest.json"))?;
    if manifest.format_version != STREAMING_MANIFEST_FORMAT_VERSION {
        return Err(StreamingExpertPackError::UnsupportedFormatVersion {
            manifest_path: model_dir.join("manifest.json"),
            actual_format_version: manifest.format_version,
        });
    }
    if manifest.expert_capacity == 0 {
        return Err(StreamingExpertPackError::PackHeaderGeometry {
            description: "the streaming manifest declares an empty expert capacity".to_owned(),
        });
    }

    let mut declared_files_by_layer: Vec<Vec<(usize, std::path::PathBuf, u64)>> = Vec::new();
    for expert_file in &manifest.expert_files {
        while declared_files_by_layer.len() <= expert_file.layer_index {
            declared_files_by_layer.push(Vec::new());
        }
        declared_files_by_layer[expert_file.layer_index].push((
            expert_file.expert_id,
            model_dir.join(&expert_file.file_name),
            expert_file.expected_file_byte_count,
        ));
    }
    let manifest_layer_count = declared_files_by_layer.len();
    let decoder_layer_count = qwen3_5_config.layer_count() as usize;
    if manifest_layer_count != decoder_layer_count {
        return Err(StreamingExpertPackError::ManifestLayerCountDisagreement {
            manifest_layer_count,
            decoder_layer_count,
        });
    }

    let mut layer_plans = Vec::with_capacity(decoder_layer_count);
    let mut expert_file_paths_by_layer = Vec::with_capacity(decoder_layer_count);
    for (layer_index, declared_files) in declared_files_by_layer.iter().enumerate() {
        let expert_zero_path = declared_files
            .iter()
            .find(|(expert_id, _, _)| *expert_id == 0)
            .map(|(_, expert_file_path, _)| expert_file_path.clone())
            .ok_or(StreamingExpertPackError::MissingExpertFile {
                layer_index,
                expert_id: 0,
            })?;
        let layer_plan = build_layer_plan_from_pack_header(
            qwen3_5_config,
            layer_index,
            &expert_zero_path,
            manifest.expert_capacity,
        )?;
        expert_file_paths_by_layer.push(validated_layer_expert_paths(
            layer_index,
            declared_files,
            layer_plan.expert_capacity,
            super::streaming_expert_packs::compute_expert_file_byte_count(&layer_plan)?,
        )?);
        layer_plans.push(layer_plan);
    }

    // Full inventory validation against the built plans: every declared file
    // exists with the geometry-derived byte count and one header binds the
    // bytes to the format.
    let streaming_sources = detect_streaming_expert_pack_sources(model_dir, &layer_plans)?
        .ok_or_else(|| StreamingExpertPackError::PackHeaderGeometry {
            description: "the streaming manifest disappeared during plan construction".to_owned(),
        })?;
    Ok((layer_plans, streaming_sources))
}

use super::streaming_expert_packs::detect_streaming_expert_pack_sources;

/// Builds one layer plan from that layer's expert-zero pack header.
///
/// The header carries the converted tensor layout in order; each descriptor
/// must match the config-derived projection/parameter sequence exactly, so a
/// pack converted from a different model or quantization contract is
/// rejected here rather than served with crossed weights.
fn build_layer_plan_from_pack_header(
    qwen3_5_config: &Qwen3_5Config,
    layer_index: usize,
    expert_zero_path: &Path,
    expert_capacity: usize,
) -> Result<QuantizedExpertLayerPlan, StreamingExpertPackError> {
    let layer_prefix = format!("language_model.model.layers.{layer_index}.mlp");
    let pack_header = read_pack_header_probe(expert_zero_path)?;
    if pack_header.format_version != STREAMING_MANIFEST_FORMAT_VERSION
        || pack_header.layer_index != layer_index
        || pack_header.expert_id != 0
    {
        return Err(StreamingExpertPackError::PackHeaderIdentity {
            expert_file_path: expert_zero_path.to_path_buf(),
        });
    }
    if pack_header.expert_capacity != expert_capacity {
        return Err(StreamingExpertPackError::PackHeaderGeometry {
            description: format!(
                "layer {layer_index} pack header declares capacity {} but the manifest declares {expert_capacity}",
                pack_header.expert_capacity
            ),
        });
    }

    let expert_zero_file_size_bytes = fs::metadata(expert_zero_path)
        .map_err(|source| StreamingExpertPackError::ExpertFileOpen {
            expert_file_path: expert_zero_path.to_path_buf(),
            source,
        })?
        .len();
    let mut tensor_sources = Vec::with_capacity(pack_header.tensor_descriptors.len());
    let mut projection_quantization_modes = Vec::with_capacity(PROJECTION_NAMES.len());
    let mut descriptor_cursor = 0_usize;
    for projection_name in PROJECTION_NAMES {
        let module_name = format!("{layer_prefix}.switch_mlp.{projection_name}");
        let projection_quantization_profile =
            qwen3_5_config.quantization_profile_for_module(&module_name);
        let projection_storage = projection_storage_contract(projection_quantization_profile)
            .map_err(|error| StreamingExpertPackError::PackHeaderGeometry {
                description: error.to_string(),
            })?;
        projection_quantization_modes.push(projection_storage.quantization_mode);
        let mut projection_sources = HashMap::new();
        for parameter_name in projection_storage.parameter_names {
            let expected_tensor_name = format!("{module_name}.{parameter_name}");
            let descriptor =
                pack_header
                    .tensor_descriptors
                    .get(descriptor_cursor)
                    .ok_or_else(|| StreamingExpertPackError::PackHeaderGeometry {
                        description: format!(
                            "layer {layer_index} pack header declares {} tensor descriptors, but the config-derived layout needs at least {}",
                            pack_header.tensor_descriptors.len(),
                            descriptor_cursor + 1
                        ),
                    })?;
            descriptor_cursor += 1;
            if descriptor.projection_name != *projection_name
                || descriptor.parameter_name != *parameter_name
                || descriptor.tensor_name != expected_tensor_name
            {
                return Err(StreamingExpertPackError::PackHeaderGeometry {
                    description: format!(
                        "layer {layer_index} pack descriptor {} does not match the config-derived layout {expected_tensor_name}",
                        descriptor.tensor_name
                    ),
                });
            }
            if descriptor.bytes_per_expert == 0 || descriptor.expert_local_shape.is_empty() {
                return Err(StreamingExpertPackError::PackHeaderGeometry {
                    description: format!(
                        "layer {layer_index} pack descriptor {expected_tensor_name} declares empty geometry"
                    ),
                });
            }
            let dtype = SafetensorsDtype::from_dtype_name(&descriptor.dtype_name).ok_or_else(
                || StreamingExpertPackError::PackHeaderGeometry {
                    description: format!(
                        "layer {layer_index} pack descriptor {expected_tensor_name} declares unsupported dtype {}",
                        descriptor.dtype_name
                    ),
                },
            )?;
            let mut full_shape = vec![expert_capacity];
            full_shape.extend(&descriptor.expert_local_shape);
            projection_sources.insert(
                *parameter_name,
                QuantizedTensorSource {
                    tensor_name: expected_tensor_name.clone(),
                    projection_name: (*projection_name).to_owned(),
                    parameter_name: (*parameter_name).to_owned(),
                    quantization_bits: projection_storage.quantization_bits,
                    quantization_group_size: projection_storage.quantization_group_size,
                    // The plan source points at this layer's expert-zero pack so
                    // descriptor retention opens real files; routed pages resolve
                    // per-expert paths through the streaming sources instead.
                    source_file: expert_zero_path.to_path_buf(),
                    source_file_size_bytes: expert_zero_file_size_bytes,
                    dtype,
                    full_shape,
                    tensor_payload_offset: descriptor.pack_segment_offset_bytes,
                    bytes_per_expert: descriptor.bytes_per_expert,
                    expert_capacity,
                },
            );
        }
        if projection_storage.quantization_mode == QuantizationMode::Affine {
            let weight_source = projection_sources
                .get("weight")
                .expect("affine storage always declares a weight");
            let scales_source = projection_sources
                .get("scales")
                .expect("affine storage always declares scales");
            let biases_source = projection_sources
                .get("biases")
                .expect("affine storage always declares biases");
            validate_projection_quantization_geometry(
                projection_name,
                weight_source,
                scales_source,
                biases_source,
                projection_storage.quantization_bits,
                projection_storage.quantization_group_size,
            )
            .map_err(|error| StreamingExpertPackError::PackHeaderGeometry {
                description: error.to_string(),
            })?;
        }
        for parameter_name in projection_storage.parameter_names {
            let assembled_source = projection_sources
                .remove(*parameter_name)
                .expect("each declared parameter was built above");
            tensor_sources.push(assembled_source);
        }
    }
    let quantization_mode = layer_quantization_mode(&projection_quantization_modes);
    let plan_quantization_bits = match quantization_mode {
        QuantizationMode::Affine => i32::try_from(qwen3_5_config.default_quantization_bits())
            .map_err(|_| StreamingExpertPackError::PackHeaderGeometry {
                description: "default quantization bits exceed the i32 range".to_owned(),
            })?,
        QuantizationMode::NativeBfloat16 => 0,
    };
    let plan_quantization_group_size = match quantization_mode {
        QuantizationMode::Affine => i32::try_from(qwen3_5_config.default_quantization_group_size())
            .map_err(|_| StreamingExpertPackError::PackHeaderGeometry {
                description: "default quantization group size exceeds the i32 range".to_owned(),
            })?,
        QuantizationMode::NativeBfloat16 => 0,
    };
    if pack_header.quantization_bits != plan_quantization_bits
        || pack_header.quantization_group_size != plan_quantization_group_size
    {
        return Err(StreamingExpertPackError::PackHeaderGeometry {
            description: format!(
                "layer {layer_index} pack header quantization ({}, {}) disagrees with the config-derived plan ({}, {})",
                pack_header.quantization_bits,
                pack_header.quantization_group_size,
                plan_quantization_bits,
                plan_quantization_group_size
            ),
        });
    }
    Ok(QuantizedExpertLayerPlan {
        layer_prefix: layer_prefix.clone(),
        tensor_sources,
        expert_capacity,
        quantization_bits: plan_quantization_bits,
        quantization_group_size: plan_quantization_group_size,
        quantization_mode,
    })
}
