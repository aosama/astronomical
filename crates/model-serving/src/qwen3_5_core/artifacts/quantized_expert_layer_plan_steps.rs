//! Named planning steps for one quantized expert decoder layer.
//!
//! Split out of `quantized_expert_layer_plan.rs` so each unit stays under
//! the workspace's 100-code-line function budget: this module owns the
//! per-projection pipeline (storage contract, shard resolution, geometry
//! validation, drain) and the final cross-projection plan assembly.

use std::collections::{HashMap, hash_map::Entry};
use std::path::{Path, PathBuf};

use crate::expert_paging::quantized_expert_manifest::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource,
};
use crate::expert_paging::safetensors_header::SafetensorsHeader;
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::quantizations::mlx_quantization::{
    ModuleQuantizationProfile, ModuleWeightCodec,
};

use super::{
    ProjectionStorageContract, layer_quantization_mode, projection_storage_contract,
    validate_projection_codebook_geometry, validate_projection_quantization_geometry,
    validate_quantized_tensor_source,
};

/// Inputs every projection planning step shares for one decoder layer.
pub(super) struct ProjectionPlanInputs<'a> {
    pub(super) model_dir: &'a Path,
    pub(super) weight_map: &'a HashMap<String, String>,
    pub(super) stored_tensor_name_by_canonical_name: &'a HashMap<String, String>,
    pub(super) layer_prefix: &'a str,
    pub(super) qwen3_5_config: &'a Qwen3_5Config,
}

/// Everything one projection contributes to its layer's plan.
pub(super) struct ProjectionSourceCollection {
    pub(super) tensor_sources: Vec<QuantizedTensorSource>,
    pub(super) quantization_mode: QuantizationMode,
    pub(super) jangtq2_rotation:
        Option<astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation>,
}

/// Plans one projection: storage contract, shard sources, and geometry.
pub(super) fn build_projection_source_collection(
    plan_inputs: &ProjectionPlanInputs<'_>,
    projection_name: &str,
    header_cache: &mut HashMap<PathBuf, SafetensorsHeader>,
) -> Result<ProjectionSourceCollection, ExpertManifestError> {
    let projection_module_name =
        format!("{}.switch_mlp.{projection_name}", plan_inputs.layer_prefix);
    let projection_quantization_profile = plan_inputs
        .qwen3_5_config
        .quantization_profile_for_module(&projection_module_name);
    // Storage is a module fact. A model-wide affine document does not make a
    // native expert projection into zero-bit affine.
    let projection_storage =
        projection_storage_contract(&projection_module_name, projection_quantization_profile)?;
    let jangtq2_rotation = if let ModuleWeightCodec::Jangtq2CodebookQuantized { rotation } =
        projection_quantization_profile.weight_codec
    {
        Some(rotation)
    } else {
        None
    };
    let projection_sources = resolve_projection_tensor_sources(
        plan_inputs,
        projection_name,
        &projection_storage,
        header_cache,
    )?;
    // Validate the source geometry specific to the storage format.
    validate_projection_storage_geometry(
        plan_inputs,
        &projection_module_name,
        projection_name,
        &projection_sources,
        &projection_storage,
        projection_quantization_profile,
    )?;
    Ok(ProjectionSourceCollection {
        tensor_sources: drain_projection_tensor_sources(
            plan_inputs,
            projection_name,
            projection_sources,
            &projection_storage,
        )?,
        quantization_mode: projection_storage.quantization_mode,
        jangtq2_rotation,
    })
}

/// Resolves every stored parameter tensor of one projection against its shard.
fn resolve_projection_tensor_sources(
    plan_inputs: &ProjectionPlanInputs<'_>,
    projection_name: &str,
    projection_storage: &ProjectionStorageContract,
    header_cache: &mut HashMap<PathBuf, SafetensorsHeader>,
) -> Result<HashMap<&'static str, Option<QuantizedTensorSource>>, ExpertManifestError> {
    let mut projection_sources: HashMap<&'static str, Option<QuantizedTensorSource>> =
        HashMap::new();
    for parameter_name in projection_storage.parameter_names {
        let tensor_name = format!(
            "{}.switch_mlp.{projection_name}.{parameter_name}",
            plan_inputs.layer_prefix
        );
        let source_file_name = plan_inputs.weight_map.get(&tensor_name).ok_or_else(|| {
            ExpertManifestError::MissingTensorEntry {
                tensor_name: tensor_name.clone(),
            }
        })?;
        let source_file = plan_inputs.model_dir.join(source_file_name);
        let stored_tensor_name = plan_inputs
            .stored_tensor_name_by_canonical_name
            .get(&tensor_name)
            .map_or(tensor_name.as_str(), String::as_str);
        let header = match header_cache.entry(source_file.clone()) {
            Entry::Occupied(cached_header_entry) => cached_header_entry.into_mut(),
            Entry::Vacant(vacant_header_entry) => vacant_header_entry.insert(
                crate::expert_paging::safetensors_header::parse_safetensors_header(&source_file)?,
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
    Ok(projection_sources)
}

/// Validates the packed/scales geometry required by the projection's storage format.
fn validate_projection_storage_geometry(
    plan_inputs: &ProjectionPlanInputs<'_>,
    projection_module_name: &str,
    projection_name: &str,
    projection_sources: &HashMap<&'static str, Option<QuantizedTensorSource>>,
    projection_storage: &ProjectionStorageContract,
    projection_quantization_profile: ModuleQuantizationProfile,
) -> Result<(), ExpertManifestError> {
    let packed_parameter_name = match projection_storage.quantization_mode {
        QuantizationMode::Jangtq2Codebook => "tq2_packed",
        QuantizationMode::Affine | QuantizationMode::NativeBfloat16 => "weight",
    };
    let packed_source = projection_sources
        .get(packed_parameter_name)
        .and_then(Option::as_ref)
        .ok_or_else(|| ExpertManifestError::MissingShardTensor {
            tensor_name: format!(
                "{}.switch_mlp.{projection_name}.{packed_parameter_name}",
                plan_inputs.layer_prefix
            ),
        })?;
    match projection_storage.quantization_mode {
        QuantizationMode::Affine => {
            let scales_source = projection_sources
                .get("scales")
                .and_then(Option::as_ref)
                .ok_or_else(|| ExpertManifestError::MissingShardTensor {
                    tensor_name: format!(
                        "{}.switch_mlp.{projection_name}.scales",
                        plan_inputs.layer_prefix
                    ),
                })?;
            let biases_source = projection_sources
                .get("biases")
                .and_then(Option::as_ref)
                .ok_or_else(|| ExpertManifestError::MissingShardTensor {
                    tensor_name: format!(
                        "{}.switch_mlp.{projection_name}.biases",
                        plan_inputs.layer_prefix
                    ),
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
                        "{}.switch_mlp.{projection_name}.tq2_scales",
                        plan_inputs.layer_prefix
                    ),
                })?;
            validate_projection_codebook_geometry(
                projection_module_name,
                projection_name,
                packed_source,
                scales_source,
                projection_storage.quantization_bits,
                projection_quantization_profile,
                plan_inputs.qwen3_5_config,
            )?;
        }
        QuantizationMode::NativeBfloat16 => {}
    }
    Ok(())
}

/// Flattens the projection's per-parameter sources into the layer tensor list.
///
/// The storage contract's parameter order is the plan's tensor order, so the
/// pager reads each projection's tensors in a stable sequence.
pub(super) fn drain_projection_tensor_sources(
    plan_inputs: &ProjectionPlanInputs<'_>,
    projection_name: &str,
    mut projection_sources: HashMap<&'static str, Option<QuantizedTensorSource>>,
    projection_storage: &ProjectionStorageContract,
) -> Result<Vec<QuantizedTensorSource>, ExpertManifestError> {
    let mut tensor_sources = Vec::new();
    for parameter_name in projection_storage.parameter_names {
        let tensor_name = format!(
            "{}.switch_mlp.{projection_name}.{parameter_name}",
            plan_inputs.layer_prefix
        );
        let tensor_source = projection_sources
            .remove(*parameter_name)
            .flatten()
            .ok_or(ExpertManifestError::MissingShardTensor { tensor_name })?;
        tensor_sources.push(tensor_source);
    }
    Ok(tensor_sources)
}

/// Validates cross-projection agreement and assembles the final layer plan.
pub(super) fn assemble_expert_layer_plan(
    plan_inputs: &ProjectionPlanInputs<'_>,
    tensor_sources: Vec<QuantizedTensorSource>,
    projection_quantization_modes: Vec<QuantizationMode>,
    jangtq2_projection_rotations: Vec<(
        String,
        astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
    )>,
) -> Result<QuantizedExpertLayerPlan, ExpertManifestError> {
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
        layer_prefix: plan_inputs.layer_prefix.to_owned(),
        tensor_sources,
        jangtq2_projection_rotations,
        expert_capacity,
        quantization_bits: match quantization_mode {
            QuantizationMode::Affine => {
                i32::try_from(plan_inputs.qwen3_5_config.default_quantization_bits())
                    .map_err(|_| ExpertManifestError::InvalidBits)?
            }
            QuantizationMode::NativeBfloat16 | QuantizationMode::Jangtq2Codebook => 0,
        },
        quantization_group_size: match quantization_mode {
            QuantizationMode::Affine => {
                i32::try_from(plan_inputs.qwen3_5_config.default_quantization_group_size())
                    .map_err(|_| ExpertManifestError::InvalidGroupSize)?
            }
            QuantizationMode::NativeBfloat16 | QuantizationMode::Jangtq2Codebook => 0,
        },
        quantization_mode,
    })
}
