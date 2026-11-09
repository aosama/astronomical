//! Expert layer plans for an Expert Atlas model.
//!
//! The ordinary plan builder finds each routed-expert tensor through the
//! shard index and a cache of parsed safetensors headers, then validates its
//! geometry. An Atlas keeps those tensors in layer record files instead, so
//! this module hands the same builder an index that points the expert
//! tensors at their layer files and a header cache already holding one
//! synthesized header per layer file. Every tensor then passes through the
//! same shape, bit-width and mode validation as a stock checkpoint. The only
//! Atlas-specific fact the plans carry afterwards is the record stride: in a
//! layer file, consecutive experts' slices of one tensor are a whole record
//! apart rather than back to back.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use astronomical_expert_atlas::{AtlasExpertLayer, AtlasManifest};

use super::quantized_expert_layer_plan::build_quantized_expert_layer_plan_with_stored_names_and_header_cache;
use crate::expert_paging::quantized_expert_manifest::{
    ExpertManifestError, QuantizedExpertLayerPlan,
};
use crate::expert_paging::safetensors_header::{SafetensorsHeader, TensorHeaderEntry, parse_dtype};
use crate::qwen3_5_core::configuration::Qwen3_5Config;

const ROUTED_EXPERT_NAME_MARKER: &str = "switch_mlp.";

/// Every decoder layer's plan for a validated sparse artifact, stock or Atlas,
/// built from the maps the artifact's (possibly synthesized) shard index and
/// tensor inventory carry.
pub fn expert_layer_plans_for_validated_artifact(
    validated_artifact: &super::ValidatedQwen3_5Artifact,
    model_dir: &Path,
) -> Result<Vec<QuantizedExpertLayerPlan>, ExpertManifestError> {
    let weight_map: HashMap<String, String> = validated_artifact
        .shard_index()
        .language_tensor_name_to_shard_file_name()
        .iter()
        .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
        .collect();
    let stored_tensor_name_by_canonical_name: HashMap<String, String> = validated_artifact
        .tensor_inventory()
        .locations()
        .map(|location| {
            (
                location.canonical_name().to_owned(),
                location.stored_name().to_owned(),
            )
        })
        .collect();
    build_expert_layer_plans(
        model_dir,
        validated_artifact.expert_atlas(),
        &weight_map,
        &stored_tensor_name_by_canonical_name,
        validated_artifact.config(),
        validated_artifact.config().layer_count() as usize,
    )
}

/// Every decoder layer's plan for any sparse checkpoint: from the Atlas
/// manifest when the model has one, from the shard index otherwise. Both the
/// loader and the RAM budget call this, so they always agree on the plans.
pub(crate) fn build_expert_layer_plans(
    model_dir: &Path,
    expert_atlas: Option<&AtlasManifest>,
    weight_map: &HashMap<String, String>,
    stored_tensor_name_by_canonical_name: &HashMap<String, String>,
    config: &Qwen3_5Config,
    decoder_layer_count: usize,
) -> Result<Vec<QuantizedExpertLayerPlan>, ExpertManifestError> {
    match expert_atlas {
        Some(expert_atlas) => build_atlas_expert_layer_plans(
            model_dir,
            expert_atlas,
            weight_map,
            stored_tensor_name_by_canonical_name,
            config,
            decoder_layer_count,
        ),
        None => super::quantized_expert_layer_plan::build_quantized_expert_layer_plans(
            model_dir,
            weight_map,
            stored_tensor_name_by_canonical_name,
            config,
            decoder_layer_count,
        ),
    }
}

/// Builds every decoder layer's plan for an Atlas model. `weight_map` and
/// `stored_tensor_name_by_canonical_name` are the synthesized index's maps;
/// the expert entries are replaced here.
fn build_atlas_expert_layer_plans(
    model_dir: &Path,
    expert_atlas: &AtlasManifest,
    weight_map: &HashMap<String, String>,
    stored_tensor_name_by_canonical_name: &HashMap<String, String>,
    config: &Qwen3_5Config,
    decoder_layer_count: usize,
) -> Result<Vec<QuantizedExpertLayerPlan>, ExpertManifestError> {
    let mut atlas_weight_map = weight_map.clone();
    let mut atlas_stored_names = stored_tensor_name_by_canonical_name.clone();
    let mut header_cache = HashMap::new();
    let mut record_stride_by_layer_file = HashMap::new();
    for expert_layer in &expert_atlas.expert_layers {
        let layer_file_path = model_dir.join(&expert_layer.file_name);
        let synthesized_header = synthesize_layer_header(expert_layer)?;
        for tensor_entry in &synthesized_header.tensor_entries {
            atlas_weight_map.insert(
                tensor_entry.tensor_name.clone(),
                expert_layer.file_name.clone(),
            );
            atlas_stored_names.remove(&tensor_entry.tensor_name);
        }
        header_cache.insert(layer_file_path.clone(), synthesized_header);
        record_stride_by_layer_file.insert(layer_file_path, expert_layer.record_stride_bytes);
    }
    let mut layer_plans = Vec::with_capacity(decoder_layer_count);
    for decoder_layer_index in 0..decoder_layer_count {
        let layer_prefix = layer_prefix(decoder_layer_index);
        let mut layer_plan = build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
            model_dir,
            &atlas_weight_map,
            &atlas_stored_names,
            &layer_prefix,
            config,
            &mut header_cache,
        )?;
        apply_record_strides(&mut layer_plan, &record_stride_by_layer_file)?;
        layer_plans.push(layer_plan);
    }
    Ok(layer_plans)
}

fn layer_prefix(decoder_layer_index: usize) -> String {
    format!("language_model.model.layers.{decoder_layer_index}.mlp")
}

/// One header per layer file: every slot as a full expert-batched tensor
/// starting at its offset in record 0. The entry is named with the canonical
/// name the plan builder asks for, so stored-name differences in the source
/// cannot break the lookup.
fn synthesize_layer_header(
    expert_layer: &AtlasExpertLayer,
) -> Result<SafetensorsHeader, ExpertManifestError> {
    let mut tensor_entries = Vec::with_capacity(expert_layer.slots.len());
    for slot in &expert_layer.slots {
        let (_, parameter_path) = slot
            .tensor_name
            .split_once(ROUTED_EXPERT_NAME_MARKER)
            .ok_or_else(|| ExpertManifestError::MissingTensorEntry {
                tensor_name: slot.tensor_name.clone(),
            })?;
        let canonical_tensor_name = format!(
            "{}.{ROUTED_EXPERT_NAME_MARKER}{parameter_path}",
            layer_prefix(expert_layer.layer_index)
        );
        let dtype = parse_dtype(&slot.dtype).ok_or_else(|| ExpertManifestError::ReaderError {
            description: format!(
                "Atlas slot {} has unknown dtype {}",
                slot.tensor_name, slot.dtype
            ),
        })?;
        tensor_entries.push(TensorHeaderEntry {
            tensor_name: canonical_tensor_name,
            dtype,
            shape: slot.shape.clone(),
            data_start_offset: slot.offset_in_record,
            data_end_offset: slot
                .bytes_per_expert
                .checked_mul(expert_layer.expert_count as u64)
                .and_then(|slot_byte_count| slot_byte_count.checked_add(slot.offset_in_record))
                .ok_or_else(|| ExpertManifestError::ReaderError {
                    description: format!("Atlas slot {} overflows its layer", slot.tensor_name),
                })?,
        });
    }
    Ok(SafetensorsHeader {
        tensor_entries,
        payload_start_offset: 0,
        total_file_size_bytes: expert_layer.file_byte_count(),
    })
}

/// Every tensor read from a layer file steps by the record stride.
fn apply_record_strides(
    layer_plan: &mut QuantizedExpertLayerPlan,
    record_stride_by_layer_file: &HashMap<PathBuf, u64>,
) -> Result<(), ExpertManifestError> {
    for tensor_source in &mut layer_plan.tensor_sources {
        if let Some(&record_stride_bytes) =
            record_stride_by_layer_file.get(&tensor_source.source_file)
        {
            tensor_source.expert_stride_bytes = usize::try_from(record_stride_bytes).map_err(
                |_| ExpertManifestError::ReaderError {
                    description: format!(
                        "record stride {record_stride_bytes} exceeds this machine's address range"
                    ),
                },
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "atlas_expert_layer_plans_tests.rs"]
mod atlas_expert_layer_plans_tests;
