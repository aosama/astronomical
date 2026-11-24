//! Answers "does this model have a vision tower, and is it whole?".

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::Value;

use crate::artifact_source::{ArtifactSource, WEIGHT_INDEX_FILE_NAME};
use crate::error::VisionTowerError;
use crate::model_config::{ModelConfigSummary, read_model_config};
use crate::tensor_names::{
    QUANTIZATION_SCALES_SUFFIX, block_index, is_vision_tower_tensor, tower_name_prefix,
};
use crate::weight_inventory::{LocatedTensor, WeightInventory, read_weight_inventory};

const MAXIMUM_WEIGHT_INDEX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TowerStatus {
    /// Tower tensors exist, and they match the config's block count.
    Present,
    /// Tower tensors exist, but the config is missing or disagrees with them.
    Partial,
    /// A `vision_config` with no tensors behind it.
    AbsentWithVisionConfig,
    /// Image tokens in the config, kept from a multimodal family, and no tower.
    AbsentWithImageTokens,
    /// Nothing about this model suggests images.
    Absent,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TowerSummary {
    pub tensor_count: u64,
    pub parameter_count: u64,
    pub byte_count: u64,
    pub block_count: u64,
    pub name_prefixes: Vec<String>,
    pub weight_files: Vec<String>,
    pub data_type_counts: BTreeMap<String, u64>,
    pub is_quantized: bool,
    /// Whether `model.safetensors.index.json` names the tower. Loaders that go by
    /// the index, Astronomical among them, miss a tower it leaves out. `None` when
    /// the artifact has no index.
    pub is_in_weight_index: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TowerInspection {
    pub artifact: String,
    pub pinned_revision: Option<String>,
    pub license: Option<String>,
    pub pipeline_tag: Option<String>,
    pub config: ModelConfigSummary,
    pub weight_file_count: u64,
    pub total_tensor_count: u64,
    pub total_weight_bytes: u64,
    pub status: TowerStatus,
    pub tower: Option<TowerSummary>,
}

pub async fn inspect_artifact(
    source: &ArtifactSource,
) -> Result<TowerInspection, VisionTowerError> {
    let config = read_model_config(source).await?;
    let inventory = read_weight_inventory(source).await?;
    let is_in_weight_index = read_weight_index_mentions_tower(source).await?;
    let tower = summarize_tower(&inventory, is_in_weight_index);
    let status = tower_status(&config, tower.as_ref());
    let mut total_weight_bytes = 0_u64;
    for located_tensor in inventory.tensors.values() {
        total_weight_bytes = total_weight_bytes.saturating_add(located_tensor.entry.byte_count);
    }
    let listing = source.listing();
    Ok(TowerInspection {
        artifact: listing.display_name.clone(),
        pinned_revision: listing.pinned_revision.clone(),
        license: listing.license.clone(),
        pipeline_tag: listing.pipeline_tag.clone(),
        config,
        weight_file_count: inventory.weight_file_paths.len() as u64,
        total_tensor_count: inventory.tensors.len() as u64,
        total_weight_bytes,
        status,
        tower,
    })
}

/// How many tower tensors an inventory holds.
#[must_use]
pub fn tower_tensor_count(inventory: &WeightInventory) -> u64 {
    let mut counted_tensors = 0_u64;
    for tensor_name in inventory.tensors.keys() {
        if is_vision_tower_tensor(tensor_name) {
            counted_tensors += 1;
        }
    }
    counted_tensors
}

fn summarize_tower(
    inventory: &WeightInventory,
    is_in_weight_index: Option<bool>,
) -> Option<TowerSummary> {
    let mut summary = TowerSummary {
        tensor_count: 0,
        parameter_count: 0,
        byte_count: 0,
        block_count: 0,
        name_prefixes: Vec::new(),
        weight_files: Vec::new(),
        data_type_counts: BTreeMap::new(),
        is_quantized: false,
        is_in_weight_index,
    };
    let mut block_indexes = BTreeSet::new();
    let mut name_prefixes = BTreeSet::new();
    let mut weight_files = BTreeSet::new();
    for (tensor_name, located_tensor) in &inventory.tensors {
        if !is_vision_tower_tensor(tensor_name) {
            continue;
        }
        add_tensor_to_summary(&mut summary, tensor_name, located_tensor);
        name_prefixes.insert(tower_name_prefix(tensor_name));
        weight_files.insert(located_tensor.relative_path.clone());
        if let Some(block_index) = block_index(tensor_name) {
            block_indexes.insert(block_index);
        }
    }
    if summary.tensor_count == 0 {
        return None;
    }
    summary.block_count = block_indexes.len() as u64;
    summary.name_prefixes = name_prefixes.into_iter().collect();
    summary.weight_files = weight_files.into_iter().collect();
    Some(summary)
}

fn add_tensor_to_summary(
    summary: &mut TowerSummary,
    tensor_name: &str,
    located_tensor: &LocatedTensor,
) {
    let entry = &located_tensor.entry;
    summary.tensor_count += 1;
    summary.parameter_count = summary
        .parameter_count
        .saturating_add(entry.element_count());
    summary.byte_count = summary.byte_count.saturating_add(entry.byte_count);
    *summary
        .data_type_counts
        .entry(entry.data_type.clone())
        .or_insert(0) += 1;
    if tensor_name.ends_with(QUANTIZATION_SCALES_SUFFIX) {
        summary.is_quantized = true;
    }
}

fn tower_status(config: &ModelConfigSummary, tower: Option<&TowerSummary>) -> TowerStatus {
    let Some(tower) = tower else {
        if config.vision.is_some() {
            return TowerStatus::AbsentWithVisionConfig;
        }
        if config.declares_image_tokens() {
            return TowerStatus::AbsentWithImageTokens;
        }
        return TowerStatus::Absent;
    };
    let Some(vision_config) = &config.vision else {
        return TowerStatus::Partial;
    };
    match vision_config.block_count {
        Some(block_count) if block_count != tower.block_count => TowerStatus::Partial,
        _ => TowerStatus::Present,
    }
}

async fn read_weight_index_mentions_tower(
    source: &ArtifactSource,
) -> Result<Option<bool>, VisionTowerError> {
    let Some(index_bytes) = source
        .read_whole_file(WEIGHT_INDEX_FILE_NAME, MAXIMUM_WEIGHT_INDEX_BYTES)
        .await?
    else {
        return Ok(None);
    };
    let index_document: Value = match serde_json::from_slice(&index_bytes) {
        Ok(index_document) => index_document,
        Err(parse_error) => {
            return Err(VisionTowerError::JsonMalformed {
                file: WEIGHT_INDEX_FILE_NAME.to_owned(),
                description: parse_error.to_string(),
            });
        }
    };
    let Some(weight_map) = index_document.get("weight_map").and_then(Value::as_object) else {
        return Ok(Some(false));
    };
    for tensor_name in weight_map.keys() {
        if is_vision_tower_tensor(tensor_name) {
            return Ok(Some(true));
        }
    }
    Ok(Some(false))
}

#[cfg(test)]
#[path = "tower_inspection_tests.rs"]
mod tests;
