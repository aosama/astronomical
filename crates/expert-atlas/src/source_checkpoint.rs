//! What an Atlas conversion reads: a stock safetensors checkpoint, split into
//! routed-expert tensors (grouped by layer) and every other tensor.
//!
//! Only headers are read here. A routed-expert tensor is one whose name holds
//! `.mlp.switch_mlp.`: it batches every expert of one projection parameter of
//! one layer as `[expert_count, ...]`, which is exactly what the Atlas
//! re-lays out expert-major.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;

const SHARD_INDEX_FILE_NAME: &str = "model.safetensors.index.json";
const ROUTED_EXPERT_NAME_MARKER: &str = ".mlp.switch_mlp.";
/// The two spellings of a vision-tower tensor name (canonical and unified
/// Hugging Face), as the serving shard index recognises them.
const VISION_TOWER_NAME_PREFIXES: [&str; 2] = ["vision_tower.", "model.visual."];
/// The one place an Atlas carries a vision tower. Serving loads the tower
/// only from this file, so a conversion that put it anywhere else would
/// produce a model that refuses to load.
pub const VISION_BUNDLE_FILE_NAME: &str = "optiq/optiq_vision.safetensors";
/// A safetensors header is JSON metadata; anything larger is not a header.
const MAXIMUM_SAFETENSORS_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// One tensor as stored in a source shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceTensor {
    pub tensor_name: String,
    pub shard_file_name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute offset of the payload in the shard file.
    pub payload_file_offset: u64,
    pub payload_byte_count: u64,
}

/// The routed-expert tensors of one decoder layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceExpertLayer {
    pub layer_index: usize,
    pub expert_tensors: Vec<SourceTensor>,
}

#[derive(Clone, Debug)]
pub struct SourceCheckpoint {
    pub model_directory: PathBuf,
    pub expert_count: usize,
    pub expert_layers: Vec<SourceExpertLayer>,
    pub resident_tensors: Vec<SourceTensor>,
    /// Every language shard the index names; these are replaced, not copied.
    pub shard_file_names: BTreeSet<String>,
    /// The vision bundle (`VISION_BUNDLE_FILE_NAME`) when the source has a
    /// vision tower. Serving loads it through the separate vision path, so it
    /// travels with the companion files unchanged rather than being folded
    /// into the resident bundle, where the tower would then exist twice.
    pub vision_sidecar_file_names: BTreeSet<String>,
    /// The first shard's `__metadata__`, carried into the resident bundle.
    pub safetensors_metadata: Option<serde_json::Value>,
}

#[derive(Debug, thiserror::Error)]
pub enum SourceCheckpointError {
    #[error("{path} could not be read: {source}")]
    Unreadable {
        path: String,
        source: std::io::Error,
    },
    #[error("{path} is not valid: {reason}")]
    Invalid { path: String, reason: String },
    #[error("{path} has no routed experts to convert (no `.mlp.switch_mlp.` tensors)")]
    NoRoutedExperts { path: String },
    #[error("{path} cannot become an Atlas: {reason}")]
    UnsupportedVisionLayout { path: String, reason: String },
}

#[derive(Deserialize)]
struct ShardIndexDocument {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct HeaderEntryDocument {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl SourceCheckpoint {
    /// Reads the shard index, every shard header, and the expert count.
    pub fn read(model_directory: &Path) -> Result<Self, SourceCheckpointError> {
        let source_config = read_source_config(model_directory)?;
        let mut shard_index = read_shard_index(model_directory)?;
        let vision_sidecar_file_names =
            vision_sidecar_file_names(&shard_index, source_config.declares_vision_tower).map_err(
                |reason| SourceCheckpointError::UnsupportedVisionLayout {
                    path: model_directory.display().to_string(),
                    reason,
                },
            )?;
        let expert_count = source_config.expert_count;
        shard_index
            .weight_map
            .retain(|_, shard_file_name| !vision_sidecar_file_names.contains(shard_file_name));
        let shard_file_names: BTreeSet<String> = shard_index.weight_map.values().cloned().collect();
        let mut tensors_by_name = BTreeMap::new();
        let mut safetensors_metadata = None;
        for shard_file_name in &shard_file_names {
            let shard_header = read_shard_header(model_directory, shard_file_name)?;
            safetensors_metadata = safetensors_metadata.or(shard_header.metadata);
            for source_tensor in shard_header.tensors {
                tensors_by_name.insert(source_tensor.tensor_name.clone(), source_tensor);
            }
        }
        let (expert_layers, resident_tensors) =
            split_expert_tensors(&shard_index, tensors_by_name, model_directory)?;
        if expert_layers.is_empty() {
            return Err(SourceCheckpointError::NoRoutedExperts {
                path: model_directory.display().to_string(),
            });
        }
        for expert_layer in &expert_layers {
            check_expert_layer(expert_layer, expert_count)?;
        }
        Ok(Self {
            model_directory: model_directory.to_path_buf(),
            expert_count,
            expert_layers,
            resident_tensors,
            shard_file_names,
            vision_sidecar_file_names,
            safetensors_metadata,
        })
    }

    #[must_use]
    pub fn expert_payload_byte_count(&self) -> u64 {
        self.expert_layers
            .iter()
            .flat_map(|expert_layer| &expert_layer.expert_tensors)
            .map(|source_tensor| source_tensor.payload_byte_count)
            .sum()
    }

    #[must_use]
    pub fn resident_payload_byte_count(&self) -> u64 {
        self.resident_tensors
            .iter()
            .map(|source_tensor| source_tensor.payload_byte_count)
            .sum()
    }
}

/// What conversion needs from `config.json`.
struct SourceConfig {
    expert_count: usize,
    /// Whether the config declares a vision tower, which serving then
    /// requires to find in the vision bundle.
    declares_vision_tower: bool,
}

fn read_source_config(model_directory: &Path) -> Result<SourceConfig, SourceCheckpointError> {
    let config_path = model_directory.join("config.json");
    let config_document: serde_json::Value = read_json(&config_path)?;
    let declares_vision_tower = config_document
        .get("vision_config")
        .is_some_and(|vision_config| !vision_config.is_null());
    Ok(SourceConfig {
        expert_count: read_expert_count(&config_path, &config_document)?,
        declares_vision_tower,
    })
}

fn read_expert_count(
    config_path: &Path,
    config_document: &serde_json::Value,
) -> Result<usize, SourceCheckpointError> {
    let text_config = config_document
        .get("text_config")
        .unwrap_or(config_document);
    text_config
        .get("num_experts")
        .and_then(serde_json::Value::as_u64)
        .and_then(|expert_count| usize::try_from(expert_count).ok())
        .filter(|&expert_count| expert_count > 0)
        .ok_or_else(|| SourceCheckpointError::Invalid {
            path: config_path.display().to_string(),
            reason: "no positive num_experts; only mixture-of-experts models convert".to_owned(),
        })
}

fn read_shard_index(model_directory: &Path) -> Result<ShardIndexDocument, SourceCheckpointError> {
    read_json(&model_directory.join(SHARD_INDEX_FILE_NAME))
}

fn read_json<Document: serde::de::DeserializeOwned>(
    path: &Path,
) -> Result<Document, SourceCheckpointError> {
    let document_bytes =
        std::fs::read(path).map_err(|source| SourceCheckpointError::Unreadable {
            path: path.display().to_string(),
            source,
        })?;
    serde_json::from_slice(&document_bytes).map_err(|source| SourceCheckpointError::Invalid {
        path: path.display().to_string(),
        reason: source.to_string(),
    })
}

struct ShardHeader {
    tensors: Vec<SourceTensor>,
    metadata: Option<serde_json::Value>,
}

fn read_shard_header(
    model_directory: &Path,
    shard_file_name: &str,
) -> Result<ShardHeader, SourceCheckpointError> {
    let shard_path = model_directory.join(shard_file_name);
    let path = shard_path.display().to_string();
    let unreadable = |source| SourceCheckpointError::Unreadable {
        path: path.clone(),
        source,
    };
    let mut shard_file = File::open(&shard_path).map_err(unreadable)?;
    let mut length_prefix = [0u8; 8];
    shard_file
        .read_exact(&mut length_prefix)
        .map_err(unreadable)?;
    let header_byte_count = u64::from_le_bytes(length_prefix);
    if header_byte_count > MAXIMUM_SAFETENSORS_HEADER_BYTES {
        return Err(SourceCheckpointError::Invalid {
            path,
            reason: format!("header claims {header_byte_count} bytes"),
        });
    }
    let mut header_bytes = vec![0u8; header_byte_count as usize];
    shard_file
        .read_exact(&mut header_bytes)
        .map_err(unreadable)?;
    let mut header_document: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header_bytes).map_err(|source| SourceCheckpointError::Invalid {
            path: path.clone(),
            reason: source.to_string(),
        })?;
    let metadata = header_document.remove("__metadata__");
    let payload_start = 8 + header_byte_count;
    let mut tensors = Vec::with_capacity(header_document.len());
    for (tensor_name, entry_value) in header_document {
        let entry: HeaderEntryDocument = serde_json::from_value(entry_value).map_err(|source| {
            SourceCheckpointError::Invalid {
                path: path.clone(),
                reason: format!("{tensor_name}: {source}"),
            }
        })?;
        let (payload_file_offset, payload_byte_count) =
            payload_span(payload_start, entry.data_offsets).ok_or_else(|| {
                SourceCheckpointError::Invalid {
                    path: path.clone(),
                    reason: format!(
                        "{tensor_name} has impossible data offsets {:?}",
                        entry.data_offsets
                    ),
                }
            })?;
        tensors.push(SourceTensor {
            tensor_name,
            shard_file_name: shard_file_name.to_owned(),
            dtype: entry.dtype,
            shape: entry.shape,
            payload_file_offset,
            payload_byte_count,
        });
    }
    Ok(ShardHeader { tensors, metadata })
}

/// The absolute file offset and byte count of one tensor payload, or `None`
/// for offsets a valid header cannot hold (end before start, or past
/// `u64`). Unchecked arithmetic here would turn a corrupt header into a
/// multi-exabyte copy request instead of a refusal.
fn payload_span(payload_start: u64, data_offsets: [u64; 2]) -> Option<(u64, u64)> {
    let [start_offset, end_offset] = data_offsets;
    let payload_byte_count = end_offset.checked_sub(start_offset)?;
    let payload_file_offset = payload_start.checked_add(start_offset)?;
    payload_file_offset.checked_add(payload_byte_count)?;
    Some((payload_file_offset, payload_byte_count))
}

/// Every indexed tensor goes to exactly one side: its decoder layer's
/// experts, or the resident bundle.
fn split_expert_tensors(
    shard_index: &ShardIndexDocument,
    mut tensors_by_name: BTreeMap<String, SourceTensor>,
    model_directory: &Path,
) -> Result<(Vec<SourceExpertLayer>, Vec<SourceTensor>), SourceCheckpointError> {
    let mut expert_tensors_by_layer: BTreeMap<usize, Vec<SourceTensor>> = BTreeMap::new();
    let mut resident_tensors = Vec::new();
    for tensor_name in shard_index.weight_map.keys() {
        let source_tensor =
            tensors_by_name
                .remove(tensor_name)
                .ok_or_else(|| SourceCheckpointError::Invalid {
                    path: model_directory
                        .join(SHARD_INDEX_FILE_NAME)
                        .display()
                        .to_string(),
                    reason: format!("{tensor_name} is indexed but absent from its shard"),
                })?;
        match routed_expert_layer_index(tensor_name) {
            Some(layer_index) => expert_tensors_by_layer
                .entry(layer_index)
                .or_default()
                .push(source_tensor),
            None => resident_tensors.push(source_tensor),
        }
    }
    let expert_layers = expert_tensors_by_layer
        .into_iter()
        .map(|(layer_index, expert_tensors)| SourceExpertLayer {
            layer_index,
            expert_tensors,
        })
        .collect();
    Ok((expert_layers, resident_tensors))
}

/// The vision bundle to carry unchanged, or why this checkpoint's vision
/// layout cannot become an Atlas. Serving reads a vision tower only from
/// `VISION_BUNDLE_FILE_NAME` and requires it whenever the config declares
/// one, so conversion refuses every other layout up front rather than
/// writing tens of gigabytes of a model that will not load.
fn vision_sidecar_file_names(
    shard_index: &ShardIndexDocument,
    declares_vision_tower: bool,
) -> Result<BTreeSet<String>, String> {
    let mut has_vision_bundle = false;
    for (tensor_name, shard_file_name) in &shard_index.weight_map {
        let is_in_vision_bundle = shard_file_name == VISION_BUNDLE_FILE_NAME;
        if is_vision_tower_tensor(tensor_name) != is_in_vision_bundle {
            return Err(format!(
                "{tensor_name} sits in {shard_file_name}, but an Atlas keeps the vision tower, and only the vision tower, in {VISION_BUNDLE_FILE_NAME}"
            ));
        }
        has_vision_bundle |= is_in_vision_bundle;
    }
    if declares_vision_tower && !has_vision_bundle {
        return Err(format!(
            "config.json declares a vision tower but no tensor is indexed in {VISION_BUNDLE_FILE_NAME}"
        ));
    }
    Ok(if has_vision_bundle {
        BTreeSet::from([VISION_BUNDLE_FILE_NAME.to_owned()])
    } else {
        BTreeSet::new()
    })
}

fn is_vision_tower_tensor(tensor_name: &str) -> bool {
    VISION_TOWER_NAME_PREFIXES
        .iter()
        .any(|vision_prefix| tensor_name.starts_with(vision_prefix))
}

/// The decoder layer of a routed-expert tensor name, or `None` for any other
/// tensor. `...layers.12.mlp.switch_mlp.gate_proj.weight` -> `Some(12)`.
#[must_use]
pub fn routed_expert_layer_index(tensor_name: &str) -> Option<usize> {
    let (before_marker, _) = tensor_name.split_once(ROUTED_EXPERT_NAME_MARKER)?;
    let (_, layer_index) = before_marker.rsplit_once("layers.")?;
    layer_index.parse().ok()
}

fn check_expert_layer(
    expert_layer: &SourceExpertLayer,
    expert_count: usize,
) -> Result<(), SourceCheckpointError> {
    for source_tensor in &expert_layer.expert_tensors {
        let leads_with_experts = source_tensor.shape.first() == Some(&expert_count);
        // An empty payload would give zero-byte expert slices, which the
        // record writer cannot split.
        if !leads_with_experts
            || source_tensor.payload_byte_count == 0
            || source_tensor.payload_byte_count % expert_count as u64 != 0
        {
            return Err(SourceCheckpointError::Invalid {
                path: source_tensor.shard_file_name.clone(),
                reason: format!(
                    "{} has shape {:?} and {} bytes, which do not split into {expert_count} experts",
                    source_tensor.tensor_name,
                    source_tensor.shape,
                    source_tensor.payload_byte_count
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "source_checkpoint_tests.rs"]
mod source_checkpoint_tests;
