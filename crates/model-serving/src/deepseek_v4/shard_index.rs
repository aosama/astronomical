use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use serde::Deserialize;
use thiserror::Error;

use super::DeepSeekV4DsparkArtifactCapability;

pub const MAXIMUM_DEEPSEEK_V4_INDEX_BYTES: usize = 1024 * 1024;
const EXPECTED_LAYER_COUNT: usize = 43;

/// Structural tensor-to-shard inventory for DeepSeek-V4-Flash-0731.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4ShardIndex {
    shard_file_names: Vec<String>,
    tensor_names_by_shard_file_name: BTreeMap<String, BTreeSet<String>>,
    total_payload_bytes: u64,
}

impl DeepSeekV4ShardIndex {
    /// Parses the selected model’s indexed tensor ownership without loading weights.
    pub fn from_json_bytes(
        index_bytes: &[u8],
        dspark_artifact_capability: &DeepSeekV4DsparkArtifactCapability,
    ) -> Result<Self, DeepSeekV4ShardIndexError> {
        if index_bytes.len() > MAXIMUM_DEEPSEEK_V4_INDEX_BYTES {
            return Err(DeepSeekV4ShardIndexError::IndexTooLarge {
                actual_index_bytes: index_bytes.len(),
                maximum_index_bytes: MAXIMUM_DEEPSEEK_V4_INDEX_BYTES,
            });
        }
        let index_document = serde_json::from_slice::<DeepSeekV4ShardIndexDocument>(index_bytes)
            .map_err(DeepSeekV4ShardIndexError::DeserializeIndex)?;
        if index_document.weight_map.is_empty() {
            return Err(DeepSeekV4ShardIndexError::EmptyWeightMap);
        }

        let mut tensor_names_by_shard_file_name = BTreeMap::<String, BTreeSet<String>>::new();
        let mut discovered_model_layer_indices = BTreeSet::new();
        let mut has_embedding = false;
        let mut has_model_norm = false;
        let mut has_language_head = false;
        let mut discovered_dspark_tensor_names = BTreeSet::new();
        for (tensor_name, shard_file_name) in index_document.weight_map {
            validate_shard_file_name(&shard_file_name)?;
            match tensor_name.as_str() {
                tensor_name if tensor_name.starts_with("model.embed_tokens.") => {
                    has_embedding = true;
                }
                tensor_name if tensor_name.starts_with("model.norm.") => {
                    has_model_norm = true;
                }
                tensor_name if tensor_name.starts_with("model.hc_head.") => {}
                tensor_name if tensor_name.starts_with("model.layers.") => {
                    discovered_model_layer_indices.insert(parse_model_layer_index(&tensor_name)?);
                }
                tensor_name if tensor_name.starts_with("lm_head.") => {
                    has_language_head = true;
                }
                tensor_name if tensor_name.starts_with("mtp.") => {
                    if !dspark_artifact_capability.is_declared() {
                        return Err(DeepSeekV4ShardIndexError::UnexpectedDsparkTensor {
                            tensor_name: tensor_name.to_owned(),
                        });
                    }
                    discovered_dspark_tensor_names.insert(tensor_name.to_owned());
                }
                _ => {
                    return Err(DeepSeekV4ShardIndexError::UnsupportedTensorNamespace {
                        tensor_name,
                    });
                }
            }
            tensor_names_by_shard_file_name
                .entry(shard_file_name)
                .or_default()
                .insert(tensor_name);
        }
        if !has_embedding || !has_model_norm || !has_language_head {
            return Err(DeepSeekV4ShardIndexError::MissingTargetMarker);
        }
        for expected_layer_index in 0..EXPECTED_LAYER_COUNT {
            if !discovered_model_layer_indices.contains(&expected_layer_index) {
                return Err(DeepSeekV4ShardIndexError::MissingModelLayer {
                    layer_index: expected_layer_index,
                });
            }
        }
        validate_dspark_tensor_inventory(
            dspark_artifact_capability,
            &discovered_dspark_tensor_names,
        )?;
        let shard_file_names = tensor_names_by_shard_file_name.keys().cloned().collect();
        Ok(Self {
            shard_file_names,
            tensor_names_by_shard_file_name,
            total_payload_bytes: index_document.metadata.total_size,
        })
    }

    /// Returns indexed shard file names in deterministic order.
    #[must_use]
    pub fn shard_file_names(&self) -> &[String] {
        &self.shard_file_names
    }

    /// Returns the indexed tensor names assigned to one shard.
    #[must_use]
    pub fn tensor_names_for_shard(&self, shard_file_name: &str) -> Option<&BTreeSet<String>> {
        self.tensor_names_by_shard_file_name.get(shard_file_name)
    }

    /// Returns the payload size declared by the safetensors index.
    #[must_use]
    pub const fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }
}

#[derive(Debug, Deserialize)]
struct DeepSeekV4ShardIndexDocument {
    metadata: DeepSeekV4ShardIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct DeepSeekV4ShardIndexMetadata {
    total_size: u64,
}

fn validate_shard_file_name(shard_file_name: &str) -> Result<(), DeepSeekV4ShardIndexError> {
    let shard_file_path = Path::new(shard_file_name);
    if shard_file_name.is_empty()
        || shard_file_path
            .extension()
            .and_then(|extension| extension.to_str())
            != Some("safetensors")
        || shard_file_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(DeepSeekV4ShardIndexError::InvalidShardFileName {
            shard_file_name: shard_file_name.to_owned(),
        });
    }
    Ok(())
}

fn parse_model_layer_index(tensor_name: &str) -> Result<usize, DeepSeekV4ShardIndexError> {
    let mut tensor_name_components = tensor_name.split('.');
    let _model_component = tensor_name_components.next();
    let _layers_component = tensor_name_components.next();
    let layer_index_text = tensor_name_components.next().ok_or_else(|| {
        DeepSeekV4ShardIndexError::InvalidModelLayerTensorName {
            tensor_name: tensor_name.to_owned(),
        }
    })?;
    let layer_index = layer_index_text.parse::<usize>().map_err(|_| {
        DeepSeekV4ShardIndexError::InvalidModelLayerTensorName {
            tensor_name: tensor_name.to_owned(),
        }
    })?;
    if layer_index >= EXPECTED_LAYER_COUNT {
        return Err(DeepSeekV4ShardIndexError::InvalidModelLayerIndex {
            tensor_name: tensor_name.to_owned(),
            layer_index,
        });
    }
    Ok(layer_index)
}

fn validate_dspark_tensor_inventory(
    dspark_artifact_capability: &DeepSeekV4DsparkArtifactCapability,
    discovered_dspark_tensor_names: &BTreeSet<String>,
) -> Result<(), DeepSeekV4ShardIndexError> {
    if !dspark_artifact_capability.is_declared() {
        return Ok(());
    }
    for tensor_name in discovered_dspark_tensor_names {
        let mut tensor_name_components = tensor_name.split('.');
        let _mtp_component = tensor_name_components.next();
        let stage_index = tensor_name_components
            .next()
            .and_then(|stage_index_text| stage_index_text.parse::<usize>().ok());
        if !matches!(stage_index, Some(0..=2)) {
            return Err(DeepSeekV4ShardIndexError::InvalidDsparkTensorName {
                tensor_name: tensor_name.clone(),
            });
        }
    }
    const REQUIRED_DSPARK_TENSOR_NAMES: [&str; 8] = [
        "mtp.0.main_proj.weight",
        "mtp.0.main_norm.weight",
        "mtp.0.attn.wq_a.weight",
        "mtp.1.attn.wq_a.weight",
        "mtp.2.attn.wq_a.weight",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
        "mtp.2.confidence_head.proj.weight",
    ];
    for required_tensor_name in REQUIRED_DSPARK_TENSOR_NAMES {
        if !discovered_dspark_tensor_names.contains(required_tensor_name) {
            return Err(DeepSeekV4ShardIndexError::MissingDsparkTensor {
                tensor_name: required_tensor_name.to_owned(),
            });
        }
    }
    for stage_index in 0..3 {
        let stage_prefix = format!("mtp.{stage_index}.");
        if !discovered_dspark_tensor_names
            .iter()
            .any(|tensor_name| tensor_name.starts_with(&stage_prefix))
        {
            return Err(DeepSeekV4ShardIndexError::MissingDsparkTensor {
                tensor_name: stage_prefix,
            });
        }
    }
    Ok(())
}

/// A bounded DeepSeek-V4 safetensors index failure.
#[derive(Debug, Error)]
pub enum DeepSeekV4ShardIndexError {
    /// The index exceeds the bounded structural-read limit.
    #[error(
        "DeepSeek-V4 shard index is {actual_index_bytes} bytes, exceeding {maximum_index_bytes}"
    )]
    IndexTooLarge {
        actual_index_bytes: usize,
        maximum_index_bytes: usize,
    },
    /// The index could not be decoded.
    #[error("failed to decode the DeepSeek-V4 shard index")]
    DeserializeIndex(#[source] serde_json::Error),
    /// The index does not name any tensors.
    #[error("DeepSeek-V4 shard index has an empty weight map")]
    EmptyWeightMap,
    /// An indexed shard path is unsafe or unsupported.
    #[error("DeepSeek-V4 index contains invalid shard file name '{shard_file_name}'")]
    InvalidShardFileName { shard_file_name: String },
    /// An indexed tensor lies outside the selected target or DSpark namespace.
    #[error("DeepSeek-V4 index contains unsupported tensor namespace '{tensor_name}'")]
    UnsupportedTensorNamespace { tensor_name: String },
    /// An indexed target-layer tensor has no valid layer index.
    #[error("DeepSeek-V4 index contains invalid target-layer tensor '{tensor_name}'")]
    InvalidModelLayerTensorName { tensor_name: String },
    /// An indexed target-layer tensor lies outside the selected model layer range.
    #[error("DeepSeek-V4 index tensor '{tensor_name}' has unsupported layer index {layer_index}")]
    InvalidModelLayerIndex {
        tensor_name: String,
        layer_index: usize,
    },
    /// The index does not contain the target model roots needed by a later loader.
    #[error("DeepSeek-V4 index is missing one or more target model markers")]
    MissingTargetMarker,
    /// The index omitted an expected target layer.
    #[error("DeepSeek-V4 index is missing target layer {layer_index}")]
    MissingModelLayer { layer_index: usize },
    /// A target-only artifact must not contain DSpark tensors.
    #[error("target-only DeepSeek-V4 artifact contains DSpark tensor '{tensor_name}'")]
    UnexpectedDsparkTensor { tensor_name: String },
    /// A declared DSpark artifact omitted a required structural tensor.
    #[error("DeepSeek-V4 DSpark artifact is missing tensor '{tensor_name}'")]
    MissingDsparkTensor { tensor_name: String },
    /// A declared DSpark artifact names an unsupported DSpark stage.
    #[error("DeepSeek-V4 DSpark artifact contains invalid tensor '{tensor_name}'")]
    InvalidDsparkTensorName { tensor_name: String },
}
