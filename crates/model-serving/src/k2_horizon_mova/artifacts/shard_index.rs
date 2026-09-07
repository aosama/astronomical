//! Safetensors index projection for K2 Horizon MoVA stacked affine artifacts.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use super::K2HorizonMoVAArtifactValidationError;

#[derive(Debug, Deserialize)]
struct K2HorizonMoVAIndexDocument {
    #[serde(default)]
    metadata: K2HorizonMoVAIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
struct K2HorizonMoVAIndexMetadata {
    #[serde(default)]
    total_size: Option<u64>,
}

/// Index-owned tensor-to-shard mapping plus unique shard file names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct K2HorizonMoVAShardIndex {
    weight_map: BTreeMap<String, String>,
    shard_file_names: Vec<String>,
    declared_total_size: Option<u64>,
}

impl K2HorizonMoVAShardIndex {
    pub fn from_json_bytes(
        index_bytes: &[u8],
    ) -> Result<Self, K2HorizonMoVAArtifactValidationError> {
        let document: K2HorizonMoVAIndexDocument =
            serde_json::from_slice(index_bytes).map_err(|source| {
                K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                    description: format!(
                        "model.safetensors.index.json is not valid JSON: {source}"
                    ),
                }
            })?;
        if document.weight_map.is_empty() {
            return Err(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                description: "model.safetensors.index.json weight_map is empty".to_owned(),
            });
        }
        let mut shard_file_names = document
            .weight_map
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        shard_file_names.sort();
        Ok(Self {
            weight_map: document.weight_map,
            shard_file_names,
            declared_total_size: document.metadata.total_size,
        })
    }

    #[must_use]
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    #[must_use]
    pub fn shard_file_names(&self) -> &[String] {
        &self.shard_file_names
    }

    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shard_file_names.len()
    }

    #[must_use]
    pub fn shard_file_name_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    #[must_use]
    pub const fn declared_total_size(&self) -> Option<u64> {
        self.declared_total_size
    }
}
