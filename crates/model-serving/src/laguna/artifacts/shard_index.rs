use std::collections::{BTreeMap, btree_map::Entry};
use std::fmt;
use std::path::{Component, Path};

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use super::artifact_error::LagunaShardIndexError;

const MAXIMUM_INDEX_BYTES: usize = 32 * 1024 * 1024;
const MAXIMUM_TENSOR_COUNT: usize = 1_000_000;
const MAXIMUM_TENSOR_NAME_BYTES: usize = 1_024;
const MAXIMUM_SHARD_FILE_NAME_BYTES: usize = 255;
const MAXIMUM_ERROR_LABEL_CHARACTERS: usize = 256;

/// Bounded, duplicate-aware, deterministic ownership from an indexed Laguna artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaShardIndex {
    tensor_name_to_shard_file_name: BTreeMap<String, String>,
    shard_file_name_to_tensor_names: BTreeMap<String, Vec<String>>,
    declared_total_size_bytes: u64,
}

impl LagunaShardIndex {
    /// Parses exact tensor ownership without allowing JSON map replacement semantics.
    pub fn from_json_bytes(index_bytes: &[u8]) -> Result<Self, LagunaShardIndexError> {
        if index_bytes.len() > MAXIMUM_INDEX_BYTES {
            return Err(LagunaShardIndexError::IndexTooLarge {
                actual_bytes: index_bytes.len(),
                maximum_bytes: MAXIMUM_INDEX_BYTES,
            });
        }
        let index_document = serde_json::from_slice::<LagunaShardIndexDocument>(index_bytes)
            .map_err(LagunaShardIndexError::MalformedIndex)?;
        if let Some(tensor_name) = index_document.weight_map.duplicate_tensor_name {
            return Err(LagunaShardIndexError::DuplicateTensorName { tensor_name });
        }
        if index_document.weight_map.entries.len() > MAXIMUM_TENSOR_COUNT {
            return Err(LagunaShardIndexError::TensorCountTooLarge {
                actual_count: index_document.weight_map.entries.len(),
                maximum_count: MAXIMUM_TENSOR_COUNT,
            });
        }

        let mut shard_file_name_to_tensor_names = BTreeMap::<String, Vec<String>>::new();
        for (tensor_name, shard_file_name) in &index_document.weight_map.entries {
            validate_tensor_name(tensor_name)?;
            validate_shard_file_name(shard_file_name)?;
            shard_file_name_to_tensor_names
                .entry(shard_file_name.clone())
                .or_default()
                .push(tensor_name.clone());
        }
        Ok(Self {
            tensor_name_to_shard_file_name: index_document.weight_map.entries,
            shard_file_name_to_tensor_names,
            declared_total_size_bytes: index_document.metadata.total_size,
        })
    }

    /// Returns exact tensor-to-shard ownership in lexical tensor order.
    #[must_use]
    pub const fn tensor_name_to_shard_file_name(&self) -> &BTreeMap<String, String> {
        &self.tensor_name_to_shard_file_name
    }

    /// Returns exact shard-to-tensor ownership in lexical shard and tensor order.
    #[must_use]
    pub const fn shard_file_name_to_tensor_names(&self) -> &BTreeMap<String, Vec<String>> {
        &self.shard_file_name_to_tensor_names
    }

    /// Returns plain retained shard file names in deterministic order.
    pub fn shard_file_names(&self) -> impl Iterator<Item = &str> {
        self.shard_file_name_to_tensor_names
            .keys()
            .map(String::as_str)
    }

    /// Returns the exact indexed owner for one raw tensor name.
    #[must_use]
    pub fn shard_file_name_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.tensor_name_to_shard_file_name
            .get(tensor_name)
            .map(String::as_str)
    }

    /// Returns metadata.total_size before producer-specific semantics are reconciled.
    #[must_use]
    pub const fn declared_total_size_bytes(&self) -> u64 {
        self.declared_total_size_bytes
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LagunaShardIndexDocument {
    metadata: LagunaShardIndexMetadata,
    weight_map: DuplicateAwareWeightMap,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LagunaShardIndexMetadata {
    total_size: u64,
}

struct DuplicateAwareWeightMap {
    entries: BTreeMap<String, String>,
    duplicate_tensor_name: Option<String>,
}

impl<'de> Deserialize<'de> for DuplicateAwareWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(DuplicateAwareWeightMapVisitor)
    }
}

struct DuplicateAwareWeightMapVisitor;

impl<'de> Visitor<'de> for DuplicateAwareWeightMapVisitor {
    type Value = DuplicateAwareWeightMap;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a Laguna tensor-name to shard-file map")
    }

    fn visit_map<A>(self, mut map_access: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut entries = BTreeMap::new();
        let mut duplicate_tensor_name = None;
        while let Some((tensor_name, shard_file_name)) =
            map_access.next_entry::<String, String>()?
        {
            match entries.entry(tensor_name) {
                Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(shard_file_name);
                }
                Entry::Occupied(occupied_entry) => {
                    if duplicate_tensor_name.is_none() {
                        duplicate_tensor_name = Some(bounded_error_label(occupied_entry.key()));
                    }
                }
            }
        }
        Ok(DuplicateAwareWeightMap {
            entries,
            duplicate_tensor_name,
        })
    }
}

fn validate_tensor_name(tensor_name: &str) -> Result<(), LagunaShardIndexError> {
    if tensor_name.is_empty() || tensor_name.len() > MAXIMUM_TENSOR_NAME_BYTES {
        return Err(LagunaShardIndexError::InvalidTensorNameLength {
            actual_bytes: tensor_name.len(),
            maximum_bytes: MAXIMUM_TENSOR_NAME_BYTES,
        });
    }
    Ok(())
}

fn validate_shard_file_name(shard_file_name: &str) -> Result<(), LagunaShardIndexError> {
    let shard_path = Path::new(shard_file_name);
    let is_one_plain_component = !shard_file_name.is_empty()
        && shard_file_name.len() <= MAXIMUM_SHARD_FILE_NAME_BYTES
        && !shard_path.is_absolute()
        && matches!(
            shard_path.components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        )
        && shard_path
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors");
    if !is_one_plain_component {
        return Err(LagunaShardIndexError::UnsafeShardFileName {
            shard_file_name: bounded_error_label(shard_file_name),
        });
    }
    Ok(())
}

fn bounded_error_label(unbounded_label: &str) -> String {
    let mut label_characters = unbounded_label.chars();
    let bounded_label = label_characters
        .by_ref()
        .take(MAXIMUM_ERROR_LABEL_CHARACTERS)
        .collect::<String>();
    if label_characters.next().is_some() {
        format!("{bounded_label}…")
    } else {
        bounded_label
    }
}
