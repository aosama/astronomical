//! Every tensor in every safetensors file of an artifact, taken from the file
//! headers themselves. Reading each header rather than trusting the index finds
//! towers kept in a sidecar the index never mentions.

use std::collections::BTreeMap;

use crate::artifact_source::ArtifactSource;
use crate::error::VisionTowerError;
use crate::safetensors_header::{
    HEADER_LENGTH_PREFIX_BYTES, TensorEntry, parse_header, parse_header_length,
};

const SAFETENSORS_SUFFIX: &str = ".safetensors";
// Far above any real model, and low enough that a hostile listing cannot turn
// one inspection into thousands of requests.
const MAXIMUM_WEIGHT_FILES: usize = 512;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocatedTensor {
    pub relative_path: String,
    pub entry: TensorEntry,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WeightInventory {
    pub tensors: BTreeMap<String, LocatedTensor>,
    pub weight_file_paths: Vec<String>,
}

pub async fn read_weight_inventory(
    source: &ArtifactSource,
) -> Result<WeightInventory, VisionTowerError> {
    let mut inventory = WeightInventory::default();
    for artifact_file in &source.listing().files {
        if !artifact_file.relative_path.ends_with(SAFETENSORS_SUFFIX) {
            continue;
        }
        if inventory.weight_file_paths.len() == MAXIMUM_WEIGHT_FILES {
            return Err(VisionTowerError::TooManyWeightFiles {
                maximum_count: MAXIMUM_WEIGHT_FILES,
            });
        }
        let relative_path = &artifact_file.relative_path;
        let header_entries =
            read_file_header(source, relative_path, artifact_file.byte_count).await?;
        inventory.weight_file_paths.push(relative_path.clone());
        for (tensor_name, entry) in header_entries {
            // Listing order is sorted, so which copy wins a duplicate is stable.
            inventory
                .tensors
                .entry(tensor_name)
                .or_insert(LocatedTensor {
                    relative_path: relative_path.clone(),
                    entry,
                });
        }
    }
    Ok(inventory)
}

async fn read_file_header(
    source: &ArtifactSource,
    relative_path: &str,
    file_byte_count: Option<u64>,
) -> Result<BTreeMap<String, TensorEntry>, VisionTowerError> {
    let prefix_bytes = source
        .read_byte_range(relative_path, 0, HEADER_LENGTH_PREFIX_BYTES)
        .await?;
    let header_length = parse_header_length(relative_path, &prefix_bytes)?;
    let header_bytes = source
        .read_byte_range(relative_path, HEADER_LENGTH_PREFIX_BYTES, header_length)
        .await?;
    parse_header(relative_path, &header_bytes, file_byte_count)
}
