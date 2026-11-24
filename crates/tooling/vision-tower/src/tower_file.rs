//! Writes a tower as one safetensors file in the MLX layout, fetching only the
//! tower's own tensors from wherever the source lives.
//!
//! Its own file rather than tensors merged into the text shards, which is what most
//! published models do, because the text file then stays byte-for-byte what its
//! publisher released: anyone can check that against the source's hash, and it
//! deduplicates against the source on disk and on the Hub.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::artifact_source::ArtifactSource;
use crate::error::VisionTowerError;
use crate::safetensors_header::{HEADER_LENGTH_PREFIX_BYTES, element_byte_width};
use crate::tensor_names::is_vision_tower_tensor;
use crate::tower_layout::{
    channels_first_to_channels_last, channels_last_shape, is_channels_first_patch_embedding,
    mlx_tower_tensor_name,
};
use crate::weight_inventory::{LocatedTensor, WeightInventory};

// safetensors pads its header so tensor data starts on an 8-byte boundary, which
// keeps every tensor aligned when the file is memory mapped.
const HEADER_ALIGNMENT_BYTES: usize = 8;
const TOWER_FILE_METADATA_FORMAT: &str = "mlx";

/// One tower tensor, under its MLX name, in the order it is written.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedTowerTensor<'inventory> {
    pub output_name: String,
    pub output_shape: Vec<u64>,
    pub source_name: &'inventory str,
    pub source_tensor: &'inventory LocatedTensor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WrittenTowerFile {
    pub tensor_names: Vec<String>,
    pub parameter_count: u64,
    pub tensor_byte_count: u64,
}

/// Told after each tensor is written.
pub trait TowerFileProgress {
    fn tower_bytes_written(&mut self, completed_bytes: u64, total_bytes: u64);
}

/// Sorted by output name, so the same inputs always produce the same file. Refuses
/// the whole tower if any tensor is in a layout this cannot convert, because half a
/// tower would load and then fail in confusing ways.
pub fn plan_tower_tensors(
    tower_inventory: &WeightInventory,
) -> Result<Vec<PlannedTowerTensor<'_>>, VisionTowerError> {
    let mut planned_by_name = BTreeMap::new();
    for (source_name, source_tensor) in &tower_inventory.tensors {
        if !is_vision_tower_tensor(source_name) {
            continue;
        }
        let Some(output_name) = mlx_tower_tensor_name(source_name) else {
            return Err(VisionTowerError::UnsupportedTowerNaming {
                tensor_name: source_name.clone(),
            });
        };
        let output_shape =
            if is_channels_first_patch_embedding(source_name, &source_tensor.entry.shape) {
                channels_last_shape(&source_tensor.entry.shape)
            } else {
                source_tensor.entry.shape.clone()
            };
        planned_by_name.insert(
            output_name.clone(),
            PlannedTowerTensor {
                output_name,
                output_shape,
                source_name,
                source_tensor,
            },
        );
    }
    Ok(planned_by_name.into_values().collect())
}

pub async fn write_tower_file(
    tower_source: &ArtifactSource,
    planned_tensors: &[PlannedTowerTensor<'_>],
    tower_file_path: &Path,
    progress: &mut dyn TowerFileProgress,
) -> Result<WrittenTowerFile, VisionTowerError> {
    let mut written_file = match File::create(tower_file_path) {
        Ok(created_file) => BufWriter::new(created_file),
        Err(write_error) => return Err(unwritable(tower_file_path, &write_error)),
    };
    let header_bytes = tower_file_header_bytes(planned_tensors);
    write_all(&mut written_file, &header_bytes, tower_file_path)?;

    let mut total_bytes = 0_u64;
    let mut parameter_count = 0_u64;
    for planned_tensor in planned_tensors {
        // Saturating, because these come from outside headers and only a report reads them.
        total_bytes = total_bytes.saturating_add(planned_tensor.source_tensor.entry.byte_count);
        parameter_count =
            parameter_count.saturating_add(planned_tensor.source_tensor.entry.element_count());
    }
    let mut completed_bytes = 0_u64;
    let mut tensor_names = Vec::with_capacity(planned_tensors.len());
    for planned_tensor in planned_tensors {
        let output_bytes = read_output_bytes(tower_source, planned_tensor).await?;
        write_all(&mut written_file, &output_bytes, tower_file_path)?;
        completed_bytes += output_bytes.len() as u64;
        tensor_names.push(planned_tensor.output_name.clone());
        progress.tower_bytes_written(completed_bytes, total_bytes);
    }
    if let Err(write_error) = written_file.flush() {
        return Err(unwritable(tower_file_path, &write_error));
    }
    Ok(WrittenTowerFile {
        tensor_names,
        parameter_count,
        tensor_byte_count: total_bytes,
    })
}

async fn read_output_bytes(
    tower_source: &ArtifactSource,
    planned_tensor: &PlannedTowerTensor<'_>,
) -> Result<Vec<u8>, VisionTowerError> {
    let source_tensor = planned_tensor.source_tensor;
    let source_bytes = tower_source
        .read_byte_range(
            &source_tensor.relative_path,
            source_tensor.entry.data_start,
            source_tensor.entry.byte_count,
        )
        .await?;
    if !is_channels_first_patch_embedding(planned_tensor.source_name, &source_tensor.entry.shape) {
        return Ok(source_bytes);
    }
    let Some(element_bytes) = element_byte_width(&source_tensor.entry.data_type) else {
        return Err(VisionTowerError::UnsupportedTowerNaming {
            tensor_name: planned_tensor.source_name.to_owned(),
        });
    };
    Ok(channels_first_to_channels_last(
        &source_bytes,
        &source_tensor.entry.shape,
        element_bytes as usize,
    ))
}

/// The length prefix and padded header for tensors written in plan order.
#[must_use]
pub fn tower_file_header_bytes(planned_tensors: &[PlannedTowerTensor<'_>]) -> Vec<u8> {
    let mut header_entries = serde_json::Map::new();
    header_entries.insert(
        "__metadata__".to_owned(),
        serde_json::json!({"format": TOWER_FILE_METADATA_FORMAT}),
    );
    let mut next_offset = 0_u64;
    for planned_tensor in planned_tensors {
        let entry = &planned_tensor.source_tensor.entry;
        let end_offset = next_offset.saturating_add(entry.byte_count);
        header_entries.insert(
            planned_tensor.output_name.clone(),
            serde_json::json!({
                "dtype": entry.data_type,
                "shape": planned_tensor.output_shape,
                "data_offsets": [next_offset, end_offset],
            }),
        );
        next_offset = end_offset;
    }
    let mut header_text = serde_json::Value::Object(header_entries).to_string();
    let header_end = HEADER_LENGTH_PREFIX_BYTES as usize + header_text.len();
    let padding =
        (HEADER_ALIGNMENT_BYTES - header_end % HEADER_ALIGNMENT_BYTES) % HEADER_ALIGNMENT_BYTES;
    header_text.push_str(&" ".repeat(padding));
    let mut file_start = (header_text.len() as u64).to_le_bytes().to_vec();
    file_start.extend_from_slice(header_text.as_bytes());
    file_start
}

fn write_all(
    written_file: &mut BufWriter<File>,
    file_bytes: &[u8],
    tower_file_path: &Path,
) -> Result<(), VisionTowerError> {
    match written_file.write_all(file_bytes) {
        Ok(()) => Ok(()),
        Err(write_error) => Err(unwritable(tower_file_path, &write_error)),
    }
}

fn unwritable(path: &Path, write_error: &std::io::Error) -> VisionTowerError {
    VisionTowerError::FileUnwritable {
        path: path.to_path_buf(),
        cause: write_error.to_string(),
    }
}
