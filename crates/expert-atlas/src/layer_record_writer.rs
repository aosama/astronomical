//! Writes one expert layer's record file and proves it holds the bytes read.
//!
//! Each source tensor is read whole (one large direct read) and its
//! per-expert slices are scattered into the layer's record buffer, which is
//! then written in one direct write. The written file is read back and
//! compared with the buffer, so an Atlas reports success only for bytes that
//! reached the disk intact; its SHA-256 goes into the manifest.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::atlas_manifest::{AtlasDeclaredFile, AtlasExpertLayer};
use crate::source_checkpoint::{SourceExpertLayer, SourceTensor};
use crate::uncached_file_io::{AlignedSpan, PageAlignedBuffer, create_uncached, open_uncached};

#[derive(Debug, thiserror::Error)]
pub enum LayerRecordWriteError {
    #[error("layer {layer_index}: {path} could not be accessed: {source}")]
    Io {
        layer_index: usize,
        path: String,
        source: std::io::Error,
    },
    #[error(
        "layer {layer_index}: {path} does not hold the bytes written (first difference at byte {first_difference})"
    )]
    ReadBackMismatch {
        layer_index: usize,
        path: String,
        first_difference: usize,
    },
}

/// Opens each source shard once for a whole conversion.
#[derive(Default)]
pub struct SourceShardFiles {
    open_files_by_name: HashMap<String, File>,
}

impl SourceShardFiles {
    pub(crate) fn file(
        &mut self,
        model_directory: &Path,
        shard_file_name: &str,
    ) -> std::io::Result<&File> {
        if !self.open_files_by_name.contains_key(shard_file_name) {
            let shard_file = open_uncached(&model_directory.join(shard_file_name))?;
            self.open_files_by_name
                .insert(shard_file_name.to_owned(), shard_file);
        }
        Ok(&self.open_files_by_name[shard_file_name])
    }
}

/// Writes `planned_layer` into `atlas_directory` and returns the file it
/// declares. Nothing is written outside `atlas_directory`.
pub fn write_layer_records(
    source_model_directory: &Path,
    source_expert_layer: &SourceExpertLayer,
    planned_layer: &AtlasExpertLayer,
    atlas_directory: &Path,
    source_shard_files: &mut SourceShardFiles,
) -> Result<AtlasDeclaredFile, LayerRecordWriteError> {
    let layer_index = planned_layer.layer_index;
    let mut layer_records = PageAlignedBuffer::new(planned_layer.file_byte_count());
    for slot in &planned_layer.slots {
        let source_tensor = source_tensor_named(source_expert_layer, &slot.tensor_name);
        let shard_path = source_model_directory.join(&source_tensor.shard_file_name);
        let io_error = |source| LayerRecordWriteError::Io {
            layer_index,
            path: shard_path.display().to_string(),
            source,
        };
        let shard_file = source_shard_files
            .file(source_model_directory, &source_tensor.shard_file_name)
            .map_err(io_error)?;
        let expert_batched_tensor = AlignedSpan::read(
            shard_file,
            source_tensor.payload_file_offset,
            source_tensor.payload_byte_count,
        )
        .map_err(io_error)?;
        scatter_into_records(
            expert_batched_tensor.bytes(),
            planned_layer,
            slot,
            layer_records.bytes_mut(),
        );
    }
    let layer_path = atlas_directory.join(&planned_layer.file_name);
    let layer_byte_count = planned_layer.file_byte_count() as usize;
    write_and_read_back(
        &layer_path,
        &layer_records.bytes()[..layer_byte_count],
        layer_index,
    )?;
    Ok(AtlasDeclaredFile {
        file_name: planned_layer.file_name.clone(),
        expected_file_byte_count: layer_byte_count as u64,
        sha256: hex_sha256(&layer_records.bytes()[..layer_byte_count]),
    })
}

fn source_tensor_named<'layer>(
    source_expert_layer: &'layer SourceExpertLayer,
    tensor_name: &str,
) -> &'layer SourceTensor {
    source_expert_layer
        .expert_tensors
        .iter()
        .find(|source_tensor| source_tensor.tensor_name == tensor_name)
        .expect("every planned slot comes from one of the layer's source tensors")
}

fn scatter_into_records(
    expert_batched_tensor: &[u8],
    planned_layer: &AtlasExpertLayer,
    slot: &crate::atlas_manifest::AtlasRecordSlot,
    layer_records: &mut [u8],
) {
    let bytes_per_expert = slot.bytes_per_expert as usize;
    for (expert_id, expert_slice) in expert_batched_tensor
        .chunks_exact(bytes_per_expert)
        .enumerate()
    {
        let destination_start = planned_layer.slot_file_offset(expert_id, slot) as usize;
        layer_records[destination_start..destination_start + bytes_per_expert]
            .copy_from_slice(expert_slice);
    }
}

/// One direct write, a sync, then a direct read-back compared byte for byte.
fn write_and_read_back(
    layer_path: &Path,
    layer_bytes: &[u8],
    layer_index: usize,
) -> Result<(), LayerRecordWriteError> {
    let path = layer_path.display().to_string();
    let io_error = |source| LayerRecordWriteError::Io {
        layer_index,
        path: path.clone(),
        source,
    };
    let layer_file = create_uncached(layer_path).map_err(io_error)?;
    layer_file.write_all_at(layer_bytes, 0).map_err(io_error)?;
    layer_file.sync_all().map_err(io_error)?;
    let written_bytes =
        AlignedSpan::read(&layer_file, 0, layer_bytes.len() as u64).map_err(io_error)?;
    if let Some(first_difference) = written_bytes
        .bytes()
        .iter()
        .zip(layer_bytes)
        .position(|(written, expected)| written != expected)
    {
        return Err(LayerRecordWriteError::ReadBackMismatch {
            layer_index,
            path,
            first_difference,
        });
    }
    Ok(())
}

pub(crate) fn hex_sha256(file_bytes: &[u8]) -> String {
    astronomical_lowercase_hex::encode_lowercase_hex(&Sha256::digest(file_bytes))
}
