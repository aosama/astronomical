//! Resident weight-bundle publication for converted streaming revisions.
//!
//! The resident class owns every tensor the model needs on every token
//! (attention, norms, embeddings, LM head, routers), so it ships as ONE
//! standard safetensors file and rides the loader's natural path. The
//! per-expert pack machinery exists purely for the paging class and never
//! touches these tensors.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use astronomical_model_serving::{
    QuantizedExpertLayerPlan, TensorHeaderEntry, parse_safetensors_header,
};

use crate::aligned_expert_pack_preparer::AlignedExpertPackPreparationError;
use crate::revision_manifest::resident_file_entry;

/// Publishes `resident.safetensors` into the staging revision: every
/// non-expert tensor from the source language shards, copied byte-for-byte
/// into one standard safetensors file at the revision root.
///
/// The resident class owns everything the model needs on every token, so it
/// stays on the loader's natural standard path; the pack machinery exists
/// purely for the paging class.
pub(super) fn publish_resident_weights(
    source_model_directory: &Path,
    layer_plans: &[QuantizedExpertLayerPlan],
    staging_model_directory: &Path,
) -> Result<(), AlignedExpertPackPreparationError> {
    let expert_tensor_names: HashSet<&str> = layer_plans
        .iter()
        .flat_map(|layer_plan| layer_plan.tensor_sources.iter())
        .map(|tensor_source| tensor_source.tensor_name.as_str())
        .collect();
    let mut shard_paths: Vec<PathBuf> = fs::read_dir(source_model_directory)?
        .filter_map(|directory_entry| directory_entry.ok())
        .map(|directory_entry| directory_entry.path())
        .filter(|source_path| {
            source_path.is_file()
                && source_path
                    .file_name()
                    .and_then(|file_name| file_name.to_str())
                    .is_some_and(|file_name_text| file_name_text.ends_with(".safetensors"))
        })
        .collect();
    shard_paths.sort();
    let mut resident_tensor_entries: Vec<(TensorHeaderEntry, PathBuf)> = Vec::new();
    for shard_path in &shard_paths {
        let shard_header = parse_safetensors_header(shard_path)?;
        for tensor_entry in shard_header.tensor_entries {
            if expert_tensor_names.contains(tensor_entry.tensor_name.as_str()) {
                continue;
            }
            resident_tensor_entries.push((tensor_entry, shard_path.clone()));
        }
    }
    resident_tensor_entries.sort_by(|left, right| left.0.tensor_name.cmp(&right.0.tensor_name));
    let total_resident_tensor_count = resident_tensor_entries.len();

    let mut header_mapping = serde_json::Map::with_capacity(total_resident_tensor_count);
    let mut payload_cursor_bytes: u64 = 0;
    for (tensor_entry, _) in &resident_tensor_entries {
        let tensor_byte_count = tensor_entry.data_end_offset - tensor_entry.data_start_offset;
        header_mapping.insert(
            tensor_entry.tensor_name.clone(),
            serde_json::json!({
                "dtype": tensor_entry.dtype.as_str(),
                "shape": tensor_entry.shape,
                "data_offsets": [payload_cursor_bytes, payload_cursor_bytes + tensor_byte_count],
            }),
        );
        payload_cursor_bytes += tensor_byte_count;
    }
    let mut header_json_bytes = serde_json::to_vec(&serde_json::Value::Object(header_mapping))?;
    // Safetensors requires the header length to be a multiple of eight.
    let header_padding_byte_count = (8 - header_json_bytes.len() % 8) % 8;
    header_json_bytes.extend(std::iter::repeat_n(b' ', header_padding_byte_count));
    let expected_file_byte_count = 8 + header_json_bytes.len() as u64 + payload_cursor_bytes;

    let resident_safetensors_path = staging_model_directory.join("resident.safetensors");
    if resident_safetensors_path.exists()
        && fs::metadata(&resident_safetensors_path)?.len() == expected_file_byte_count
    {
        return Ok(());
    }
    let resident_tensor_source_files = resident_tensor_entries
        .iter()
        .map(|(_, shard_path)| shard_path.clone())
        .collect::<HashSet<_>>();
    let mut shard_file_handles = HashMap::with_capacity(resident_tensor_source_files.len());
    for shard_path in resident_tensor_source_files {
        shard_file_handles.insert(shard_path.clone(), fs::File::open(&shard_path)?);
    }
    let mut destination_writer = BufWriter::new(fs::File::create(&resident_safetensors_path)?);
    destination_writer.write_all(&(header_json_bytes.len() as u64).to_le_bytes())?;
    destination_writer.write_all(&header_json_bytes)?;
    for (copied_tensor_count, (tensor_entry, shard_path)) in
        resident_tensor_entries.iter().enumerate()
    {
        let tensor_byte_count = tensor_entry.data_end_offset - tensor_entry.data_start_offset;
        let shard_file = shard_file_handles
            .get_mut(shard_path)
            .expect("the shard handle for this tensor was opened above");
        shard_file.seek(SeekFrom::Start(tensor_entry.data_start_offset))?;
        let copied_byte_count = std::io::copy(
            &mut shard_file.take(tensor_byte_count),
            &mut destination_writer,
        )?;
        if copied_byte_count != tensor_byte_count {
            return Err(AlignedExpertPackPreparationError::ResidentTensorTruncated {
                tensor_name: tensor_entry.tensor_name.clone(),
                expected_byte_count: tensor_byte_count,
                copied_byte_count,
            });
        }
        let copied_tensor_count = copied_tensor_count + 1;
        if copied_tensor_count == 1
            || copied_tensor_count == total_resident_tensor_count
            || copied_tensor_count.is_multiple_of(128)
        {
            eprintln!(
                "status=resident_tensors copied_tensors={copied_tensor_count}/{total_resident_tensor_count}"
            );
        }
    }
    destination_writer.flush()?;
    Ok(())
}

/// Removes root shard files staged by the superseded transitional layout.

/// Recursively declares every revision file (relative path) as a resident
/// manifest entry: config, tokenizer, chat template, the vision bundle, and
/// the resident weight bundle itself.
pub(super) fn append_declared_resident_files(
    revision_directory: &Path,
    walk_directory: &Path,
    resident_files: &mut Vec<crate::revision_manifest::StreamingModelResidentFile>,
) -> Result<(), AlignedExpertPackPreparationError> {
    for directory_entry in fs::read_dir(walk_directory)? {
        let directory_entry = directory_entry?;
        let entry_path = directory_entry.path();
        if entry_path.is_dir() {
            append_declared_resident_files(revision_directory, &entry_path, resident_files)?;
            continue;
        }
        let relative_path = entry_path
            .strip_prefix(revision_directory)
            .expect("the walk stays under the revision directory");
        let file_name_text = relative_path.to_string_lossy().replace('\\', "/");
        if file_name_text == "manifest.json" {
            continue;
        }
        resident_files.push(resident_file_entry(revision_directory, &file_name_text)?);
    }
    Ok(())
}
