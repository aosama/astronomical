//! A view of a bfloat16 source that the sensitivity probe can load.
//!
//! The probe loads its reference through the serving engine, and the engine
//! finds tensors through `model.safetensors.index.json`. A release small
//! enough for one file ships a bare `model.safetensors` and no index, so the
//! engine refuses the very checkpoints that are cheapest to measure. It
//! also refuses a model file that is a symbolic link, which is how the
//! Hugging Face cache stores every file of a downloaded release. Rather
//! than write into a source directory this crate does not own, the pipeline
//! builds a sibling directory of hard links to the source's files plus the
//! index the source lacks, and removes it when the sweep ends. Hard links
//! rather than symbolic ones: the engine's validator refuses a model file
//! that is a symbolic link, and a hard link is the file itself.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::artifact::json_file::write_pretty_json;
use crate::error::QuantizationError;
use crate::module_inventory::inventory_builder::list_shard_paths;
use crate::module_inventory::safetensors_header::read_header_tensors;
use crate::requantize::shard_writer::SHARD_INDEX_FILE_NAME;

#[cfg(test)]
#[path = "probe_source_view_tests.rs"]
mod probe_source_view_tests;

/// The view's directory name under the conversion's output directory.
pub const PROBE_SOURCE_VIEW_DIRECTORY_NAME: &str = "_indexed_source_view";

/// The index document in the Hugging Face layout.
#[derive(Serialize)]
struct ShardIndexDocument {
    metadata: ShardIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct ShardIndexMetadata {
    total_size: u64,
}

/// Returns a directory the probe can load `source_directory` from: the
/// source itself when it has a shard index and holds plain files, otherwise
/// a freshly built view under `output_directory`. The second value is
/// whether a view was built, which the caller removes after the sweep.
pub fn loadable_model_directory(
    source_directory: &Path,
    output_directory: &Path,
) -> Result<(PathBuf, bool), QuantizationError> {
    let has_shard_index = source_directory.join(SHARD_INDEX_FILE_NAME).is_file();
    if has_shard_index && !holds_symbolic_links(source_directory) {
        return Ok((source_directory.to_path_buf(), false));
    }
    let view_directory = output_directory.join(PROBE_SOURCE_VIEW_DIRECTORY_NAME);
    if view_directory.exists()
        && let Err(remove_error) = std::fs::remove_dir_all(&view_directory)
    {
        return Err(view_failure(&view_directory, &remove_error));
    }
    if let Err(create_error) = std::fs::create_dir_all(&view_directory) {
        return Err(view_failure(&view_directory, &create_error));
    }
    link_source_files(source_directory, &view_directory)?;
    if !has_shard_index {
        write_shard_index(source_directory, &view_directory)?;
    }
    Ok((view_directory, true))
}

/// Whether any top-level entry of the directory is a symbolic link.
fn holds_symbolic_links(source_directory: &Path) -> bool {
    let Ok(directory_entries) = std::fs::read_dir(source_directory) else {
        return false;
    };
    for directory_entry in directory_entries.flatten() {
        if let Ok(file_type) = directory_entry.file_type()
            && file_type.is_symlink()
        {
            return true;
        }
    }
    false
}

/// Hard-links every top-level source file into the view; the weights are
/// the whole size of the model, so they are never copied. A file that
/// cannot be linked (the output directory is on another volume) is copied
/// when it is not a weight shard, and reported otherwise.
fn link_source_files(
    source_directory: &Path,
    view_directory: &Path,
) -> Result<(), QuantizationError> {
    let directory_entries = match std::fs::read_dir(source_directory) {
        Ok(directory_entries) => directory_entries,
        Err(read_error) => return Err(view_failure(view_directory, &read_error)),
    };
    for directory_entry in directory_entries.flatten() {
        let source_path = directory_entry.path();
        if !source_path.is_file() {
            continue;
        }
        let link_target = match std::fs::canonicalize(&source_path) {
            Ok(link_target) => link_target,
            Err(resolve_error) => return Err(view_failure(view_directory, &resolve_error)),
        };
        let link_path = view_directory.join(directory_entry.file_name());
        let Err(link_error) = std::fs::hard_link(&link_target, &link_path) else {
            continue;
        };
        let is_weight_shard =
            link_path.extension().and_then(std::ffi::OsStr::to_str) == Some("safetensors");
        if is_weight_shard {
            return Err(QuantizationError::ArtifactWriteFailure {
                output_directory: view_directory.to_path_buf(),
                description: format!(
                    "the source has no shard index and its weights could not be hard-linked into a view for the probe ({link_error}); choose an output directory on the same volume as the source"
                ),
            });
        }
        if let Err(copy_error) = std::fs::copy(&link_target, &link_path) {
            return Err(view_failure(view_directory, &copy_error));
        }
    }
    Ok(())
}

/// Writes the index every shard's header implies.
fn write_shard_index(
    source_directory: &Path,
    view_directory: &Path,
) -> Result<(), QuantizationError> {
    let mut weight_map = BTreeMap::new();
    let mut total_size = 0_u64;
    for shard_path in list_shard_paths(source_directory)? {
        let shard_file_name = shard_path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or_default()
            .to_owned();
        for header_tensor in read_header_tensors(&shard_path)? {
            total_size += header_tensor.data_byte_count;
            weight_map.insert(header_tensor.tensor_name, shard_file_name.clone());
        }
    }
    let shard_index = ShardIndexDocument {
        metadata: ShardIndexMetadata { total_size },
        weight_map,
    };
    write_pretty_json(&shard_index, &view_directory.join(SHARD_INDEX_FILE_NAME))
}

fn view_failure(view_directory: &Path, failure: &std::io::Error) -> QuantizationError {
    QuantizationError::ArtifactWriteFailure {
        output_directory: view_directory.to_path_buf(),
        description: format!("the indexed source view was not built: {failure}"),
    }
}
