//! Where compiled Neural Engine programs live, and what is there.
//!
//! The folder is one declared home, `runtime.neural_engine_program_directory`, in the
//! instance's own folder by default. A set is about 10 GB for a dense 9B model, so it
//! should be easy to find. macOS keeps its own copies of about the same size outside
//! this folder, which nothing here manages or reports (issue #1878). The folder sits
//! apart from the prompt cache, because that cache trims everything under its root
//! against the user's size limit, and compiled programs are not prompt state.
//!
//! Layout: `<directory>/<model id>/<revision>-<weights fingerprint>/<set name>/layer-N.mlmodelc`,
//! with a `programs.json` manifest beside the programs once a set is complete.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const NEURAL_ENGINE_PROGRAM_DIRECTORY_NAME: &str = "neural-engine-programs";
pub const NEURAL_ENGINE_PROGRAM_MANIFEST_FILE_NAME: &str = "programs.json";
const COMPILED_PROGRAM_SUFFIX: &str = ".mlmodelc";
const WEIGHTS_FINGERPRINT_BYTES: usize = 8;

/// One model revision's folder of compiled programs, for these exact weights.
///
/// The revision only hashes `config.json`, and the model id is a folder name, so two
/// models or two quantizations can share both. A compiled program holds its weights, so
/// sharing a folder would silently run one model's weights in the other.
#[must_use]
pub fn neural_engine_program_cache_directory_for_model(
    neural_engine_program_directory: &Path,
    model_id: &str,
    model_revision: &str,
    weights_fingerprint: &str,
) -> PathBuf {
    neural_engine_program_directory
        .join(model_id)
        .join(format!("{model_revision}-{weights_fingerprint}"))
}

/// Removes this model's program folders for any other revision or weights, because copying
/// or replacing a model changes its fingerprint and a set is about 10 GB for a dense 9B
/// model. Two models that share an id replace each other's folders, which costs a rebuild
/// and nothing else. Only folders holding program sets go, so anything else someone keeps
/// here is left alone, and a failure costs disk and nothing else.
pub fn remove_other_weights_of_model(
    neural_engine_program_directory: &Path,
    model_id: &str,
    kept_folder: &Path,
) {
    if !is_one_folder_name(model_id) {
        return;
    }
    let model_directory = neural_engine_program_directory.join(model_id);
    let Ok(directory_entries) = fs::read_dir(&model_directory) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let weights_folder = directory_entry.path();
        if weights_folder != kept_folder
            && holds_program_sets(&weights_folder)
            && let Err(removal_error) = fs::remove_dir_all(&weights_folder)
        {
            tracing::warn!(error = %removal_error, "could not remove outdated Neural Engine programs");
        }
    }
}

/// An id such as `..` or an absolute path would point this removal outside the model's own
/// folder, so nothing but a plain name is ever acted on.
fn is_one_folder_name(model_id: &str) -> bool {
    let mut components = Path::new(model_id).components();
    matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none()
}

fn holds_program_sets(weights_folder: &Path) -> bool {
    let Ok(directory_entries) = fs::read_dir(weights_folder) else {
        return false;
    };
    for directory_entry in directory_entries.flatten() {
        let set_directory = directory_entry.path();
        if set_directory
            .join(NEURAL_ENGINE_PROGRAM_MANIFEST_FILE_NAME)
            .is_file()
            || set_directory
                .join(format!("layer-0{COMPILED_PROGRAM_SUFFIX}"))
                .exists()
        {
            return true;
        }
    }
    false
}

/// Identifies a model folder's weights from each shard's name, size and modification
/// time, or `None` when there are none to read. Hashing the contents would take a minute
/// on every load of a large model, and these change whenever the weights are replaced.
#[must_use]
pub fn model_weights_fingerprint(model_directory: &Path) -> Option<String> {
    let mut shard_descriptions = Vec::new();
    for directory_entry in std::fs::read_dir(model_directory).ok()?.flatten() {
        let shard_name = directory_entry.file_name().to_string_lossy().into_owned();
        if !shard_name.ends_with(".safetensors") {
            continue;
        }
        // Through the entry's path, because a Hugging Face snapshot links to its blobs.
        let shard_metadata = std::fs::metadata(directory_entry.path()).ok()?;
        let modified_seconds = shard_metadata
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();
        shard_descriptions.push(format!(
            "{shard_name}:{}:{modified_seconds}",
            shard_metadata.len()
        ));
    }
    if shard_descriptions.is_empty() {
        return None;
    }
    shard_descriptions.sort_unstable();
    let digest = Sha256::digest(shard_descriptions.join("\n").as_bytes());
    Some(astronomical_lowercase_hex::encode_lowercase_hex(
        &digest[..WEIGHTS_FINGERPRINT_BYTES],
    ))
}

/// Describes one finished set, so a person can tell what a folder is without running
/// anything.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NeuralEngineProgramSetManifest {
    pub model_id: String,
    pub model_revision: String,
    pub set_name: String,
    pub program_count: u64,
    pub byte_count: u64,
    pub macos_version: Option<String>,
    pub built_with: String,
}

/// One set found on disk, whether or not it has a manifest yet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ListedNeuralEngineProgramSet {
    pub model_id: String,
    pub model_revision: String,
    pub set_name: String,
    pub program_count: u64,
    pub byte_count: u64,
    /// False for a set still being built, or one a load was interrupted during.
    pub is_complete: bool,
}

/// Every set under the directory, sorted by path. A missing directory is an empty
/// list, because nothing has been built yet. Entries it cannot read are left out,
/// since this is a report and one unreadable folder should not hide the rest.
#[must_use]
pub fn list_neural_engine_program_sets(
    neural_engine_program_directory: &Path,
) -> Vec<ListedNeuralEngineProgramSet> {
    let mut program_sets = Vec::new();
    for (model_id, model_directory) in sorted_child_directories(neural_engine_program_directory) {
        for (model_revision, revision_directory) in sorted_child_directories(&model_directory) {
            for (set_name, set_directory) in sorted_child_directories(&revision_directory) {
                program_sets.push(ListedNeuralEngineProgramSet {
                    model_id: model_id.clone(),
                    model_revision: model_revision.clone(),
                    program_count: compiled_program_count(&set_directory),
                    byte_count: directory_byte_count(&set_directory),
                    is_complete: set_directory
                        .join(NEURAL_ENGINE_PROGRAM_MANIFEST_FILE_NAME)
                        .is_file(),
                    set_name,
                });
            }
        }
    }
    program_sets
}

/// Everything under a directory, following no symlinks, so a size is what the
/// folder itself holds.
#[must_use]
pub fn directory_byte_count(directory: &Path) -> u64 {
    let Ok(directory_entries) = fs::read_dir(directory) else {
        return 0;
    };
    let mut byte_count = 0_u64;
    for directory_entry in directory_entries.flatten() {
        let Ok(entry_metadata) = fs::symlink_metadata(directory_entry.path()) else {
            continue;
        };
        byte_count = byte_count.saturating_add(if entry_metadata.is_dir() {
            directory_byte_count(&directory_entry.path())
        } else {
            entry_metadata.len()
        });
    }
    byte_count
}

fn compiled_program_count(set_directory: &Path) -> u64 {
    let mut program_count = 0_u64;
    for (entry_name, _) in sorted_child_directories(set_directory) {
        if entry_name.ends_with(COMPILED_PROGRAM_SUFFIX) {
            program_count += 1;
        }
    }
    program_count
}

fn sorted_child_directories(directory: &Path) -> Vec<(String, PathBuf)> {
    let Ok(directory_entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut child_directories = Vec::new();
    for directory_entry in directory_entries.flatten() {
        if let Ok(entry_name) = directory_entry.file_name().into_string()
            && directory_entry.path().is_dir()
        {
            child_directories.push((entry_name, directory_entry.path()));
        }
    }
    child_directories.sort();
    child_directories
}

#[cfg(test)]
#[path = "neural_engine_program_cache_tests.rs"]
mod tests;
