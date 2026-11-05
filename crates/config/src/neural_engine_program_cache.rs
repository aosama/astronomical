//! Where compiled Neural Engine programs live, and what is there.
//!
//! The folder is one declared home, `runtime.neural_engine_program_directory`,
//! shared by a Mac's instances. macOS caches its Neural Engine compilation by a
//! program's path, so one path lets every instance reuse the first one's work, and a
//! set is about 10 GB for a dense 9B model, which is too much to keep twice. It sits
//! apart from the prompt cache, because that cache trims everything under its root
//! against the user's size limit, and compiled programs are not prompt state.
//!
//! Layout: `<directory>/<model id>/<revision>/<set name>/layer-N.mlmodelc`, with a
//! `programs.json` manifest beside the programs once a set is complete.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const NEURAL_ENGINE_PROGRAM_DIRECTORY_NAME: &str = "neural-engine-programs";
pub const NEURAL_ENGINE_PROGRAM_MANIFEST_FILE_NAME: &str = "programs.json";
const COMPILED_PROGRAM_SUFFIX: &str = ".mlmodelc";

/// One model revision's folder of compiled programs.
#[must_use]
pub fn neural_engine_program_cache_directory_for_model(
    neural_engine_program_directory: &Path,
    model_id: &str,
    model_revision: &str,
) -> PathBuf {
    neural_engine_program_directory
        .join(model_id)
        .join(model_revision)
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
