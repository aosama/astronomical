//! Whether a model written to a folder would be found by discovery.
//!
//! Tools that write a new model folder (the decision tools' calibrated and
//! trained copies) need to know before doing minutes of work whether
//! Astronomical will serve what they write. The answer follows the scan in
//! `scan_directory_recursive` exactly, so it lives beside it: a folder is
//! found when it sits below a scanned model folder, no deeper than the scan
//! goes, with no hidden folder, Hugging Face cache entry, model folder or
//! pipeline folder on the way down (the scan never descends into those).

use std::path::{Component, Path};

use super::MAXIMUM_MODEL_SCAN_DEPTH;
use super::model_family_classification::has_model_root_marker;

/// Whether discovery, scanning `model_root`, would find a model written to
/// `model_folder`. Both paths must be absolute with links resolved, so a
/// root reached through a symbolic link compares equal; the folder itself
/// and its missing parents need not exist yet.
pub fn is_discoverable_model_folder(model_root: &Path, model_folder: &Path) -> bool {
    let Ok(relative_folder) = model_folder.strip_prefix(model_root) else {
        return false;
    };
    let mut folder_names = Vec::new();
    for component in relative_folder.components() {
        let Component::Normal(folder_name) = component else {
            return false;
        };
        folder_names.push(folder_name.to_string_lossy());
    }
    // The root is scanned at depth 0, so a model folder at depth n has n names.
    if folder_names.is_empty() || folder_names.len() > MAXIMUM_MODEL_SCAN_DEPTH {
        return false;
    }
    for folder_name in &folder_names {
        if folder_name.starts_with('.') || folder_name.starts_with("models--") {
            return false;
        }
    }
    // Every folder the scan passes through on the way, the root included,
    // must be one it descends into rather than stops at
    // (a model's marker file, pipeline roots included, stops the scan).
    let mut scanned_folder = model_root.to_path_buf();
    if has_model_root_marker(&scanned_folder) {
        return false;
    }
    for folder_name in &folder_names[..folder_names.len() - 1] {
        scanned_folder.push(folder_name.as_ref());
        if has_model_root_marker(&scanned_folder) {
            return false;
        }
    }
    true
}

#[cfg(test)]
#[path = "model_folder_placement_tests.rs"]
mod tests;
