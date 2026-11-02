//! Where each artifact document lives: OptiQ-specific files under `optiq/`,
//! with the legacy root-level metadata name still accepted on read.

use std::path::{Path, PathBuf};

#[cfg(test)]
#[path = "layout_tests.rs"]
mod layout_tests;

/// The subfolder holding everything OptiQ-specific.
pub const OPTIQ_SUBDIRECTORY: &str = "optiq";
pub const METADATA_FILE_NAME: &str = "metadata.json";
pub const SENSITIVITY_FILE_NAME: &str = "sensitivity.json";
/// The pre-0.4.18 root-level metadata name that published quants still carry
/// and that readers must keep accepting.
pub const LEGACY_METADATA_FILE_NAME: &str = "optiq_metadata.json";

/// Where `name` is written for an artifact at `model_directory`.
#[must_use]
pub fn artifact_path(model_directory: &Path, file_name: &str) -> PathBuf {
    model_directory.join(OPTIQ_SUBDIRECTORY).join(file_name)
}

/// An existing path for the metadata document, canonical location first.
#[must_use]
pub fn resolve_metadata_path(model_directory: &Path) -> Option<PathBuf> {
    let canonical = artifact_path(model_directory, METADATA_FILE_NAME);
    if canonical.is_file() {
        return Some(canonical);
    }
    let legacy = model_directory.join(LEGACY_METADATA_FILE_NAME);
    legacy.is_file().then_some(legacy)
}
