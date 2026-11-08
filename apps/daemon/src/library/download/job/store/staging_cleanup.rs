//! Startup tidy-up of the hidden `.incomplete` staging tree.
//!
//! Versions before the publication tidy-up (#1929) left an empty `.incomplete/<organisation>`
//! folder behind every published model. They are harmless but are clutter people find when they
//! look inside their models folder, so startup removes them. Only folders that are empty, or that
//! hold nothing but Finder's `.DS_Store`, are removed: a paused or failed download keeps its
//! staged bytes, so it can still be resumed.

use std::{fs, path::Path};

use super::{
    DownloadJobStore, DownloadJobStoreError,
    filesystem::{metadata_without_symlink, remove_directory_if_empty, synchronize_directory},
};

/// Finder writes this file into any folder someone opens; it holds view settings, never data.
const FINDER_METADATA_FILE_NAME: &str = ".DS_Store";

impl DownloadJobStore {
    /// Removes empty organisation folders under `.incomplete`, then `.incomplete` itself when
    /// nothing is left in it.
    pub fn remove_empty_staging_folders(&self) -> Result<(), DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        let incomplete_directory = self.models_directory.join(".incomplete");
        let Some(incomplete_metadata) = metadata_without_symlink(&incomplete_directory)? else {
            return Ok(());
        };
        if !incomplete_metadata.is_dir() {
            return Ok(());
        }
        let mut removed_any_folder = false;
        for staging_entry in read_directory(&incomplete_directory)? {
            // The entry's own type does not follow links, so a symlink is skipped rather than
            // refused: it is not ours to remove, and neither is what it points to.
            let is_real_directory = staging_entry
                .file_type()
                .map_err(|source| DownloadJobStoreError::InspectPath {
                    path: staging_entry.path(),
                    source,
                })?
                .is_dir();
            if is_real_directory {
                removed_any_folder |= remove_folder_without_data(&staging_entry.path())?;
            }
        }
        if removed_any_folder {
            synchronize_directory(&incomplete_directory)?;
        }
        if remove_folder_without_data(&incomplete_directory)? {
            synchronize_directory(&self.models_directory)?;
        }
        Ok(())
    }
}

/// Removes `folder` when it holds nothing, or nothing but Finder metadata. Returns whether it
/// was removed; a folder with any other entry is left untouched.
fn remove_folder_without_data(folder: &Path) -> Result<bool, DownloadJobStoreError> {
    let mut finder_metadata_files = Vec::new();
    for folder_entry in read_directory(folder)? {
        if folder_entry.file_name() != FINDER_METADATA_FILE_NAME {
            return Ok(false);
        }
        finder_metadata_files.push(folder_entry.path());
    }
    for finder_metadata_file in finder_metadata_files {
        fs::remove_file(&finder_metadata_file).map_err(|source| {
            DownloadJobStoreError::RemovePath {
                path: finder_metadata_file.clone(),
                source,
            }
        })?;
    }
    remove_directory_if_empty(folder)
}

fn read_directory(folder: &Path) -> Result<Vec<fs::DirEntry>, DownloadJobStoreError> {
    let mut folder_entries = Vec::new();
    let directory_reader =
        fs::read_dir(folder).map_err(|source| DownloadJobStoreError::InspectPath {
            path: folder.to_path_buf(),
            source,
        })?;
    for folder_entry in directory_reader {
        folder_entries.push(
            folder_entry.map_err(|source| DownloadJobStoreError::InspectPath {
                path: folder.to_path_buf(),
                source,
            })?,
        );
    }
    Ok(folder_entries)
}

#[cfg(test)]
#[path = "staging_cleanup_tests.rs"]
mod staging_cleanup_tests;
