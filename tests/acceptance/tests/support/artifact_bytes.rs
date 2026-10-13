//! Model-artifact byte accounting shared by real-model acceptance journeys.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Scans one model directory recursively and returns its regular-file payload.
///
/// Hugging Face cache layouts store snapshot files as symlinks into `blobs/`,
/// and `DirEntry::metadata` performs an `lstat` that never traverses them, so
/// the scan resolves every entry path with following semantics and keeps a
/// canonical-directory set as the cycle guard for symlinked subdirectories.
#[allow(dead_code)]
pub(crate) fn artifact_directory_regular_file_bytes(model_directory: &Path) -> u64 {
    let mut artifact_payload_bytes = 0_u64;
    let mut visited_directory_paths = HashSet::new();
    let mut pending_directories = vec![model_directory.to_path_buf()];
    while let Some(current_directory) = pending_directories.pop() {
        let canonical_directory =
            current_directory
                .canonicalize()
                .unwrap_or_else(|canonical_error| {
                    panic!("the discovered model directory should resolve: {canonical_error}")
                });
        if !visited_directory_paths.insert(canonical_directory) {
            continue;
        }
        let directory_entries = fs::read_dir(&current_directory).unwrap_or_else(|read_error| {
            panic!("the discovered model directory should be readable: {read_error}")
        });
        for directory_entry in directory_entries {
            let directory_entry = directory_entry.unwrap_or_else(|read_error| {
                panic!("a model directory entry should be readable: {read_error}")
            });
            let entry_path = directory_entry.path();
            // Path::metadata follows symlinks, unlike DirEntry::metadata.
            let entry_metadata = fs::metadata(&entry_path).unwrap_or_else(|read_error| {
                panic!("model file metadata should be readable: {read_error}")
            });
            if entry_metadata.is_dir() {
                pending_directories.push(entry_path);
            } else if entry_metadata.is_file() {
                artifact_payload_bytes = artifact_payload_bytes
                    .saturating_add(u64::try_from(entry_metadata.len()).unwrap_or(u64::MAX));
            }
        }
    }
    artifact_payload_bytes
}

/// Resolves the isolated-home model-directory scan root from the discovered
/// artifact directory. Discovery over a `models--<org>--<repo>` cache entry
/// registers the leaf model id; discovery over the `snapshots/<sha>` directory
/// itself would register the commit hash instead, so every request would fail
/// with model_not_found even though the worker loads fine.
pub(crate) fn acceptance_model_scan_root(model_directory: &Path) -> PathBuf {
    let cache_entry_directory =
        model_directory
            .parent()
            .and_then(Path::parent)
            .filter(|cache_entry| {
                cache_entry
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("models--"))
            });
    cache_entry_directory
        .map(Path::to_path_buf)
        .unwrap_or_else(|| model_directory.to_path_buf())
}
