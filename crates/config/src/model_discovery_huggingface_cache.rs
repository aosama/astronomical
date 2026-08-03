use std::fs;
use std::path::{Path, PathBuf};

use crate::decode_huggingface_cache_directory_name;

/// A Hugging Face cache entry resolved to its active snapshot directory.
pub(super) struct HuggingFaceCacheEntry {
    pub(super) model_id: String,
    pub(super) snapshot_directory: PathBuf,
}

/// Resolves one Hugging Face cache entry to its active snapshot directory.
///
/// Prefer the `main` or `master` reference. If neither reference identifies an
/// existing snapshot, use the most recently modified snapshot as a local-only
/// fallback.
pub(super) fn resolve_huggingface_cache_entry(
    huggingface_cache_directory: &Path,
) -> Option<HuggingFaceCacheEntry> {
    let directory_name = huggingface_cache_directory.file_name()?.to_str()?;
    let decoded_model_id = decode_huggingface_cache_directory_name(directory_name)?;
    let snapshots_directory = huggingface_cache_directory.join("snapshots");
    if !snapshots_directory.is_dir() {
        return None;
    }

    for reference_name in ["main", "master"] {
        let reference_path = huggingface_cache_directory
            .join("refs")
            .join(reference_name);
        if let Ok(commit_hash) = fs::read_to_string(&reference_path) {
            let snapshot_directory = snapshots_directory.join(commit_hash.trim());
            if snapshot_directory.is_dir() {
                return Some(HuggingFaceCacheEntry {
                    model_id: leaf_model_id(&decoded_model_id),
                    snapshot_directory,
                });
            }
        }
    }

    let mut snapshot_directories_by_modified_time = fs::read_dir(&snapshots_directory)
        .ok()?
        .filter_map(|directory_entry| {
            let directory_entry = directory_entry.ok()?;
            let directory_metadata = directory_entry.metadata().ok()?;
            if !directory_metadata.is_dir() {
                return None;
            }
            Some((directory_entry.path(), directory_metadata.modified().ok()?))
        })
        .collect::<Vec<_>>();
    snapshot_directories_by_modified_time
        .sort_by_key(|(_, modified_time)| std::cmp::Reverse(*modified_time));
    let (snapshot_directory, _) = snapshot_directories_by_modified_time.into_iter().next()?;
    Some(HuggingFaceCacheEntry {
        model_id: leaf_model_id(&decoded_model_id),
        snapshot_directory,
    })
}

fn leaf_model_id(decoded_model_id: &str) -> String {
    decoded_model_id.split_once('/').map_or_else(
        || decoded_model_id.to_owned(),
        |(_, leaf_model_id)| leaf_model_id.to_owned(),
    )
}
