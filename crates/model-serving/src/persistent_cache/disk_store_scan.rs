//! Startup scan: discovers, validates, and cleans persistent prompt-cache files
//! during `PersistentPromptCacheDiskStore::open()`.

use std::path::Path;

use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, open_without_following_symlinks,
    parse_persistent_prompt_cache_file_hash_from_path, remove_cache_owned_file_or_confirm_absent,
};
use super::disk_store_index::{
    PersistentPromptCacheDiskStoreIndex, TrackedPersistentPromptCacheFile,
};

pub(crate) fn scan_current_format_directory<HeaderValidator>(
    directory: &Path,
    file_kind: PersistentPromptCacheFileKind,
    tracked_files: &mut PersistentPromptCacheDiskStoreIndex,
    header_validator: HeaderValidator,
) -> Result<(), PersistentPromptCacheDiskStoreError>
where
    HeaderValidator: Fn(&std::fs::File, &Path) -> bool,
{
    let directory_entries = std::fs::read_dir(directory).map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
            persistent_prompt_cache_directory: directory.to_path_buf(),
            source,
        }
    })?;
    for directory_entry_result in directory_entries {
        let directory_entry = directory_entry_result.map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                persistent_prompt_cache_directory: directory.to_path_buf(),
                source,
            }
        })?;
        let entry_path = directory_entry.path();
        let entry_file_type = directory_entry.file_type().map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: entry_path.clone(),
                source,
            }
        })?;
        if entry_path
            .extension()
            .is_some_and(|extension| extension == "tmp")
        {
            remove_cache_owned_file_or_confirm_absent(&entry_path)?;
            continue;
        }
        if !entry_file_type.is_file()
            || entry_path
                .extension()
                .is_none_or(|ext| ext != "safetensors")
        {
            continue;
        }
        let Some(persistent_prompt_cache_file_hash) =
            parse_persistent_prompt_cache_file_hash_from_path(&entry_path)
        else {
            remove_cache_owned_file_or_confirm_absent(&entry_path)?;
            continue;
        };
        let file_size_bytes = std::fs::symlink_metadata(&entry_path)
            .map(|metadata| metadata.len())
            .map_err(
                |source| PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: entry_path.clone(),
                    source,
                },
            )?;
        let file = open_without_following_symlinks(&entry_path).map_err(|source| {
            PersistentPromptCacheDiskStoreError::OpenBlockFile {
                block_file_path: entry_path.clone(),
                source,
            }
        })?;
        if !header_validator(&file, &entry_path) {
            remove_cache_owned_file_or_confirm_absent(&entry_path)?;
            continue;
        }
        tracked_files.insert_file(
            file_kind,
            persistent_prompt_cache_file_hash,
            TrackedPersistentPromptCacheFile {
                file_path: entry_path,
                file_size_bytes,
            },
        );
    }
    Ok(())
}
