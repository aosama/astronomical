//! Safe deletion of global or model-scoped persistent prompt-cache trees.
//!
//! The worker is the sole owner of this operation. The supervisor sends an IPC
//! command and never reads or mutates cache files directly.

use std::fs;
use std::path::{Component, Path, PathBuf};

use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    remove_cache_owned_directory_or_confirm_absent, remove_cache_owned_file_or_confirm_absent,
};
use super::disk_store_global_quota::reject_parent_directory_components;
use super::disk_store_index::PersistentPromptCacheDiskStoreIndex;

const BLOCKS_DIRECTORY_NAME: &str = "blocks";

/// Measured SSD space and prompt-cache blocks removed by one clear operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCacheClearOutcome {
    pub model_id: Option<String>,
    pub blocks_removed: u64,
    pub bytes_freed: u64,
}

/// Deletes either every cache namespace or all revisions of one model.
///
/// Model identities may contain multiple normal path components, such as
/// `organization/model`, because that is the cache layout used by model IDs.
/// Absolute paths, `.` and `..` are rejected before any filesystem access.
pub fn clear_persistent_prompt_cache_directory(
    global_prompt_cache_root_directory: &Path,
    model_id: Option<&str>,
) -> Result<PersistentPromptCacheClearOutcome, PersistentPromptCacheDiskStoreError> {
    let clear_target_directory =
        clear_target_directory(global_prompt_cache_root_directory, model_id)?;
    let Some(clear_target_directory) = clear_target_directory else {
        return Ok(PersistentPromptCacheClearOutcome {
            model_id: model_id.map(str::to_owned),
            blocks_removed: 0,
            bytes_freed: 0,
        });
    };
    let (blocks_removed, bytes_freed) = measure_clear_target(&clear_target_directory)?;
    if model_id.is_some() {
        remove_cache_owned_directory_or_confirm_absent(&clear_target_directory)?;
    } else {
        remove_global_root_contents(&clear_target_directory)?;
    }
    Ok(PersistentPromptCacheClearOutcome {
        model_id: model_id.map(str::to_owned),
        blocks_removed,
        bytes_freed,
    })
}

fn clear_target_directory(
    global_prompt_cache_root_directory: &Path,
    model_id: Option<&str>,
) -> Result<Option<PathBuf>, PersistentPromptCacheDiskStoreError> {
    reject_parent_directory_components(global_prompt_cache_root_directory)?;
    if !verify_existing_real_directory(global_prompt_cache_root_directory)? {
        return Ok(None);
    }
    let Some(model_id) = model_id else {
        return Ok(Some(global_prompt_cache_root_directory.to_path_buf()));
    };
    let model_id_path = Path::new(model_id);
    if model_id.is_empty()
        || model_id.contains('\0')
        || model_id.contains('\\')
        || model_id_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(
            PersistentPromptCacheDiskStoreError::UnsafePromptCacheDirectory {
                persistent_prompt_cache_directory: model_id_path.to_path_buf(),
            },
        );
    }
    let model_cache_directory = global_prompt_cache_root_directory.join(model_id_path);
    let model_cache_directory_exists =
        verify_descendant_components(global_prompt_cache_root_directory, model_id_path)?;
    Ok(model_cache_directory_exists.then_some(model_cache_directory))
}

fn verify_descendant_components(
    global_prompt_cache_root_directory: &Path,
    descendant_relative_path: &Path,
) -> Result<bool, PersistentPromptCacheDiskStoreError> {
    let mut current_directory = global_prompt_cache_root_directory.to_path_buf();
    for path_component in descendant_relative_path.components() {
        let Component::Normal(directory_name) = path_component else {
            return Err(
                PersistentPromptCacheDiskStoreError::UnsafePromptCacheDirectory {
                    persistent_prompt_cache_directory: descendant_relative_path.to_path_buf(),
                },
            );
        };
        current_directory.push(directory_name);
        match fs::symlink_metadata(&current_directory) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(_) => {
                return Err(
                    PersistentPromptCacheDiskStoreError::UnsafePromptCacheDirectory {
                        persistent_prompt_cache_directory: current_directory,
                    },
                );
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(source) => {
                return Err(PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: current_directory,
                    source,
                });
            }
        }
    }
    Ok(true)
}

fn verify_existing_real_directory(
    directory_path: &Path,
) -> Result<bool, PersistentPromptCacheDiskStoreError> {
    match fs::symlink_metadata(directory_path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(true),
        Ok(_) => Err(
            PersistentPromptCacheDiskStoreError::UnsafePromptCacheDirectory {
                persistent_prompt_cache_directory: directory_path.to_path_buf(),
            },
        ),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
            block_file_path: directory_path.to_path_buf(),
            source,
        }),
    }
}

fn measure_clear_target(
    clear_target_directory: &Path,
) -> Result<(u64, u64), PersistentPromptCacheDiskStoreError> {
    let mut blocks_removed = 0_u64;
    let mut bytes_freed = 0_u64;
    measure_directory_contents(
        clear_target_directory,
        false,
        &mut blocks_removed,
        &mut bytes_freed,
    )?;
    Ok((blocks_removed, bytes_freed))
}

fn measure_directory_contents(
    directory_path: &Path,
    directory_contains_blocks: bool,
    blocks_removed: &mut u64,
    bytes_freed: &mut u64,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    let directory_entries = fs::read_dir(directory_path).map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
            persistent_prompt_cache_directory: directory_path.to_path_buf(),
            source,
        }
    })?;
    for directory_entry in directory_entries {
        let directory_entry = directory_entry.map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                persistent_prompt_cache_directory: directory_path.to_path_buf(),
                source,
            }
        })?;
        let entry_path = directory_entry.path();
        let entry_metadata = fs::symlink_metadata(&entry_path).map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: entry_path.clone(),
                source,
            }
        })?;
        if entry_metadata.file_type().is_symlink() {
            *bytes_freed = bytes_freed.saturating_add(entry_metadata.len());
            continue;
        }
        if entry_metadata.is_file() {
            *bytes_freed = bytes_freed.saturating_add(entry_metadata.len());
            continue;
        }
        if entry_metadata.is_dir() {
            if directory_contains_blocks {
                *blocks_removed = blocks_removed.saturating_add(1);
            }
            let child_contains_blocks = directory_entry.file_name() == BLOCKS_DIRECTORY_NAME;
            measure_directory_contents(
                &entry_path,
                child_contains_blocks,
                blocks_removed,
                bytes_freed,
            )?;
        }
    }
    Ok(())
}

fn remove_global_root_contents(
    global_prompt_cache_root_directory: &Path,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    let directory_entries = fs::read_dir(global_prompt_cache_root_directory).map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
            persistent_prompt_cache_directory: global_prompt_cache_root_directory.to_path_buf(),
            source,
        }
    })?;
    for directory_entry in directory_entries {
        let directory_entry = directory_entry.map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                persistent_prompt_cache_directory: global_prompt_cache_root_directory.to_path_buf(),
                source,
            }
        })?;
        let entry_path = directory_entry.path();
        let entry_metadata = fs::symlink_metadata(&entry_path).map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: entry_path.clone(),
                source,
            }
        })?;
        if entry_metadata.is_dir() && !entry_metadata.file_type().is_symlink() {
            remove_cache_owned_directory_or_confirm_absent(&entry_path)?;
        } else {
            remove_cache_owned_file_or_confirm_absent(&entry_path)?;
        }
    }
    Ok(())
}

impl PersistentPromptCacheDiskStore {
    /// Serializes deletion against publication and keeps the active index valid.
    pub fn clear_prompt_cache(
        &self,
        model_id: Option<&str>,
    ) -> Result<PersistentPromptCacheClearOutcome, PersistentPromptCacheDiskStoreError> {
        let _write_operation_guard = self.lock_write_operations();
        let clear_target_directory =
            model_id.map(|model_id| self.global_prompt_cache_root_directory.join(model_id));
        let active_model_cache_was_cleared =
            clear_target_directory.as_ref().map_or(true, |target| {
                self.active_model_prompt_cache_directory.starts_with(target)
            });
        let clear_outcome = clear_persistent_prompt_cache_directory(
            &self.global_prompt_cache_root_directory,
            model_id,
        )?;
        if active_model_cache_was_cleared {
            *self.lock_tracked_files() = PersistentPromptCacheDiskStoreIndex::default();
            self.prepare_active_model_storage_directories()?;
        }
        self.refresh_global_prompt_cache_accounting()?;
        Ok(clear_outcome)
    }
}
