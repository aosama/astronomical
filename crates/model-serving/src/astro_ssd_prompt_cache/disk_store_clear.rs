//! Safe deletion of global or model-scoped persistent astro-ssd-prompt-cache trees.
//!
//! The worker is the sole owner of this operation. The daemon sends an IPC
//! command and never reads or mutates astro SSD prompt cache files directly.

use std::fs;
use std::path::{Component, Path, PathBuf};

use super::disk_store::AstroSsdPromptCacheDiskStore;
use super::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use super::disk_store_file::{
    remove_astro_ssd_prompt_cache_owned_directory_or_confirm_absent,
    remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent,
};
use super::disk_store_global_quota;
use super::disk_store_index::AstroSsdPromptCacheDiskStoreIndex;

const BLOCKS_DIRECTORY_NAME: &str = "blocks";

/// Measured SSD space and astro-ssd-prompt-cache blocks removed by one clear operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AstroSsdPromptCacheClearOutcome {
    pub model_id: Option<String>,
    pub blocks_removed: u64,
    pub bytes_freed: u64,
}

/// Deletes either every astro SSD prompt cache namespace or all revisions of one model.
///
/// Model identities may contain multiple normal path components, such as
/// `organization/model`, because that is the astro SSD prompt cache layout used by model IDs.
/// Absolute paths, `.` and `..` are rejected before any filesystem access.
pub fn clear_astro_ssd_prompt_cache_directory(
    global_astro_ssd_prompt_cache_root_directory: &Path,
    model_id: Option<&str>,
) -> Result<AstroSsdPromptCacheClearOutcome, AstroSsdPromptCacheDiskStoreError> {
    let clear_target_directory =
        clear_target_directory(global_astro_ssd_prompt_cache_root_directory, model_id)?;
    let Some(clear_target_directory) = clear_target_directory else {
        return Ok(AstroSsdPromptCacheClearOutcome {
            model_id: model_id.map(str::to_owned),
            blocks_removed: 0,
            bytes_freed: 0,
        });
    };
    let (blocks_removed, bytes_freed) = measure_clear_target(&clear_target_directory)?;
    if model_id.is_some() {
        remove_astro_ssd_prompt_cache_owned_directory_or_confirm_absent(&clear_target_directory)?;
    } else {
        remove_global_root_contents(&clear_target_directory)?;
    }
    Ok(AstroSsdPromptCacheClearOutcome {
        model_id: model_id.map(str::to_owned),
        blocks_removed,
        bytes_freed,
    })
}

fn clear_target_directory(
    global_astro_ssd_prompt_cache_root_directory: &Path,
    model_id: Option<&str>,
) -> Result<Option<PathBuf>, AstroSsdPromptCacheDiskStoreError> {
    disk_store_global_quota::reject_parent_directory_components(
        global_astro_ssd_prompt_cache_root_directory,
    )?;
    if !verify_existing_real_directory(global_astro_ssd_prompt_cache_root_directory)? {
        return Ok(None);
    }
    let Some(model_id) = model_id else {
        return Ok(Some(
            global_astro_ssd_prompt_cache_root_directory.to_path_buf(),
        ));
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
            AstroSsdPromptCacheDiskStoreError::UnsafeAstroSsdPromptCacheDirectory {
                astro_ssd_prompt_cache_directory: model_id_path.to_path_buf(),
            },
        );
    }
    let model_astro_ssd_prompt_cache_directory =
        global_astro_ssd_prompt_cache_root_directory.join(model_id_path);
    let model_astro_ssd_prompt_cache_directory_exists =
        verify_descendant_components(global_astro_ssd_prompt_cache_root_directory, model_id_path)?;
    Ok(model_astro_ssd_prompt_cache_directory_exists
        .then_some(model_astro_ssd_prompt_cache_directory))
}

fn verify_descendant_components(
    global_astro_ssd_prompt_cache_root_directory: &Path,
    descendant_relative_path: &Path,
) -> Result<bool, AstroSsdPromptCacheDiskStoreError> {
    let mut current_directory = global_astro_ssd_prompt_cache_root_directory.to_path_buf();
    for path_component in descendant_relative_path.components() {
        let Component::Normal(directory_name) = path_component else {
            return Err(
                AstroSsdPromptCacheDiskStoreError::UnsafeAstroSsdPromptCacheDirectory {
                    astro_ssd_prompt_cache_directory: descendant_relative_path.to_path_buf(),
                },
            );
        };
        current_directory.push(directory_name);
        match fs::symlink_metadata(&current_directory) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(_) => {
                return Err(
                    AstroSsdPromptCacheDiskStoreError::UnsafeAstroSsdPromptCacheDirectory {
                        astro_ssd_prompt_cache_directory: current_directory,
                    },
                );
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(source) => {
                return Err(AstroSsdPromptCacheDiskStoreError::ReadBlockMetadata {
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
) -> Result<bool, AstroSsdPromptCacheDiskStoreError> {
    match fs::symlink_metadata(directory_path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(true),
        Ok(_) => Err(
            AstroSsdPromptCacheDiskStoreError::UnsafeAstroSsdPromptCacheDirectory {
                astro_ssd_prompt_cache_directory: directory_path.to_path_buf(),
            },
        ),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(AstroSsdPromptCacheDiskStoreError::ReadBlockMetadata {
            block_file_path: directory_path.to_path_buf(),
            source,
        }),
    }
}

fn measure_clear_target(
    clear_target_directory: &Path,
) -> Result<(u64, u64), AstroSsdPromptCacheDiskStoreError> {
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
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    let directory_entries = fs::read_dir(directory_path).map_err(|source| {
        AstroSsdPromptCacheDiskStoreError::ReadAstroSsdPromptCacheDirectory {
            astro_ssd_prompt_cache_directory: directory_path.to_path_buf(),
            source,
        }
    })?;
    for directory_entry in directory_entries {
        let directory_entry = directory_entry.map_err(|source| {
            AstroSsdPromptCacheDiskStoreError::ReadAstroSsdPromptCacheDirectory {
                astro_ssd_prompt_cache_directory: directory_path.to_path_buf(),
                source,
            }
        })?;
        let entry_path = directory_entry.path();
        let entry_metadata = fs::symlink_metadata(&entry_path).map_err(|source| {
            AstroSsdPromptCacheDiskStoreError::ReadBlockMetadata {
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
    global_astro_ssd_prompt_cache_root_directory: &Path,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    let directory_entries =
        fs::read_dir(global_astro_ssd_prompt_cache_root_directory).map_err(|source| {
            AstroSsdPromptCacheDiskStoreError::ReadAstroSsdPromptCacheDirectory {
                astro_ssd_prompt_cache_directory: global_astro_ssd_prompt_cache_root_directory
                    .to_path_buf(),
                source,
            }
        })?;
    for directory_entry in directory_entries {
        let directory_entry = directory_entry.map_err(|source| {
            AstroSsdPromptCacheDiskStoreError::ReadAstroSsdPromptCacheDirectory {
                astro_ssd_prompt_cache_directory: global_astro_ssd_prompt_cache_root_directory
                    .to_path_buf(),
                source,
            }
        })?;
        let entry_path = directory_entry.path();
        let entry_metadata = fs::symlink_metadata(&entry_path).map_err(|source| {
            AstroSsdPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: entry_path.clone(),
                source,
            }
        })?;
        if entry_metadata.is_dir() && !entry_metadata.file_type().is_symlink() {
            remove_astro_ssd_prompt_cache_owned_directory_or_confirm_absent(&entry_path)?;
        } else {
            remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent(&entry_path)?;
        }
    }
    Ok(())
}

impl AstroSsdPromptCacheDiskStore {
    /// Serializes deletion against publication and keeps the active index valid.
    pub fn clear_astro_ssd_prompt_cache(
        &self,
        model_id: Option<&str>,
    ) -> Result<AstroSsdPromptCacheClearOutcome, AstroSsdPromptCacheDiskStoreError> {
        let _write_operation_guard = self.lock_write_operations();
        let clear_target_directory = model_id.map(|model_id| {
            self.global_astro_ssd_prompt_cache_root_directory
                .join(model_id)
        });
        let active_model_astro_ssd_prompt_cache_was_cleared =
            clear_target_directory.as_ref().is_none_or(|target| {
                self.active_model_astro_ssd_prompt_cache_directory
                    .starts_with(target)
            });
        let clear_outcome = clear_astro_ssd_prompt_cache_directory(
            &self.global_astro_ssd_prompt_cache_root_directory,
            model_id,
        )?;
        if active_model_astro_ssd_prompt_cache_was_cleared {
            *self.lock_tracked_files() = AstroSsdPromptCacheDiskStoreIndex::default();
            self.prepare_active_model_storage_directories()?;
        }
        self.refresh_global_astro_ssd_prompt_cache_accounting()?;
        Ok(clear_outcome)
    }
}
