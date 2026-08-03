//! One global prompt-cache byte ceiling across every model and revision.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, remove_cache_owned_file_or_confirm_absent,
};

pub(super) fn prepare_prompt_cache_directory_tree(
    global_prompt_cache_root_directory: &Path,
    active_model_prompt_cache_directory: &Path,
    active_model_storage_directories: &[&Path],
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    reject_parent_directory_components(global_prompt_cache_root_directory)?;
    reject_parent_directory_components(active_model_prompt_cache_directory)?;
    fs::create_dir_all(global_prompt_cache_root_directory).map_err(|source| {
        PersistentPromptCacheDiskStoreError::CreatePromptCacheDirectory {
            persistent_prompt_cache_directory: global_prompt_cache_root_directory.to_path_buf(),
            source,
        }
    })?;
    verify_real_directory(global_prompt_cache_root_directory)?;
    if !active_model_prompt_cache_directory.starts_with(global_prompt_cache_root_directory) {
        return Err(
            PersistentPromptCacheDiskStoreError::ActivePromptCacheDirectoryOutsideGlobalRoot {
                active_model_prompt_cache_directory: active_model_prompt_cache_directory
                    .to_path_buf(),
                global_prompt_cache_root_directory: global_prompt_cache_root_directory
                    .to_path_buf(),
            },
        );
    }
    create_descendant_directories_without_symlinks(
        global_prompt_cache_root_directory,
        active_model_prompt_cache_directory,
    )?;
    for active_model_storage_directory in active_model_storage_directories {
        create_descendant_directories_without_symlinks(
            global_prompt_cache_root_directory,
            active_model_storage_directory,
        )?;
    }
    Ok(())
}

fn reject_parent_directory_components(
    persistent_prompt_cache_directory: &Path,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    if persistent_prompt_cache_directory
        .components()
        .any(|path_component| path_component == Component::ParentDir)
    {
        return Err(
            PersistentPromptCacheDiskStoreError::UnsafePromptCacheDirectory {
                persistent_prompt_cache_directory: persistent_prompt_cache_directory.to_path_buf(),
            },
        );
    }
    Ok(())
}

fn create_descendant_directories_without_symlinks(
    global_prompt_cache_root_directory: &Path,
    descendant_prompt_cache_directory: &Path,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    let descendant_relative_path = descendant_prompt_cache_directory
        .strip_prefix(global_prompt_cache_root_directory)
        .map_err(|_| {
            PersistentPromptCacheDiskStoreError::ActivePromptCacheDirectoryOutsideGlobalRoot {
                active_model_prompt_cache_directory: descendant_prompt_cache_directory
                    .to_path_buf(),
                global_prompt_cache_root_directory: global_prompt_cache_root_directory
                    .to_path_buf(),
            }
        })?;
    let mut current_prompt_cache_directory = global_prompt_cache_root_directory.to_path_buf();
    for descendant_path_component in descendant_relative_path.components() {
        let Component::Normal(directory_name) = descendant_path_component else {
            continue;
        };
        current_prompt_cache_directory.push(directory_name);
        match fs::create_dir(&current_prompt_cache_directory) {
            Ok(()) => {}
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(source) => {
                return Err(
                    PersistentPromptCacheDiskStoreError::CreatePromptCacheDirectory {
                        persistent_prompt_cache_directory: current_prompt_cache_directory,
                        source,
                    },
                );
            }
        }
        verify_real_directory(&current_prompt_cache_directory)?;
    }
    Ok(())
}

fn verify_real_directory(
    persistent_prompt_cache_directory: &Path,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    let directory_metadata =
        fs::symlink_metadata(persistent_prompt_cache_directory).map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: persistent_prompt_cache_directory.to_path_buf(),
                source,
            }
        })?;
    if directory_metadata.file_type().is_symlink() || !directory_metadata.is_dir() {
        return Err(
            PersistentPromptCacheDiskStoreError::UnsafePromptCacheDirectory {
                persistent_prompt_cache_directory: persistent_prompt_cache_directory.to_path_buf(),
            },
        );
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct GlobalPromptCacheFile {
    pub(super) file_path: PathBuf,
    pub(super) file_size_bytes: u64,
    pub(super) modified_at: SystemTime,
    pub(super) is_visual_embedding: bool,
    pub(super) is_stale_writer_temp_file: bool,
}

pub(super) struct GlobalPromptCacheQuotaScan {
    pub(super) files_oldest_written_first: Vec<GlobalPromptCacheFile>,
    pub(super) total_size_bytes: u64,
    pub(super) visual_embedding_total_size_bytes: u64,
}

pub(super) fn scan_global_prompt_cache_quota(
    global_prompt_cache_root_directory: &Path,
) -> Result<GlobalPromptCacheQuotaScan, PersistentPromptCacheDiskStoreError> {
    let mut global_prompt_cache_files =
        scan_global_prompt_cache_files(global_prompt_cache_root_directory)?;
    let global_prompt_cache_total_size_bytes = global_prompt_cache_files.iter().try_fold(
        0_u64,
        |accumulated_size_bytes, prompt_cache_file| {
            accumulated_size_bytes
                .checked_add(prompt_cache_file.file_size_bytes)
                .ok_or_else(
                    || PersistentPromptCacheDiskStoreError::GlobalPromptCacheSizeOverflow {
                        global_prompt_cache_root_directory: global_prompt_cache_root_directory
                            .to_path_buf(),
                    },
                )
        },
    )?;
    let global_visual_embedding_total_size_bytes = global_prompt_cache_files
        .iter()
        .filter(|prompt_cache_file| prompt_cache_file.is_visual_embedding)
        .try_fold(0_u64, |accumulated_size_bytes, prompt_cache_file| {
            accumulated_size_bytes
                .checked_add(prompt_cache_file.file_size_bytes)
                .ok_or_else(
                    || PersistentPromptCacheDiskStoreError::GlobalPromptCacheSizeOverflow {
                        global_prompt_cache_root_directory: global_prompt_cache_root_directory
                            .to_path_buf(),
                    },
                )
        })?;

    global_prompt_cache_files.sort_by(|left_file, right_file| {
        left_file
            .modified_at
            .cmp(&right_file.modified_at)
            .then_with(|| left_file.file_path.cmp(&right_file.file_path))
    });

    Ok(GlobalPromptCacheQuotaScan {
        files_oldest_written_first: global_prompt_cache_files,
        total_size_bytes: global_prompt_cache_total_size_bytes,
        visual_embedding_total_size_bytes: global_visual_embedding_total_size_bytes,
    })
}

fn scan_global_prompt_cache_files(
    global_prompt_cache_root_directory: &Path,
) -> Result<Vec<GlobalPromptCacheFile>, PersistentPromptCacheDiskStoreError> {
    let mut pending_directories = vec![global_prompt_cache_root_directory.to_path_buf()];
    let mut global_prompt_cache_files = Vec::new();
    while let Some(pending_directory) = pending_directories.pop() {
        let directory_entries = fs::read_dir(&pending_directory).map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                persistent_prompt_cache_directory: pending_directory.clone(),
                source,
            }
        })?;
        for directory_entry_result in directory_entries {
            let directory_entry = directory_entry_result.map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                    persistent_prompt_cache_directory: pending_directory.clone(),
                    source,
                }
            })?;
            let file_path = directory_entry.path();
            let file_type = directory_entry.file_type().map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: file_path.clone(),
                    source,
                }
            })?;
            if file_type.is_dir() {
                pending_directories.push(file_path);
                continue;
            }
            let file_metadata = fs::symlink_metadata(&file_path).map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: file_path.clone(),
                    source,
                }
            })?;
            let modified_at = file_metadata.modified().map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: file_path.clone(),
                    source,
                }
            })?;
            let is_visual_embedding = file_path.parent().is_some_and(|parent_directory| {
                parent_directory
                    .file_name()
                    .is_some_and(|directory_name| directory_name == "visual_embeddings")
            });
            let is_stale_writer_temp_file = file_path
                .extension()
                .is_some_and(|extension| extension == "tmp");
            global_prompt_cache_files.push(GlobalPromptCacheFile {
                file_path,
                file_size_bytes: file_metadata.len(),
                modified_at,
                is_visual_embedding,
                is_stale_writer_temp_file,
            });
        }
    }
    Ok(global_prompt_cache_files)
}

impl PersistentPromptCacheDiskStore {
    pub(crate) fn untrack_file_and_subtract_global_accounting(
        &self,
        persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
        persistent_prompt_cache_file_hash: [u8; 32],
    ) {
        let removed_tracked_file = self.lock_tracked_files().remove_file(
            persistent_prompt_cache_file_kind,
            &persistent_prompt_cache_file_hash,
        );
        let Some(removed_tracked_file) = removed_tracked_file else {
            return;
        };
        subtract_atomic_size_bytes(
            &self.global_prompt_cache_total_size_bytes,
            removed_tracked_file.file_size_bytes,
        );
        if matches!(
            persistent_prompt_cache_file_kind,
            PersistentPromptCacheFileKind::VisualEmbedding
        ) {
            subtract_atomic_size_bytes(
                &self.global_visual_embedding_total_size_bytes,
                removed_tracked_file.file_size_bytes,
            );
        }
    }

    pub(super) fn refresh_global_prompt_cache_accounting(
        &self,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        let global_quota_scan =
            scan_global_prompt_cache_quota(&self.global_prompt_cache_root_directory)?;
        self.global_prompt_cache_total_size_bytes
            .store(global_quota_scan.total_size_bytes, Ordering::Relaxed);
        self.global_visual_embedding_total_size_bytes.store(
            global_quota_scan.visual_embedding_total_size_bytes,
            Ordering::Relaxed,
        );
        Ok(())
    }

    pub(crate) fn rollback_newly_saved_files_after_error(
        &self,
        newly_saved_file_paths: &[PathBuf],
        original_error: PersistentPromptCacheDiskStoreError,
    ) -> PersistentPromptCacheDiskStoreError {
        let mut first_rollback_error = None;
        for newly_saved_file_path in newly_saved_file_paths {
            match remove_cache_owned_file_or_confirm_absent(newly_saved_file_path) {
                Ok(()) => self
                    .lock_tracked_files()
                    .remove_files_by_path(std::slice::from_ref(newly_saved_file_path)),
                Err(rollback_error) => {
                    if first_rollback_error.is_none() {
                        first_rollback_error = Some(rollback_error);
                    }
                }
            }
        }
        let accounting_refresh_error = self.refresh_global_prompt_cache_accounting().err();
        first_rollback_error
            .or(accounting_refresh_error)
            .unwrap_or(original_error)
    }

    pub(crate) fn enforce_global_prompt_cache_quota(
        &self,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        let global_quota_scan =
            scan_global_prompt_cache_quota(&self.global_prompt_cache_root_directory)?;
        let mut global_prompt_cache_total_size_bytes = global_quota_scan.total_size_bytes;
        let mut global_visual_embedding_total_size_bytes =
            global_quota_scan.visual_embedding_total_size_bytes;
        self.global_prompt_cache_total_size_bytes
            .store(global_prompt_cache_total_size_bytes, Ordering::Relaxed);
        self.global_visual_embedding_total_size_bytes
            .store(global_visual_embedding_total_size_bytes, Ordering::Relaxed);
        for global_prompt_cache_file in global_quota_scan.files_oldest_written_first {
            if !global_prompt_cache_file.is_stale_writer_temp_file
                && global_prompt_cache_total_size_bytes
                    <= self.global_prompt_cache_maximum_size_bytes
            {
                continue;
            }
            remove_cache_owned_file_or_confirm_absent(&global_prompt_cache_file.file_path)?;
            self.lock_tracked_files()
                .remove_files_by_path(std::slice::from_ref(&global_prompt_cache_file.file_path));
            global_prompt_cache_total_size_bytes = global_prompt_cache_total_size_bytes
                .saturating_sub(global_prompt_cache_file.file_size_bytes);
            if global_prompt_cache_file.is_visual_embedding {
                global_visual_embedding_total_size_bytes = global_visual_embedding_total_size_bytes
                    .saturating_sub(global_prompt_cache_file.file_size_bytes);
            }
            self.global_prompt_cache_total_size_bytes
                .store(global_prompt_cache_total_size_bytes, Ordering::Relaxed);
            self.global_visual_embedding_total_size_bytes
                .store(global_visual_embedding_total_size_bytes, Ordering::Relaxed);
        }
        if global_prompt_cache_total_size_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(
                PersistentPromptCacheDiskStoreError::GlobalPromptCacheQuotaNotSatisfied {
                    maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                    remaining_size_bytes: global_prompt_cache_total_size_bytes,
                },
            );
        }
        Ok(())
    }
}

fn subtract_atomic_size_bytes(atomic_size_bytes: &AtomicU64, removed_size_bytes: u64) {
    let _previous_size_bytes = atomic_size_bytes
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current_size_bytes| {
            Some(current_size_bytes.saturating_sub(removed_size_bytes))
        })
        .unwrap_or_else(|unchanged_size_bytes| unchanged_size_bytes);
}
