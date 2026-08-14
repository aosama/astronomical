//! One global prompt-cache byte ceiling across every model and revision.
//!
//! The quota owner operates on scan-produced units. Abandoned transactions are
//! always removed first; committed blocks are removed only as complete subtrees;
//! and directories belonging to the active publication ancestry are protected.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, remove_cache_owned_directory_or_confirm_absent,
    remove_cache_owned_file_or_confirm_absent,
};
use super::disk_store_global_quota_candidate::{
    GlobalPromptCacheCleanupClassification, GlobalPromptCacheEvictionCandidate,
};
use super::disk_store_global_quota_scan::scan_global_prompt_cache_quota;
use super::startup_cleanup_evidence::PersistentPromptCacheStartupCleanupEvidence;

pub(super) fn prepare_prompt_cache_directory_tree(
    global_prompt_cache_root_directory: &Path,
    active_model_prompt_cache_directory: &Path,
    active_model_storage_directories: &[&Path],
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    // Deletion helpers recurse, so directory creation is also the trust boundary:
    // reject lexical escapes and verify every created component is a real
    // directory rather than following a symlink into user-owned data.
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

pub(super) fn reject_parent_directory_components(
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

impl PersistentPromptCacheDiskStore {
    pub(super) fn remove_unconditionally_reclaimable_startup_artifacts(
        &self,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        // Interrupted transactions and obsolete formats have no readable
        // current-format value, so cleanup is not pressure-dependent.
        let global_quota_scan =
            scan_global_prompt_cache_quota(&self.global_prompt_cache_root_directory, None)?;
        let mut startup_cleanup_evidence = PersistentPromptCacheStartupCleanupEvidence::default();
        for stale_transaction_artifact in global_quota_scan
            .eviction_candidates_oldest_written_first
            .into_iter()
            .filter(GlobalPromptCacheEvictionCandidate::is_unconditionally_removable)
        {
            let Some(cleanup_classification) =
                stale_transaction_artifact.unconditional_cleanup_classification()
            else {
                continue;
            };
            remove_global_prompt_cache_eviction_candidate(&stale_transaction_artifact)?;
            record_removed_startup_candidate(
                &mut startup_cleanup_evidence,
                cleanup_classification,
                &stale_transaction_artifact,
            );
            self.lock_tracked_files()
                .remove_files_by_path(stale_transaction_artifact.tracked_file_paths());
            self.lock_tracked_files().remove_blocks_by_directory_paths(
                stale_transaction_artifact.block_directory_paths(),
            );
        }
        self.record_startup_cleanup_evidence(startup_cleanup_evidence);
        self.refresh_global_prompt_cache_accounting()
    }

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
        // Atomic counters are telemetry snapshots, not the source of truth.
        // Re-scan after recovery or rollback because another cleanup path may
        // have changed disk state without incrementally updating every counter.
        let global_quota_scan =
            scan_global_prompt_cache_quota(&self.global_prompt_cache_root_directory, None)?;
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
        self.enforce_global_prompt_cache_quota_for_commit(0, 0, &[], None)
    }

    pub(super) fn enforce_startup_global_prompt_cache_quota(
        &self,
        protected_block_directory_paths: &[PathBuf],
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        self.enforce_global_prompt_cache_quota_for_commit_internal(
            0,
            0,
            protected_block_directory_paths,
            None,
            true,
        )
    }

    pub(crate) fn enforce_global_prompt_cache_quota_for_commit(
        &self,
        additional_committed_size_bytes: u64,
        post_commit_reclaimable_size_bytes: u64,
        protected_block_directory_paths: &[PathBuf],
        excluded_directory: Option<&Path>,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        self.enforce_global_prompt_cache_quota_for_commit_internal(
            additional_committed_size_bytes,
            post_commit_reclaimable_size_bytes,
            protected_block_directory_paths,
            excluded_directory,
            false,
        )
    }

    fn enforce_global_prompt_cache_quota_for_commit_internal(
        &self,
        additional_committed_size_bytes: u64,
        post_commit_reclaimable_size_bytes: u64,
        protected_block_directory_paths: &[PathBuf],
        excluded_directory: Option<&Path>,
        should_record_startup_cleanup: bool,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        // Exclude the current staging directory because `additional_committed_size_bytes`
        // already accounts for it. Counting both would charge the transaction twice.
        let global_quota_scan = scan_global_prompt_cache_quota(
            &self.global_prompt_cache_root_directory,
            excluded_directory,
        )?;
        let mut global_prompt_cache_total_size_bytes = global_quota_scan.total_size_bytes;
        let mut global_visual_embedding_total_size_bytes =
            global_quota_scan.visual_embedding_total_size_bytes;
        self.global_prompt_cache_total_size_bytes
            .store(global_prompt_cache_total_size_bytes, Ordering::Relaxed);
        self.global_visual_embedding_total_size_bytes
            .store(global_visual_embedding_total_size_bytes, Ordering::Relaxed);
        let mut removed_eviction_paths = HashSet::<PathBuf>::new();
        let mut startup_cleanup_evidence = PersistentPromptCacheStartupCleanupEvidence::default();
        for global_prompt_cache_eviction_candidate in
            global_quota_scan.eviction_candidates_oldest_written_first
        {
            if eviction_candidate_was_already_removed(
                &global_prompt_cache_eviction_candidate,
                &removed_eviction_paths,
            ) {
                continue;
            }
            // Interrupted transactions and obsolete formats are removed even
            // when the committed projection already fits.
            if !global_prompt_cache_eviction_candidate.is_unconditionally_removable()
                && global_prompt_cache_eviction_candidate
                    .contains_protected_block_directory(protected_block_directory_paths)
            {
                continue;
            }
            // Stop deleting committed value once the post-commit projection
            // fits. `post_commit_reclaimable_size_bytes` is subtracted only in
            // arithmetic; its actual file remains until commit becomes durable.
            if !global_prompt_cache_eviction_candidate.is_unconditionally_removable()
                && committed_size_after_addition(
                    global_prompt_cache_total_size_bytes,
                    additional_committed_size_bytes,
                    post_commit_reclaimable_size_bytes,
                    &self.global_prompt_cache_root_directory,
                )? <= self.global_prompt_cache_maximum_size_bytes
            {
                continue;
            }
            remove_global_prompt_cache_eviction_candidate(&global_prompt_cache_eviction_candidate)?;
            if should_record_startup_cleanup {
                record_removed_startup_candidate(
                    &mut startup_cleanup_evidence,
                    global_prompt_cache_eviction_candidate
                        .unconditional_cleanup_classification()
                        .unwrap_or(GlobalPromptCacheCleanupClassification::QuotaEviction),
                    &global_prompt_cache_eviction_candidate,
                );
            }
            self.lock_tracked_files()
                .remove_files_by_path(global_prompt_cache_eviction_candidate.tracked_file_paths());
            self.lock_tracked_files().remove_blocks_by_directory_paths(
                global_prompt_cache_eviction_candidate.block_directory_paths(),
            );
            global_prompt_cache_total_size_bytes = global_prompt_cache_total_size_bytes
                .saturating_sub(global_prompt_cache_eviction_candidate.file_size_bytes());
            global_visual_embedding_total_size_bytes = global_visual_embedding_total_size_bytes
                .saturating_sub(
                    global_prompt_cache_eviction_candidate.visual_embedding_size_bytes(),
                );
            removed_eviction_paths.insert(
                global_prompt_cache_eviction_candidate
                    .tie_breaker_path()
                    .to_path_buf(),
            );
            for removed_block_directory_path in
                global_prompt_cache_eviction_candidate.block_directory_paths()
            {
                removed_eviction_paths.insert(removed_block_directory_path.clone());
            }
            self.global_prompt_cache_total_size_bytes
                .store(global_prompt_cache_total_size_bytes, Ordering::Relaxed);
            self.global_visual_embedding_total_size_bytes
                .store(global_visual_embedding_total_size_bytes, Ordering::Relaxed);
        }
        let final_committed_size_bytes = committed_size_after_addition(
            global_prompt_cache_total_size_bytes,
            additional_committed_size_bytes,
            post_commit_reclaimable_size_bytes,
            &self.global_prompt_cache_root_directory,
        )?;
        if final_committed_size_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(
                PersistentPromptCacheDiskStoreError::GlobalPromptCacheQuotaNotSatisfied {
                    maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                    remaining_size_bytes: final_committed_size_bytes,
                },
            );
        }
        if should_record_startup_cleanup {
            self.record_startup_cleanup_evidence(startup_cleanup_evidence);
        }
        Ok(())
    }
}

fn record_removed_startup_candidate(
    startup_cleanup_evidence: &mut PersistentPromptCacheStartupCleanupEvidence,
    cleanup_classification: GlobalPromptCacheCleanupClassification,
    removed_candidate: &GlobalPromptCacheEvictionCandidate,
) {
    let cleanup_category = match cleanup_classification {
        GlobalPromptCacheCleanupClassification::InterruptedTransactionRecovery => {
            &mut startup_cleanup_evidence.interrupted_transaction_recovery
        }
        GlobalPromptCacheCleanupClassification::ObsoleteFormat => {
            &mut startup_cleanup_evidence.obsolete_format
        }
        GlobalPromptCacheCleanupClassification::QuotaEviction => {
            &mut startup_cleanup_evidence.quota_eviction
        }
    };
    for _removed_artifact_index in 0..removed_candidate.removed_artifact_count() {
        cleanup_category.record_artifact(0);
    }
    cleanup_category.record_blocks(removed_candidate.removed_block_count(), 0);
    cleanup_category.byte_count = cleanup_category
        .byte_count
        .saturating_add(removed_candidate.file_size_bytes());
}

fn eviction_candidate_was_already_removed(
    global_prompt_cache_eviction_candidate: &GlobalPromptCacheEvictionCandidate,
    removed_eviction_paths: &HashSet<PathBuf>,
) -> bool {
    // Subtree candidates overlap by construction. A removed ancestor can cover
    // a later candidate even when their root paths differ, hence the member scan.
    removed_eviction_paths.contains(global_prompt_cache_eviction_candidate.tie_breaker_path())
        || global_prompt_cache_eviction_candidate
            .block_directory_paths()
            .iter()
            .any(|block_directory_path| removed_eviction_paths.contains(block_directory_path))
}

fn remove_global_prompt_cache_eviction_candidate(
    global_prompt_cache_eviction_candidate: &GlobalPromptCacheEvictionCandidate,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    match global_prompt_cache_eviction_candidate {
        GlobalPromptCacheEvictionCandidate::StandaloneFile(global_prompt_cache_file) => {
            remove_cache_owned_file_or_confirm_absent(&global_prompt_cache_file.file_path)
        }
        GlobalPromptCacheEvictionCandidate::StaleDirectory(global_prompt_cache_stale_directory) => {
            remove_cache_owned_directory_or_confirm_absent(
                &global_prompt_cache_stale_directory.directory_path,
            )
        }
        GlobalPromptCacheEvictionCandidate::BlockSubtree(global_prompt_cache_block_subtree) => {
            for block_directory_path in &global_prompt_cache_block_subtree.block_directory_paths {
                remove_cache_owned_directory_or_confirm_absent(block_directory_path)?;
            }
            Ok(())
        }
    }
}

fn committed_size_after_addition(
    current_committed_size_bytes: u64,
    additional_committed_size_bytes: u64,
    post_commit_reclaimable_size_bytes: u64,
    global_prompt_cache_root_directory: &Path,
) -> Result<u64, PersistentPromptCacheDiskStoreError> {
    current_committed_size_bytes
        .checked_add(additional_committed_size_bytes)
        .map(|size_before_post_commit_reclamation| {
            size_before_post_commit_reclamation.saturating_sub(post_commit_reclaimable_size_bytes)
        })
        .ok_or_else(
            || PersistentPromptCacheDiskStoreError::GlobalPromptCacheSizeOverflow {
                global_prompt_cache_root_directory: global_prompt_cache_root_directory
                    .to_path_buf(),
            },
        )
}

fn subtract_atomic_size_bytes(atomic_size_bytes: &AtomicU64, removed_size_bytes: u64) {
    let _previous_size_bytes = atomic_size_bytes
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current_size_bytes| {
            Some(current_size_bytes.saturating_sub(removed_size_bytes))
        })
        .unwrap_or_else(|unchanged_size_bytes| unchanged_size_bytes);
}
