//! Crash-safe publication transactions for format-11 block directories.
//!
//! A complete block follows this order:
//! 1. write every state file and the manifest into a unique staging directory;
//! 2. synchronize that directory so its contents survive a crash;
//! 3. reserve global quota while protecting the chain being extended;
//! 4. atomically rename the staging directory to its content hash;
//! 5. synchronize `blocks/`, then expose the block through the in-memory index;
//! 6. reclaim a redundant parent boundary only after the child is durable.
//!
//! Reordering those steps can expose half-written state, break ancestry, or
//! delete the only restorable boundary during an interrupted publication.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::block_key::PersistentPromptCacheBlockKey;
use super::block_manifest::{
    BLOCK_MANIFEST_FILE_NAME, BOUNDARY_STATE_FILE_NAME, PersistentPromptCacheBlockManifest,
    SEQUENCE_STATE_FILE_NAME,
};
use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, StagedPersistentPromptCacheStateFile, read_file_size_bytes,
    remove_cache_owned_directory_or_confirm_absent, remove_cache_owned_file_or_confirm_absent,
    save_direct_safetensors_file_with_name, synchronize_directory,
};
use super::disk_store_index::{
    TrackedPersistentPromptCacheBlock, TrackedPersistentPromptCacheFile,
};
use super::retention_policy::persistent_prompt_cache_boundary_is_common_prefix_checkpoint;

impl PersistentPromptCacheDiskStore {
    pub(super) fn publish_new_block_transaction(
        &self,
        runtime: &MlxRuntime,
        block_key: &PersistentPromptCacheBlockKey,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
        sequence_state_tensors: &HashMap<String, MlxArray>,
        boundary_state_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        // The final name is deterministic, but staging must be unique so a
        // crashed writer cannot be mistaken for a committed block on restart.
        let block_directory_name = super::disk_store_file::hex_encode(block_key.block_hash());
        let final_block_directory = self.blocks_directory.join(&block_directory_name);
        let staging_block_directory =
            unique_staging_block_directory(&self.blocks_directory, &block_directory_name);
        create_staging_directory(&staging_block_directory)?;

        let staged_files = match self.stage_complete_block(
            runtime,
            block_key,
            parent_block_key,
            sequence_state_tensors,
            boundary_state_tensors,
            performance_attribution,
            &staging_block_directory,
        ) {
            Ok(staged_files) => staged_files,
            Err(staging_error) => {
                remove_cache_owned_directory_or_confirm_absent(&staging_block_directory)?;
                return Err(staging_error);
            }
        };
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCachePublicationSynchronizationWait,
                |_performance_attribution| synchronize_directory(&staging_block_directory),
            )
            .map_err(|sync_error| {
                cleanup_staging_after_error(&staging_block_directory, sync_error)
            })?;

        // Quota projection may subtract this boundary because it becomes
        // redundant after commit. The file is not physically removed until the
        // child rename and parent-directory synchronization have succeeded.
        let parent_boundary_reclaim = self
            .parent_boundary_reclaim_after_commit(parent_block_key, staged_files.total_size_bytes);
        let mut protected_block_directory_paths =
            self.protected_ancestry_for_commit(parent_block_key);
        protected_block_directory_paths.push(final_block_directory.clone());
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheGlobalQuotaEviction,
                |_performance_attribution| {
                    self.enforce_global_prompt_cache_quota_for_commit(
                        staged_files.total_size_bytes,
                        parent_boundary_reclaim
                            .as_ref()
                            .map_or(0, |parent_boundary_reclaim| {
                                parent_boundary_reclaim.file_size_bytes
                            }),
                        &protected_block_directory_paths,
                        Some(&staging_block_directory),
                    )
                },
            )
            .map_err(|quota_error| {
                cleanup_staging_after_error(&staging_block_directory, quota_error)
            })?;
        // A writer holding the process-local publication lock should not race
        // itself. Presence here therefore means disk topology changed outside
        // this transaction; do not replace content we did not validate.
        if final_block_directory.exists() {
            remove_cache_owned_directory_or_confirm_absent(&staging_block_directory)?;
            return Err(
                PersistentPromptCacheDiskStoreError::ExistingBlockTopologyMismatch {
                    block_hash: block_key.block_hash(),
                },
            );
        }
        performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCacheAtomicCommit,
            |_performance_attribution| {
                std::fs::rename(&staging_block_directory, &final_block_directory).map_err(
                    |source| {
                        let _cleanup_result = remove_cache_owned_directory_or_confirm_absent(
                            &staging_block_directory,
                        );
                        PersistentPromptCacheDiskStoreError::RenameTempFile {
                            temp_file_path: staging_block_directory.clone(),
                            block_file_path: final_block_directory.clone(),
                            source,
                        }
                    },
                )
            },
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCachePublicationSynchronizationWait,
            |_performance_attribution| synchronize_directory(&self.blocks_directory),
        )?;
        // Index publication occurs after durable filesystem publication. A
        // lookup can therefore never observe an indexed staging transaction.
        self.track_committed_block(
            block_key,
            parent_block_key,
            &final_block_directory,
            staged_files,
        );
        if let Some(parent_boundary_reclaim) = parent_boundary_reclaim {
            performance_attribution.measure_operation(
                PerformanceOperation::PersistentPromptCacheRetentionCleanup,
                |_performance_attribution| {
                    remove_cache_owned_file_or_confirm_absent(&parent_boundary_reclaim.file_path)?;
                    synchronize_directory(&parent_boundary_reclaim.block_directory_path)
                },
            )?;
            self.lock_tracked_files().remove_file(
                PersistentPromptCacheFileKind::BoundaryStateSnapshot,
                &parent_boundary_reclaim.block_hash,
            );
        }
        self.refresh_global_prompt_cache_accounting()?;
        Ok(())
    }

    pub(super) fn publish_missing_boundary_state_transaction(
        &self,
        runtime: &MlxRuntime,
        block_key: &PersistentPromptCacheBlockKey,
        boundary_state_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        // Startup compaction may legally leave sequence state on a non-checkpoint
        // parent without its boundary snapshot. If that exact block is reached
        // again as a leaf, publication restores only the missing boundary; it
        // never rewrites the already durable sequence state or manifest.
        let block_hash = block_key.block_hash();
        let existing_block = self
            .lock_tracked_files()
            .block(&block_hash)
            .cloned()
            .ok_or(
                PersistentPromptCacheDiskStoreError::ExistingBlockTopologyMismatch { block_hash },
            )?;
        let staging_directory = unique_staging_block_directory(
            &self.blocks_directory,
            &format!(
                "{}-boundary",
                super::disk_store_file::hex_encode(block_hash)
            ),
        );
        create_staging_directory(&staging_directory)?;
        let staged_boundary_state = save_direct_safetensors_file_with_name(
            runtime,
            &staging_directory,
            BOUNDARY_STATE_FILE_NAME,
            boundary_state_tensors,
            &self.model_contract,
            performance_attribution,
        )
        .map_err(|staging_error| cleanup_staging_after_error(&staging_directory, staging_error))?;
        validate_staged_file_size(
            &staged_boundary_state,
            self.model_contract.boundary_state_file_bytes(),
        )
        .map_err(|size_error| cleanup_staging_after_error(&staging_directory, size_error))?;
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCachePublicationSynchronizationWait,
                |_performance_attribution| synchronize_directory(&staging_directory),
            )
            .map_err(|sync_error| cleanup_staging_after_error(&staging_directory, sync_error))?;
        let protected_block_directory_paths = self
            .lock_tracked_files()
            .protected_ancestry_directory_paths(block_hash);
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheGlobalQuotaEviction,
                |_performance_attribution| {
                    self.enforce_global_prompt_cache_quota_for_commit(
                        staged_boundary_state.file_size_bytes,
                        0,
                        &protected_block_directory_paths,
                        Some(&staging_directory),
                    )
                },
            )
            .map_err(|quota_error| cleanup_staging_after_error(&staging_directory, quota_error))?;
        // Renaming one file within an already committed block is atomic. The
        // subsequent block-directory sync makes the new directory entry durable.
        let final_boundary_file_path = existing_block
            .block_directory_path
            .join(BOUNDARY_STATE_FILE_NAME);
        performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCacheAtomicCommit,
            |_performance_attribution| {
                std::fs::rename(&staged_boundary_state.file_path, &final_boundary_file_path)
                    .map_err(|source| {
                        cleanup_staging_after_error(
                            &staging_directory,
                            PersistentPromptCacheDiskStoreError::RenameTempFile {
                                temp_file_path: staged_boundary_state.file_path.clone(),
                                block_file_path: final_boundary_file_path.clone(),
                                source,
                            },
                        )
                    })
            },
        )?;
        remove_cache_owned_directory_or_confirm_absent(&staging_directory)?;
        performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCachePublicationSynchronizationWait,
            |_performance_attribution| {
                synchronize_directory(&existing_block.block_directory_path)?;
                synchronize_directory(&self.blocks_directory)
            },
        )?;
        self.lock_tracked_files().insert_file(
            PersistentPromptCacheFileKind::BoundaryStateSnapshot,
            block_hash,
            TrackedPersistentPromptCacheFile {
                file_path: final_boundary_file_path,
                file_size_bytes: staged_boundary_state.file_size_bytes,
            },
        );
        self.refresh_global_prompt_cache_accounting()?;
        Ok(())
    }

    fn stage_complete_block(
        &self,
        runtime: &MlxRuntime,
        block_key: &PersistentPromptCacheBlockKey,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
        sequence_state_tensors: &HashMap<String, MlxArray>,
        boundary_state_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
        staging_block_directory: &Path,
    ) -> Result<StagedBlockFiles, PersistentPromptCacheDiskStoreError> {
        // State-kind presence comes from the immutable model contract. Empty
        // placeholder files are forbidden because they would make a block look
        // complete while carrying no restorable model state.
        let sequence_state_file = if self.model_contract.has_sequence_state() {
            let staged_sequence_state = save_direct_safetensors_file_with_name(
                runtime,
                staging_block_directory,
                SEQUENCE_STATE_FILE_NAME,
                sequence_state_tensors,
                &self.model_contract,
                performance_attribution,
            )?;
            validate_staged_file_size(
                &staged_sequence_state,
                self.model_contract.sequence_state_file_bytes(),
            )?;
            Some(staged_sequence_state)
        } else {
            None
        };
        let boundary_state_file = if self.model_contract.has_boundary_state() {
            let staged_boundary_state = save_direct_safetensors_file_with_name(
                runtime,
                staging_block_directory,
                BOUNDARY_STATE_FILE_NAME,
                boundary_state_tensors,
                &self.model_contract,
                performance_attribution,
            )?;
            validate_staged_file_size(
                &staged_boundary_state,
                self.model_contract.boundary_state_file_bytes(),
            )?;
            Some(staged_boundary_state)
        } else {
            None
        };
        PersistentPromptCacheBlockManifest::new(block_key, parent_block_key, &self.model_contract)
            .write_to_staging_directory(staging_block_directory)?;
        let manifest_file_size_bytes =
            read_file_size_bytes(&staging_block_directory.join(BLOCK_MANIFEST_FILE_NAME))?;
        if manifest_file_size_bytes > self.model_contract.maximum_block_manifest_file_bytes() {
            return Err(PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                maximum_size_bytes: self.model_contract.maximum_block_manifest_file_bytes(),
                estimated_block_bytes: manifest_file_size_bytes,
            });
        }
        // Use actual written file sizes for quota admission. The contract's
        // exact-size checks above prove those bytes also match predicted geometry.
        let total_size_bytes = sequence_state_file
            .as_ref()
            .map_or(0, |staged_file| staged_file.file_size_bytes)
            .checked_add(
                boundary_state_file
                    .as_ref()
                    .map_or(0, |staged_file| staged_file.file_size_bytes),
            )
            .and_then(|state_size_bytes| state_size_bytes.checked_add(manifest_file_size_bytes))
            .ok_or_else(
                || PersistentPromptCacheDiskStoreError::GlobalPromptCacheSizeOverflow {
                    global_prompt_cache_root_directory: staging_block_directory.to_path_buf(),
                },
            )?;
        if total_size_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                estimated_block_bytes: total_size_bytes,
            });
        }
        Ok(StagedBlockFiles {
            sequence_state_file,
            boundary_state_file,
            total_size_bytes,
        })
    }

    fn track_committed_block(
        &self,
        block_key: &PersistentPromptCacheBlockKey,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
        final_block_directory: &Path,
        staged_files: StagedBlockFiles,
    ) {
        self.lock_tracked_files().insert_block(
            block_key.block_hash(),
            TrackedPersistentPromptCacheBlock {
                block_directory_path: final_block_directory.to_path_buf(),
                block_index: block_key.block_index(),
                parent_block_hash: parent_block_key.map(PersistentPromptCacheBlockKey::block_hash),
                sequence_state_file: staged_files.sequence_state_file.map(|staged_file| {
                    TrackedPersistentPromptCacheFile {
                        file_path: final_block_directory.join(SEQUENCE_STATE_FILE_NAME),
                        file_size_bytes: staged_file.file_size_bytes,
                    }
                }),
                boundary_state_file: staged_files.boundary_state_file.map(|staged_file| {
                    TrackedPersistentPromptCacheFile {
                        file_path: final_block_directory.join(BOUNDARY_STATE_FILE_NAME),
                        file_size_bytes: staged_file.file_size_bytes,
                    }
                }),
            },
        );
    }

    fn protected_ancestry_for_commit(
        &self,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
    ) -> Vec<PathBuf> {
        parent_block_key.map_or_else(Vec::new, |parent_block_key| {
            self.lock_tracked_files()
                .protected_ancestry_directory_paths(parent_block_key.block_hash())
        })
    }

    fn parent_boundary_reclaim_after_commit(
        &self,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
        child_size_bytes: u64,
    ) -> Option<ParentBoundaryReclaim> {
        if !self.model_contract.has_sequence_state() || !self.model_contract.has_boundary_state() {
            return None;
        }
        let parent_block_key = parent_block_key?;
        if persistent_prompt_cache_boundary_is_common_prefix_checkpoint(
            parent_block_key.block_index(),
            self.model_contract.common_prefix_checkpoint_stride_blocks(),
        ) || self.total_size_bytes().saturating_add(child_size_bytes)
            <= self.global_prompt_cache_maximum_size_bytes
        {
            return None;
        }
        let tracked_files = self.lock_tracked_files();
        let parent_block = tracked_files.block(&parent_block_key.block_hash())?;
        let parent_boundary_file = parent_block.boundary_state_file.as_ref()?;
        Some(ParentBoundaryReclaim {
            block_hash: parent_block_key.block_hash(),
            block_directory_path: parent_block.block_directory_path.clone(),
            file_path: parent_boundary_file.file_path.clone(),
            file_size_bytes: parent_boundary_file.file_size_bytes,
        })
    }
}

struct StagedBlockFiles {
    sequence_state_file: Option<StagedPersistentPromptCacheStateFile>,
    boundary_state_file: Option<StagedPersistentPromptCacheStateFile>,
    total_size_bytes: u64,
}

struct ParentBoundaryReclaim {
    block_hash: [u8; 32],
    block_directory_path: PathBuf,
    file_path: PathBuf,
    file_size_bytes: u64,
}

fn create_staging_directory(
    staging_directory: &Path,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    std::fs::create_dir(staging_directory).map_err(|source| {
        PersistentPromptCacheDiskStoreError::CreatePromptCacheDirectory {
            persistent_prompt_cache_directory: staging_directory.to_path_buf(),
            source,
        }
    })
}

fn unique_staging_block_directory(blocks_directory: &Path, block_name: &str) -> PathBuf {
    let current_time_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    blocks_directory.join(format!(
        "{block_name}.staging-{}-{current_time_nanos}",
        std::process::id()
    ))
}

fn cleanup_staging_after_error(
    staging_directory: &Path,
    original_error: PersistentPromptCacheDiskStoreError,
) -> PersistentPromptCacheDiskStoreError {
    // A cleanup failure wins because it leaves bytes that startup must recover
    // and may prevent the quota from being satisfied. Otherwise retain the
    // operation's original, more useful failure.
    remove_cache_owned_directory_or_confirm_absent(staging_directory)
        .err()
        .unwrap_or(original_error)
}

fn validate_staged_file_size(
    staged_file: &StagedPersistentPromptCacheStateFile,
    expected_file_size_bytes: u64,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    if staged_file.file_size_bytes == expected_file_size_bytes {
        return Ok(());
    }
    Err(
        PersistentPromptCacheDiskStoreError::WrittenFileSizeMismatch {
            file_path: staged_file.file_path.clone(),
            reported_size_bytes: expected_file_size_bytes,
            actual_size_bytes: staged_file.file_size_bytes,
        },
    )
}
