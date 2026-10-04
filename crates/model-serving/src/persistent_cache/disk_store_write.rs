//! Synchronous publication entry point and idempotency rules.
//!
//! A successful return means the requested block is durably available now. The
//! caller may advance its parent cursor for both `Published` and
//! `AlreadyPublished`; there is no queued, skipped, or eventually-written state.

use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_runtime_integration::MlxRuntime;

use crate::{PerformanceAttribution, PerformanceOperation};

use super::block_key::PersistentPromptCacheBlockKey;
use super::block_manifest::PersistentPromptCacheBlockManifest;
use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, open_without_following_symlinks,
    remove_cache_owned_directory_or_confirm_absent, validate_current_file_header,
};
use super::disk_store_index::TrackedPersistentPromptCacheBlock;
use astronomical_mlx_c_rust::MlxArray;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentPromptCachePublicationOutcome {
    /// This call durably committed missing state.
    Published,
    /// An exact, fully validated block was already durable.
    AlreadyPublished,
}

impl PersistentPromptCacheDiskStore {
    pub fn publish_block(
        &self,
        runtime: &MlxRuntime,
        block_key: &PersistentPromptCacheBlockKey,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
        sequence_state_tensors: &HashMap<String, MlxArray>,
        boundary_state_tensors: &HashMap<String, MlxArray>,
    ) -> Result<PersistentPromptCachePublicationOutcome, PersistentPromptCacheDiskStoreError> {
        let mut performance_attribution = PerformanceAttribution::disabled();
        self.publish_block_with_performance_attribution(
            runtime,
            block_key,
            parent_block_key,
            sequence_state_tensors,
            boundary_state_tensors,
            &mut performance_attribution,
        )
    }

    pub fn publish_block_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        block_key: &PersistentPromptCacheBlockKey,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
        sequence_state_tensors: &HashMap<String, MlxArray>,
        boundary_state_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PersistentPromptCachePublicationOutcome, PersistentPromptCacheDiskStoreError> {
        // Serialize scan/quota/rename/index mutations inside this process. The
        // lock does not weaken disk validation: files can still be modified by
        // another process or by the user, so existing content is revalidated.
        let _write_operation_guard = self.lock_write_operations();
        self.prepare_active_model_storage_directories()?;
        validate_state_kind_tensor_presence(
            "sequence",
            self.model_contract.has_sequence_state(),
            sequence_state_tensors.len(),
        )?;
        validate_state_kind_tensor_presence(
            "boundary",
            self.model_contract.has_boundary_state(),
            boundary_state_tensors.len(),
        )?;
        validate_requested_block_ancestry(block_key, parent_block_key)?;

        let block_hash = block_key.block_hash();
        // The index is an acceleration structure, never authority. If its path
        // disappeared, discard the stale entry and account from disk again.
        let mut existing_block = self.lock_tracked_files().block(&block_hash).cloned();
        if existing_block
            .as_ref()
            .is_some_and(|tracked_block| !tracked_block.block_directory_path.is_dir())
        {
            self.lock_tracked_files().remove_block(&block_hash);
            self.refresh_global_prompt_cache_accounting()?;
            existing_block = None;
        }
        if let Some(existing_block) = existing_block {
            self.validate_existing_block_for_publication(
                block_key,
                parent_block_key,
                &existing_block,
            )?;
            if existing_block.block_index != block_key.block_index()
                || existing_block.parent_block_hash
                    != parent_block_key.map(PersistentPromptCacheBlockKey::block_hash)
            {
                return Err(
                    PersistentPromptCacheDiskStoreError::ExistingBlockTopologyMismatch {
                        block_hash,
                    },
                );
            }
            let sequence_state_is_complete = !self.model_contract.has_sequence_state()
                || existing_block.sequence_state_file.is_some();
            let boundary_state_is_complete = !self.model_contract.has_boundary_state()
                || existing_block.boundary_state_file.is_some();
            // Idempotency is granted only after manifest and present state files
            // pass full validation. Hash equality alone cannot prove topology.
            if sequence_state_is_complete && boundary_state_is_complete {
                return Ok(PersistentPromptCachePublicationOutcome::AlreadyPublished);
            }
            if !sequence_state_is_complete {
                return Err(
                    PersistentPromptCacheDiskStoreError::ExistingBlockTopologyMismatch {
                        block_hash,
                    },
                );
            }
            // Sequence state may remain valid after retention compacts a parent
            // boundary. Reaching the same block as a leaf restores that single
            // missing file without replacing the sequence state.
            self.publish_missing_boundary_state_transaction(
                runtime,
                block_key,
                boundary_state_tensors,
                performance_attribution,
            )?;
            return Ok(PersistentPromptCachePublicationOutcome::Published);
        }

        // Children are never admitted speculatively. Requiring the parent in the
        // validated index guarantees every published non-root remains restorable.
        if let Some(parent_block_key) = parent_block_key
            && self
                .lock_tracked_files()
                .block(&parent_block_key.block_hash())
                .is_none()
        {
            return Err(
                PersistentPromptCacheDiskStoreError::ParentStateNotPublished {
                    block_index: block_key.block_index(),
                },
            );
        }
        self.publish_new_block_transaction(
            runtime,
            block_key,
            parent_block_key,
            sequence_state_tensors,
            boundary_state_tensors,
            performance_attribution,
        )?;
        self.supersede_shorter_partial_tail_siblings(
            block_key,
            parent_block_key,
            performance_attribution,
        )?;
        Ok(PersistentPromptCachePublicationOutcome::Published)
    }

    /// Removes strictly shorter tail siblings sharing the new tail's chain
    /// position. Prompts grow monotonically within a conversation, so once a
    /// longer tail is durable at a chain position, every shorter tail written
    /// earlier by that same conversation can never be probed again. Siblings
    /// with equal or larger sequence files stay: equal-length tails serve
    /// divergent conversations that share the parent prefix, and longer tails
    /// remain the growth path of their own conversations. Full blocks are never
    /// touched because their sequence files hold more tokens than any tail.
    fn supersede_shorter_partial_tail_siblings(
        &self,
        block_key: &PersistentPromptCacheBlockKey,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        if !self.model_contract.has_sequence_state()
            || block_key.token_count() >= self.model_contract.block_token_count()
        {
            return Ok(());
        }
        // Sequence-state file bytes grow strictly with token count, so a
        // smaller file proves a strictly shorter tail.
        let new_tail_sequence_file_size_bytes = self
            .model_contract
            .sequence_state_file_bytes_for_block_token_count(block_key.token_count())?;
        let parent_block_hash = parent_block_key.map(PersistentPromptCacheBlockKey::block_hash);
        let superseded_sibling_directory_paths: Vec<PathBuf> = self
            .lock_tracked_files()
            .tracked_blocks()
            .into_iter()
            .filter(|(sibling_block_hash, tracked_sibling)| {
                *sibling_block_hash != block_key.block_hash()
                    && tracked_sibling.block_index == block_key.block_index()
                    && tracked_sibling.parent_block_hash == parent_block_hash
                    && tracked_sibling.sequence_state_file.as_ref().is_some_and(
                        |sequence_state_file| {
                            sequence_state_file.file_size_bytes < new_tail_sequence_file_size_bytes
                        },
                    )
            })
            .map(|(_, tracked_sibling)| tracked_sibling.block_directory_path)
            .collect();
        if superseded_sibling_directory_paths.is_empty() {
            return Ok(());
        }
        performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCacheRetentionCleanup,
            |_performance_attribution| {
                superseded_sibling_directory_paths.iter().try_for_each(
                    |superseded_directory_path| {
                        remove_cache_owned_directory_or_confirm_absent(superseded_directory_path)
                    },
                )
            },
        )?;
        self.lock_tracked_files()
            .remove_blocks_by_directory_paths(&superseded_sibling_directory_paths);
        self.refresh_global_prompt_cache_accounting()
    }

    fn validate_existing_block_for_publication(
        &self,
        block_key: &PersistentPromptCacheBlockKey,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
        existing_block: &TrackedPersistentPromptCacheBlock,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        let block_hash = block_key.block_hash();
        let block_manifest = PersistentPromptCacheBlockManifest::read_from_block_directory(
            &existing_block.block_directory_path,
            &self.model_contract,
        )?;
        if block_manifest.block_hash().ok() != Some(block_hash)
            || block_manifest.block_index() != block_key.block_index()
            || block_manifest.parent_block_hash()
                != parent_block_key.map(PersistentPromptCacheBlockKey::block_hash)
        {
            return Err(
                PersistentPromptCacheDiskStoreError::ExistingBlockTopologyMismatch { block_hash },
            );
        }
        validate_existing_state_file(
            existing_block.sequence_state_file.as_ref(),
            PersistentPromptCacheFileKind::SequenceStateBlock,
            &self.model_contract,
        )?;
        validate_existing_state_file(
            existing_block.boundary_state_file.as_ref(),
            PersistentPromptCacheFileKind::BoundaryStateSnapshot,
            &self.model_contract,
        )?;
        Ok(())
    }
}

fn validate_existing_state_file(
    tracked_file: Option<&super::disk_store_index::TrackedPersistentPromptCacheFile>,
    file_kind: PersistentPromptCacheFileKind,
    model_contract: &crate::PersistentPromptCacheModelContract,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    let Some(tracked_file) = tracked_file else {
        return Ok(());
    };
    let state_file =
        open_without_following_symlinks(&tracked_file.file_path).map_err(|source| {
            PersistentPromptCacheDiskStoreError::OpenBlockFile {
                block_file_path: tracked_file.file_path.clone(),
                source,
            }
        })?;
    validate_current_file_header(
        file_kind,
        &state_file,
        &tracked_file.file_path,
        model_contract,
    )
    .map(|_header_block_token_count| ())
    .map_err(
        |source| PersistentPromptCacheDiskStoreError::ValidateBlock {
            block_file_path: tracked_file.file_path.clone(),
            source,
        },
    )
}

fn validate_requested_block_ancestry(
    block_key: &PersistentPromptCacheBlockKey,
    parent_block_key: Option<&PersistentPromptCacheBlockKey>,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    // Root is exactly index zero with no parent. Every other block must advance
    // one ordinal from a supplied parent; gaps and alternate roots fail closed.
    let ancestry_is_valid = match (block_key.block_index(), parent_block_key) {
        (0, None) => true,
        (0, Some(_)) | (_, None) => false,
        (block_index, Some(parent_block_key)) => {
            parent_block_key.block_index().checked_add(1) == Some(block_index)
        }
    };
    if ancestry_is_valid {
        Ok(())
    } else {
        Err(
            PersistentPromptCacheDiskStoreError::InvalidRequestedBlockAncestry {
                block_index: block_key.block_index(),
            },
        )
    }
}

fn validate_state_kind_tensor_presence(
    state_kind: &'static str,
    expected_present: bool,
    actual_tensor_count: usize,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    if expected_present == (actual_tensor_count > 0) {
        return Ok(());
    }
    Err(
        PersistentPromptCacheDiskStoreError::StateKindTensorPresenceMismatch {
            state_kind,
            expected_present,
            actual_tensor_count,
        },
    )
}
