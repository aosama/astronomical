//! Synchronous publication entry point and idempotency rules.
//!
//! A successful return means the requested block is durably available now. The
//! caller may advance its parent cursor for both `Published` and
//! `AlreadyPublished`; there is no queued, skipped, or eventually-written state.

use std::collections::HashMap;
use std::path::PathBuf;

use runtime_integration::MlxRuntime;

use model_primitives::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::block_key::AstroSsdPromptCacheBlockKey;
use super::block_manifest::AstroSsdPromptCacheBlockManifest;
use super::disk_store::AstroSsdPromptCacheDiskStore;
use super::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use super::disk_store_file::{
    AstroSsdPromptCacheFileKind, open_without_following_symlinks,
    remove_astro_ssd_prompt_cache_owned_directory_or_confirm_absent, validate_current_file_header,
};
use super::disk_store_index::TrackedAstroSsdPromptCacheBlock;
use mlx_bridge::MlxArray;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AstroSsdPromptCachePublicationOutcome {
    /// This call durably committed missing state.
    Published,
    /// An exact, fully validated block was already durable.
    AlreadyPublished,
}

impl AstroSsdPromptCacheDiskStore {
    pub fn publish_block(
        &self,
        runtime: &MlxRuntime,
        block_key: &AstroSsdPromptCacheBlockKey,
        parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
        sequence_state_tensors: &HashMap<String, MlxArray>,
        boundary_state_tensors: &HashMap<String, MlxArray>,
    ) -> Result<AstroSsdPromptCachePublicationOutcome, AstroSsdPromptCacheDiskStoreError> {
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
        block_key: &AstroSsdPromptCacheBlockKey,
        parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
        sequence_state_tensors: &HashMap<String, MlxArray>,
        boundary_state_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<AstroSsdPromptCachePublicationOutcome, AstroSsdPromptCacheDiskStoreError> {
        // Serialize scan/quota/rename/index mutations inside this process. The
        // lock does not weaken disk validation: files can still be modified by
        // another process or by the user, so existing content is revalidated.
        let _write_operation_guard = self.lock_write_operations();
        self.prepare_active_model_storage_directories()?;
        self.validate_publication_request(
            block_key,
            parent_block_key,
            sequence_state_tensors,
            boundary_state_tensors,
        )?;

        if let Some(existing_block) = self.indexed_block_still_on_disk(&block_key.block_hash())? {
            return self.publish_over_existing_block(
                runtime,
                block_key,
                parent_block_key,
                &existing_block,
                boundary_state_tensors,
                performance_attribution,
            );
        }

        self.require_published_parent(block_key, parent_block_key)?;
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
        Ok(AstroSsdPromptCachePublicationOutcome::Published)
    }

    fn validate_publication_request(
        &self,
        block_key: &AstroSsdPromptCacheBlockKey,
        parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
        sequence_state_tensors: &HashMap<String, MlxArray>,
        boundary_state_tensors: &HashMap<String, MlxArray>,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
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
        validate_requested_block_ancestry(block_key, parent_block_key)
    }

    /// The block the index holds for `block_hash`, or `None` when it holds none
    /// or when the block's directory is gone from disk.
    fn indexed_block_still_on_disk(
        &self,
        block_hash: &[u8; 32],
    ) -> Result<Option<TrackedAstroSsdPromptCacheBlock>, AstroSsdPromptCacheDiskStoreError> {
        // The index is an acceleration structure, never authority. If its path
        // disappeared, discard the stale entry and account from disk again.
        let Some(existing_block) = self.lock_tracked_files().block(block_hash).cloned() else {
            return Ok(None);
        };
        if existing_block.block_directory_path.is_dir() {
            return Ok(Some(existing_block));
        }
        self.lock_tracked_files().remove_block(block_hash);
        self.refresh_global_astro_ssd_prompt_cache_accounting()?;
        Ok(None)
    }

    /// Publishing a block the index already holds: nothing to do when it is
    /// complete, a repair when only its boundary snapshot is missing, and a
    /// refusal when what is stored disagrees with the request.
    fn publish_over_existing_block(
        &self,
        runtime: &MlxRuntime,
        block_key: &AstroSsdPromptCacheBlockKey,
        parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
        existing_block: &TrackedAstroSsdPromptCacheBlock,
        boundary_state_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<AstroSsdPromptCachePublicationOutcome, AstroSsdPromptCacheDiskStoreError> {
        let block_hash = block_key.block_hash();
        self.validate_existing_block_for_publication(block_key, parent_block_key, existing_block)?;
        if !existing_block_matches_requested_topology(
            existing_block,
            block_key.block_index(),
            parent_block_key.map(AstroSsdPromptCacheBlockKey::block_hash),
        ) {
            return Err(
                AstroSsdPromptCacheDiskStoreError::ExistingBlockTopologyMismatch { block_hash },
            );
        }
        // Idempotency is granted only after manifest and present state files
        // pass full validation. Hash equality alone cannot prove topology.
        match existing_block_completeness(
            self.model_contract.has_sequence_state(),
            self.model_contract.has_boundary_state(),
            existing_block,
        ) {
            ExistingBlockCompleteness::Complete => {
                Ok(AstroSsdPromptCachePublicationOutcome::AlreadyPublished)
            }
            ExistingBlockCompleteness::SequenceStateMissing => {
                Err(AstroSsdPromptCacheDiskStoreError::ExistingBlockTopologyMismatch { block_hash })
            }
            ExistingBlockCompleteness::BoundaryStateMissing => {
                // Sequence state may remain valid after retention compacts a parent
                // boundary. Reaching the same block as a leaf restores that single
                // missing file without replacing the sequence state.
                self.publish_missing_boundary_state_transaction(
                    runtime,
                    block_key,
                    boundary_state_tensors,
                    performance_attribution,
                )?;
                Ok(AstroSsdPromptCachePublicationOutcome::Published)
            }
        }
    }

    fn require_published_parent(
        &self,
        block_key: &AstroSsdPromptCacheBlockKey,
        parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
        // Children are never admitted speculatively. Requiring the parent in the
        // validated index guarantees every published non-root remains restorable.
        if let Some(parent_block_key) = parent_block_key
            && self
                .lock_tracked_files()
                .block(&parent_block_key.block_hash())
                .is_none()
        {
            return Err(AstroSsdPromptCacheDiskStoreError::ParentStateNotPublished {
                block_index: block_key.block_index(),
            });
        }
        Ok(())
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
        block_key: &AstroSsdPromptCacheBlockKey,
        parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
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
        let parent_block_hash = parent_block_key.map(AstroSsdPromptCacheBlockKey::block_hash);
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
            PerformanceOperation::AstroSsdPromptCacheRetentionCleanup,
            |_performance_attribution| {
                superseded_sibling_directory_paths.iter().try_for_each(
                    |superseded_directory_path| {
                        remove_astro_ssd_prompt_cache_owned_directory_or_confirm_absent(
                            superseded_directory_path,
                        )
                    },
                )
            },
        )?;
        self.lock_tracked_files()
            .remove_blocks_by_directory_paths(&superseded_sibling_directory_paths);
        self.refresh_global_astro_ssd_prompt_cache_accounting()
    }

    fn validate_existing_block_for_publication(
        &self,
        block_key: &AstroSsdPromptCacheBlockKey,
        parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
        existing_block: &TrackedAstroSsdPromptCacheBlock,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
        let block_hash = block_key.block_hash();
        let block_manifest = AstroSsdPromptCacheBlockManifest::read_from_block_directory(
            &existing_block.block_directory_path,
            &self.model_contract,
        )?;
        if block_manifest.block_hash().ok() != Some(block_hash)
            || block_manifest.block_index() != block_key.block_index()
            || block_manifest.parent_block_hash()
                != parent_block_key.map(AstroSsdPromptCacheBlockKey::block_hash)
        {
            return Err(
                AstroSsdPromptCacheDiskStoreError::ExistingBlockTopologyMismatch { block_hash },
            );
        }
        validate_existing_state_file(
            existing_block.sequence_state_file.as_ref(),
            AstroSsdPromptCacheFileKind::SequenceStateBlock,
            &self.model_contract,
        )?;
        validate_existing_state_file(
            existing_block.boundary_state_file.as_ref(),
            AstroSsdPromptCacheFileKind::BoundaryStateSnapshot,
            &self.model_contract,
        )?;
        Ok(())
    }
}

/// What an already-indexed block lacks, judged against the state kinds the
/// model contract requires of every block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExistingBlockCompleteness {
    Complete,
    /// The sequence state is missing. Retention removes only a boundary
    /// snapshot and keeps sequence state, so publishing a boundary cannot
    /// repair this block and the request is refused.
    SequenceStateMissing,
    /// Only the boundary snapshot is missing, as retention leaves a compacted
    /// parent. Publishing it again restores the block.
    BoundaryStateMissing,
}

fn existing_block_completeness(
    model_requires_sequence_state: bool,
    model_requires_boundary_state: bool,
    existing_block: &TrackedAstroSsdPromptCacheBlock,
) -> ExistingBlockCompleteness {
    let sequence_state_is_complete =
        !model_requires_sequence_state || existing_block.sequence_state_file.is_some();
    let boundary_state_is_complete =
        !model_requires_boundary_state || existing_block.boundary_state_file.is_some();
    if sequence_state_is_complete && boundary_state_is_complete {
        ExistingBlockCompleteness::Complete
    } else if !sequence_state_is_complete {
        ExistingBlockCompleteness::SequenceStateMissing
    } else {
        ExistingBlockCompleteness::BoundaryStateMissing
    }
}

fn existing_block_matches_requested_topology(
    existing_block: &TrackedAstroSsdPromptCacheBlock,
    requested_block_index: u32,
    requested_parent_block_hash: Option<[u8; 32]>,
) -> bool {
    existing_block.block_index == requested_block_index
        && existing_block.parent_block_hash == requested_parent_block_hash
}

fn validate_existing_state_file(
    tracked_file: Option<&super::disk_store_index::TrackedAstroSsdPromptCacheFile>,
    file_kind: AstroSsdPromptCacheFileKind,
    model_contract: &crate::AstroSsdPromptCacheModelContract,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    let Some(tracked_file) = tracked_file else {
        return Ok(());
    };
    let state_file =
        open_without_following_symlinks(&tracked_file.file_path).map_err(|source| {
            AstroSsdPromptCacheDiskStoreError::OpenBlockFile {
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
    .map_err(|source| AstroSsdPromptCacheDiskStoreError::ValidateBlock {
        block_file_path: tracked_file.file_path.clone(),
        source,
    })
}

fn validate_requested_block_ancestry(
    block_key: &AstroSsdPromptCacheBlockKey,
    parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
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
            AstroSsdPromptCacheDiskStoreError::InvalidRequestedBlockAncestry {
                block_index: block_key.block_index(),
            },
        )
    }
}

fn validate_state_kind_tensor_presence(
    state_kind: &'static str,
    expected_present: bool,
    actual_tensor_count: usize,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    if expected_present == (actual_tensor_count > 0) {
        return Ok(());
    }
    Err(
        AstroSsdPromptCacheDiskStoreError::StateKindTensorPresenceMismatch {
            state_kind,
            expected_present,
            actual_tensor_count,
        },
    )
}

#[cfg(test)]
#[path = "disk_store_write_tests.rs"]
mod disk_store_write_tests;
