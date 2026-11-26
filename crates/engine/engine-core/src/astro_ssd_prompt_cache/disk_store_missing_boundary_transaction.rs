//! Crash-safe publication of the one boundary-state file a block lacks.
//!
//! Startup compaction can leave a block with its sequence state and without
//! its boundary snapshot. This transaction restores only that file, in the
//! same order a whole-block publication follows: stage in a unique directory,
//! synchronize, admit under the global quota, rename, then synchronize and
//! expose the file through the in-memory index.

use std::collections::HashMap;
use std::path::Path;

use mlx_bridge::MlxArray;
use model_primitives::performance_attribution::{PerformanceAttribution, PerformanceOperation};
use runtime_integration::MlxRuntime;

use super::block_key::AstroSsdPromptCacheBlockKey;
use super::block_manifest::BOUNDARY_STATE_FILE_NAME;
use super::disk_store::AstroSsdPromptCacheDiskStore;
use super::disk_store_block_transaction::{
    cleanup_staging_after_error, create_staging_directory, unique_staging_block_directory,
    validate_staged_file_size,
};
use super::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use super::disk_store_file::{
    AstroSsdPromptCacheFileKind, StagedAstroSsdPromptCacheStateFile,
    remove_astro_ssd_prompt_cache_owned_directory_or_confirm_absent,
    save_direct_safetensors_file_with_name, synchronize_directory,
};
use super::disk_store_index::TrackedAstroSsdPromptCacheFile;

#[cfg(test)]
#[path = "disk_store_missing_boundary_transaction_tests.rs"]
mod disk_store_missing_boundary_transaction_tests;

impl AstroSsdPromptCacheDiskStore {
    pub(super) fn publish_missing_boundary_state_transaction(
        &self,
        runtime: &MlxRuntime,
        block_key: &AstroSsdPromptCacheBlockKey,
        boundary_state_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
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
                AstroSsdPromptCacheDiskStoreError::ExistingBlockTopologyMismatch { block_hash },
            )?;
        let staging_directory = unique_staging_block_directory(
            &self.blocks_directory,
            &missing_boundary_staging_name(&block_hash),
        );
        create_staging_directory(&staging_directory)?;
        let staged_boundary_state = self.stage_missing_boundary_state(
            runtime,
            block_key,
            boundary_state_tensors,
            &staging_directory,
            performance_attribution,
        )?;
        self.admit_missing_boundary_state(
            block_hash,
            staged_boundary_state.file_size_bytes,
            &staging_directory,
            performance_attribution,
        )?;
        let final_boundary_file_path = existing_block
            .block_directory_path
            .join(BOUNDARY_STATE_FILE_NAME);
        rename_staged_boundary_into_block(
            &staged_boundary_state,
            &final_boundary_file_path,
            &staging_directory,
            performance_attribution,
        )?;
        remove_astro_ssd_prompt_cache_owned_directory_or_confirm_absent(&staging_directory)?;
        performance_attribution.measure_operation(
            PerformanceOperation::AstroSsdPromptCachePublicationSynchronizationWait,
            |_performance_attribution| {
                synchronize_directory(&existing_block.block_directory_path)?;
                synchronize_directory(&self.blocks_directory)
            },
        )?;
        self.lock_tracked_files().insert_file(
            AstroSsdPromptCacheFileKind::BoundaryStateSnapshot,
            block_hash,
            TrackedAstroSsdPromptCacheFile {
                file_path: final_boundary_file_path,
                file_size_bytes: staged_boundary_state.file_size_bytes,
            },
        );
        self.refresh_global_astro_ssd_prompt_cache_accounting()?;
        Ok(())
    }

    /// Writes the boundary-state file into the staging directory, checks its
    /// size against the model contract, and synchronizes the directory. A
    /// failed write, a size that differs from the contract's, or a failed
    /// synchronization removes the staging directory before the error
    /// propagates.
    fn stage_missing_boundary_state(
        &self,
        runtime: &MlxRuntime,
        block_key: &AstroSsdPromptCacheBlockKey,
        boundary_state_tensors: &HashMap<String, MlxArray>,
        staging_directory: &Path,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<StagedAstroSsdPromptCacheStateFile, AstroSsdPromptCacheDiskStoreError> {
        let staged_boundary_state = save_direct_safetensors_file_with_name(
            runtime,
            staging_directory,
            BOUNDARY_STATE_FILE_NAME,
            boundary_state_tensors,
            block_key.token_count(),
            &self.model_contract,
            performance_attribution,
        )
        .map_err(|staging_error| cleanup_staging_after_error(staging_directory, staging_error))?;
        validate_staged_file_size(
            &staged_boundary_state,
            self.model_contract
                .boundary_state_file_bytes_for_block_token_count(block_key.token_count())?,
        )
        .map_err(|size_error| cleanup_staging_after_error(staging_directory, size_error))?;
        performance_attribution
            .measure_operation(
                PerformanceOperation::AstroSsdPromptCachePublicationSynchronizationWait,
                |_performance_attribution| synchronize_directory(staging_directory),
            )
            .map_err(|sync_error| cleanup_staging_after_error(staging_directory, sync_error))?;
        Ok(staged_boundary_state)
    }

    /// Makes room under the global quota for the staged boundary file while
    /// protecting the ancestry of the block being repaired. A refusal removes
    /// the staging directory before the error propagates.
    fn admit_missing_boundary_state(
        &self,
        block_hash: [u8; 32],
        staged_boundary_state_bytes: u64,
        staging_directory: &Path,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
        let protected_block_directory_paths = self
            .lock_tracked_files()
            .protected_ancestry_directory_paths(block_hash);
        performance_attribution
            .measure_operation(
                PerformanceOperation::AstroSsdPromptCacheGlobalQuotaEviction,
                |_performance_attribution| {
                    self.enforce_global_astro_ssd_prompt_cache_quota_for_commit(
                        staged_boundary_state_bytes,
                        0,
                        &protected_block_directory_paths,
                        Some(staging_directory),
                    )
                },
            )
            .map_err(|quota_error| cleanup_staging_after_error(staging_directory, quota_error))
    }
}

/// The staging directory name for a block's missing boundary file: the block's
/// hash plus a `-boundary` suffix, which keeps it apart from the staging
/// directory a whole-block publication of the same block would use.
fn missing_boundary_staging_name(block_hash: &[u8; 32]) -> String {
    format!(
        "{}-boundary",
        lowercase_hex::encode_lowercase_hex(block_hash)
    )
}

/// Moves the staged boundary file into the committed block's directory. A
/// failed rename removes the staging directory before the error propagates.
fn rename_staged_boundary_into_block(
    staged_boundary_state: &StagedAstroSsdPromptCacheStateFile,
    final_boundary_file_path: &Path,
    staging_directory: &Path,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    // Renaming one file within an already committed block is atomic. The
    // subsequent block-directory sync makes the new directory entry durable.
    performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCacheAtomicCommit,
        |_performance_attribution| {
            std::fs::rename(&staged_boundary_state.file_path, final_boundary_file_path).map_err(
                |source| {
                    cleanup_staging_after_error(
                        staging_directory,
                        AstroSsdPromptCacheDiskStoreError::RenameTempFile {
                            temp_file_path: staged_boundary_state.file_path.clone(),
                            block_file_path: final_boundary_file_path.to_path_buf(),
                            source,
                        },
                    )
                },
            )
        },
    )
}
