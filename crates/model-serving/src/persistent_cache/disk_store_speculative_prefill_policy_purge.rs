use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, open_without_following_symlinks,
    remove_cache_owned_file_or_confirm_absent,
};
use super::speculative_prefill_policy::PersistentSpeculativePrefillPolicyIdentity;
use super::speculative_prefill_selection::PersistentSpeculativePrefillSelectionFileHeader;
use super::speculative_prefill_target_state::PersistentSpeculativePrefillTargetStateFileHeader;

/// Counts policy-dependent files removed without including preserved exact,
/// visual, dense-drafter, or unrelated model state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PersistentSpeculativePrefillPolicyPurgeOutcome {
    pub speculative_prefill_selection_count: usize,
    pub speculative_prefill_target_state_count: usize,
}

impl PersistentPromptCacheDiskStore {
    /// Removes only SpecPrefill selections and sparse target states belonging to
    /// the active target-and-drafter pairing under another keep percentage.
    pub fn purge_obsolete_speculative_prefill_keep_percentage_entries(
        &self,
        active_policy_identity: &PersistentSpeculativePrefillPolicyIdentity,
    ) -> Result<
        PersistentSpeculativePrefillPolicyPurgeOutcome,
        PersistentPromptCacheDiskStoreError,
    > {
        let _write_operation_guard = self.lock_write_operations();
        let speculative_prefill_selection_count = self.purge_obsolete_policy_file_kind(
            PersistentPromptCacheFileKind::SpeculativePrefillSelection,
            active_policy_identity,
        )?;
        let speculative_prefill_target_state_count = self.purge_obsolete_policy_file_kind(
            PersistentPromptCacheFileKind::SpeculativePrefillTargetState,
            active_policy_identity,
        )?;
        Ok(PersistentSpeculativePrefillPolicyPurgeOutcome {
            speculative_prefill_selection_count,
            speculative_prefill_target_state_count,
        })
    }

    fn purge_obsolete_policy_file_kind(
        &self,
        persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
        active_policy_identity: &PersistentSpeculativePrefillPolicyIdentity,
    ) -> Result<usize, PersistentPromptCacheDiskStoreError> {
        let tracked_policy_files = self
            .lock_tracked_files()
            .files(persistent_prompt_cache_file_kind);
        let mut purged_file_count = 0usize;
        for (policy_file_hash, tracked_policy_file) in tracked_policy_files {
            let policy_file = open_without_following_symlinks(&tracked_policy_file.file_path)
                .map_err(|source| PersistentPromptCacheDiskStoreError::OpenBlockFile {
                    block_file_path: tracked_policy_file.file_path.clone(),
                    source,
                })?;
            let stored_policy_identity = match persistent_prompt_cache_file_kind {
                PersistentPromptCacheFileKind::SpeculativePrefillSelection => {
                    PersistentSpeculativePrefillSelectionFileHeader::read_model_bound_from_file(
                        &policy_file,
                        &tracked_policy_file.file_path,
                        self.model_contract_ref(),
                    )
                    .map(|selection_header| selection_header.policy_identity().clone())
                    .map_err(|source| {
                        PersistentPromptCacheDiskStoreError::ValidateModelSpecificArtifact {
                            artifact_file_path: tracked_policy_file.file_path.clone(),
                            source: Box::new(source),
                        }
                    })?
                }
                PersistentPromptCacheFileKind::SpeculativePrefillTargetState => {
                    PersistentSpeculativePrefillTargetStateFileHeader::read_model_bound_from_file(
                        &policy_file,
                        &tracked_policy_file.file_path,
                        self.model_contract_ref(),
                    )
                    .map(|target_state_header| target_state_header.policy_identity().clone())
                    .map_err(|description| {
                        PersistentPromptCacheDiskStoreError::ValidateModelSpecificArtifact {
                            artifact_file_path: tracked_policy_file.file_path.clone(),
                            source: Box::new(std::io::Error::other(description)),
                        }
                    })?
                }
                PersistentPromptCacheFileKind::SequenceStateBlock
                | PersistentPromptCacheFileKind::BoundaryStateSnapshot
                | PersistentPromptCacheFileKind::VisualEmbedding => continue,
            };
            if !stored_policy_identity.should_purge_for_active_keep_percentage(
                active_policy_identity.target_model_id(),
                active_policy_identity.target_model_revision(),
                active_policy_identity.drafter_model_id(),
                active_policy_identity.drafter_model_revision(),
                active_policy_identity.keep_percentage(),
            ) {
                continue;
            }
            remove_cache_owned_file_or_confirm_absent(&tracked_policy_file.file_path)?;
            self.untrack_file_and_subtract_global_accounting(
                persistent_prompt_cache_file_kind,
                policy_file_hash,
            );
            purged_file_count = purged_file_count.saturating_add(1);
        }
        Ok(purged_file_count)
    }
}
