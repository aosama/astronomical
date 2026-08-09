use crate::{
    InferenceEngineError, PersistentPromptCacheDiskStore,
    PersistentSpeculativePrefillPolicyIdentity,
};

use super::Qwen3_5EngineState;
use super::fatal_engine_error;
use super::speculative_prefill_failure::configured_speculative_prefill_activation_failure;

impl Qwen3_5EngineState {
    pub(super) fn purge_obsolete_speculative_prefill_policy_state(
        &self,
        resolved_target_model_id: &str,
        resolved_target_model_revision: &str,
        loaded_draft_model_revision: Option<&str>,
        target_persistent_prompt_cache: Option<&PersistentPromptCacheDiskStore>,
        draft_persistent_prompt_cache: Option<&PersistentPromptCacheDiskStore>,
    ) -> Result<(), InferenceEngineError> {
        if !self.speculative_prefill.enabled
            || self.persistent_prompt_cache_disk_store_config.is_none()
        {
            return Ok(());
        }
        let active_speculative_prefill_policy_identity =
            PersistentSpeculativePrefillPolicyIdentity::new(
                resolved_target_model_id.to_owned(),
                resolved_target_model_revision.to_owned(),
                self.speculative_prefill
                    .draft_model_id
                    .clone()
                    .ok_or_else(|| {
                        fatal_engine_error("configured SpecPrefill has no drafter model identifier")
                    })?,
                loaded_draft_model_revision
                    .ok_or_else(|| {
                        fatal_engine_error("configured SpecPrefill has no loaded drafter revision")
                    })?
                    .to_owned(),
                self.speculative_prefill.keep_percentage,
            );
        let target_purge_outcome = target_persistent_prompt_cache
            .ok_or_else(|| {
                fatal_engine_error(
                    "configured SpecPrefill target prompt-state storage is unavailable",
                )
            })?
            .purge_obsolete_speculative_prefill_keep_percentage_entries(
                &active_speculative_prefill_policy_identity,
            )
            .map_err(|purge_error| {
                configured_speculative_prefill_activation_failure(
                    "target keep-percentage state purge",
                    purge_error,
                )
            })?;
        let drafter_purge_outcome = draft_persistent_prompt_cache
            .ok_or_else(|| {
                fatal_engine_error(
                    "configured SpecPrefill drafter prompt-state storage is unavailable",
                )
            })?
            .purge_obsolete_speculative_prefill_keep_percentage_entries(
                &active_speculative_prefill_policy_identity,
            )
            .map_err(|purge_error| {
                configured_speculative_prefill_activation_failure(
                    "drafter keep-percentage selection purge",
                    purge_error,
                )
            })?;
        tracing::info!(
            target_selection_count = target_purge_outcome.speculative_prefill_selection_count,
            target_sparse_state_count = target_purge_outcome.speculative_prefill_target_state_count,
            drafter_selection_count = drafter_purge_outcome.speculative_prefill_selection_count,
            drafter_sparse_state_count =
                drafter_purge_outcome.speculative_prefill_target_state_count,
            "purged obsolete keep-percentage SpecPrefill SSD state"
        );
        Ok(())
    }
}
