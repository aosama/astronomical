//! Restores the longest complete persistent prompt-cache prefix into live Qwen state.
//!
//! Lookup is allocation-free. After a complete chain is identified, all sequence
//! blocks and the newest required boundary are loaded, reconstructed, synchronized,
//! and released before memory admission is repeated for only the remaining context.
//!
//! The lookup diagnostics, block-key derivation, prefix enumeration, and the
//! load-and-reconstruct pipeline are shared with the streaming engine in
//! `crate::persistent_cache::restore_support`; this module keeps the resident
//! engine's in-memory prefix adoption and its no-reclamation admission policy.

use astronomical_ipc_protocol::{
    RequestId, WorkerInMemoryPrefixAdoptionOutcome, WorkerPersistentPromptCacheRequestDiagnostics,
};

use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::persistent_cache::prefix_lookup::PersistentPromptCachePrefixLookup;
use crate::persistent_cache::restore_support;
use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
};

use super::Qwen3_5EngineState;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;

/// The cache-specific portion of a newly admitted request's starting state.
///
/// `restored_token_count` drives the prefill cursor, while the u32 field is
/// reported in the public generation usage contract. Keeping the final block
/// key lets later cold-prefilled blocks extend the same content-addressed
/// chain instead of creating an unrelated root block.
pub(in crate::qwen3_5_resident::inference_execution) struct PersistentPromptCacheRestoreOutcome {
    pub(in crate::qwen3_5_resident::inference_execution) persistent_prompt_cache_token_count: u32,
    pub(in crate::qwen3_5_resident::inference_execution) restored_token_count: usize,
    pub(in crate::qwen3_5_resident::inference_execution) last_restored_persistent_prompt_cache_block_key:
        Option<PersistentPromptCacheBlockKey>,
    pub(in crate::qwen3_5_resident::inference_execution) persistent_prompt_cache_diagnostics:
        WorkerPersistentPromptCacheRequestDiagnostics,
}

/// How the in-memory adoption check resolved: either the retained live prefix
/// took over the restore, or the disk path continues with an adoption outcome
/// to report. The adopted outcome is boxed to keep the enum's cold variant
/// from inflating every resolution.
enum InMemoryPrefixAdoptionResolution {
    Adopted(Box<PersistentPromptCacheRestoreOutcome>),
    NotAdopted(Option<WorkerInMemoryPrefixAdoptionOutcome>),
}

/// The inputs of a restore-hit finalization: everything the counter stamping,
/// hit logging, and outcome construction need. The resident engine re-admits
/// nothing — its experts stay resident — so no admission fields appear here.
struct RestoreHitFinalization<'a> {
    request_id: &'a RequestId,
    prompt_token_count: usize,
    total_context_tokens: usize,
    restored_token_count: usize,
    complete_block_count: usize,
    restored_partial_tail_block_key: Option<&'a PersistentPromptCacheBlockKey>,
    lookup_diagnostics: &'a crate::PersistentPromptCacheLookupDiagnostics,
    target_expert_payload_bytes_reclaimed_before_restore: u64,
    persistent_prompt_cache_diagnostics: WorkerPersistentPromptCacheRequestDiagnostics,
    last_restored_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey>,
}

impl Qwen3_5EngineState {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::qwen3_5_resident::inference_execution) fn restore_persistent_prompt_cache_prefix(
        &mut self,
        request_id: RequestId,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        prompt_token_ids: &[u32],
        block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
        total_context_tokens: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PersistentPromptCacheRestoreOutcome, InferenceEngineError> {
        // First decide the longest usable prefix using only the store's small
        // in-memory index. No MLX array is allocated until a whole contiguous
        // prefix is known to be available.
        let lookup_result = performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCachePrefixLookup,
            |_performance_attribution| {
                PersistentPromptCachePrefixLookup::for_prompt_with_partial_tail_block_recovery(
                    &persistent_prompt_cache.model_contract,
                    prompt_token_ids,
                    block_causal_inputs,
                    |block_hash| persistent_prompt_cache.has_kv_block(block_hash),
                    |block_hash| persistent_prompt_cache.has_recurrent_snapshot(block_hash),
                )
            },
        );
        let restored_token_count = lookup_result.restored_token_count();
        let restored_partial_tail_block_key =
            lookup_result.restored_partial_tail_block_key().cloned();
        let lookup_diagnostics = lookup_result.diagnostics();
        let mut persistent_prompt_cache_diagnostics =
            restore_support::persistent_prompt_cache_request_diagnostics(
                persistent_prompt_cache.model_contract.block_token_count(),
                lookup_diagnostics,
                restored_token_count,
                lookup_diagnostics.restored_partial_tail_block_token_count(),
            );
        // In-memory prefix adoption (issue #1167): match the prompt's chain
        // against the retained live state before touching the disk store. An
        // adopted prefix can even rescue a disk miss, because the live state
        // may outlive blocks the store evicted.
        let in_memory_prefix_adoption_outcome = match self.try_adopt_retained_live_prefix(
            persistent_prompt_cache,
            prompt_token_ids,
            block_causal_inputs,
            restored_token_count,
            request_decoder_state,
            performance_attribution,
        )? {
            InMemoryPrefixAdoptionResolution::Adopted(adopted_restore_outcome) => {
                return Ok(*adopted_restore_outcome);
            }
            InMemoryPrefixAdoptionResolution::NotAdopted(adoption_outcome) => adoption_outcome,
        };
        persistent_prompt_cache_diagnostics.in_memory_prefix_adoption =
            in_memory_prefix_adoption_outcome;
        if restored_token_count == 0 {
            // A miss is normal, not an error. The cold outcome keeps the
            // caller's fallback path identical for short prompts, first
            // requests, and a directory with no matching prefix.
            return Ok(self.cold_restore_miss_outcome(
                &request_id,
                prompt_token_ids.len(),
                persistent_prompt_cache,
                lookup_diagnostics,
                persistent_prompt_cache_diagnostics,
            ));
        }

        let complete_block_count =
            restored_token_count / persistent_prompt_cache.model_contract.block_token_count();
        let prefix_enumeration = restore_support::enumerate_restored_prefix_blocks(
            persistent_prompt_cache,
            prompt_token_ids,
            block_causal_inputs,
            restored_token_count,
            restored_partial_tail_block_key.as_ref(),
        )?;
        // The resident engine keeps its experts resident, so restore does not
        // gate on a transient workspace or reclaim expert payloads; the byte
        // totals are computed for parity reporting only.
        let _persistent_prompt_cache_restore_temporary_workspace_bytes =
            super::restore_transient_workspace::restore_transient_workspace_bytes(
                prefix_enumeration.total_sequence_block_file_size_bytes,
                prefix_enumeration.recurrent_snapshot_file_size_bytes,
            );
        let target_expert_payload_bytes_reclaimed_before_restore = 0_u64;
        let last_restored_persistent_prompt_cache_block_key = self
            .load_and_reconstruct_restored_prefix(
                persistent_prompt_cache,
                prefix_enumeration.restored_persistent_prompt_cache_block_keys,
                restored_token_count,
                complete_block_count,
                restored_partial_tail_block_key.as_ref(),
                request_decoder_state,
                performance_attribution,
            )?;
        // The initial admission was conservative because no prefix was restored
        // yet. Re-admit against only uncached context after temporary load memory
        // is gone, allowing the ordinary request path to use reclaimed capacity.
        self.finalize_restored_prefix_hit(RestoreHitFinalization {
            request_id: &request_id,
            prompt_token_count: prompt_token_ids.len(),
            total_context_tokens,
            restored_token_count,
            complete_block_count,
            restored_partial_tail_block_key: restored_partial_tail_block_key.as_ref(),
            lookup_diagnostics,
            target_expert_payload_bytes_reclaimed_before_restore,
            persistent_prompt_cache_diagnostics,
            last_restored_persistent_prompt_cache_block_key,
        })
    }

    /// Cold-path miss outcome: attaches startup-cleanup evidence when the miss
    /// reason is one a startup sweep cures, logs the miss, records the miss
    /// counter, and returns the explicit cold state.
    fn cold_restore_miss_outcome(
        &mut self,
        request_id: &RequestId,
        prompt_token_count: usize,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        lookup_diagnostics: &crate::PersistentPromptCacheLookupDiagnostics,
        mut persistent_prompt_cache_diagnostics: WorkerPersistentPromptCacheRequestDiagnostics,
    ) -> PersistentPromptCacheRestoreOutcome {
        if restore_support::miss_reason_warrants_startup_cleanup_evidence(lookup_diagnostics) {
            persistent_prompt_cache_diagnostics.startup_cleanup_evidence = persistent_prompt_cache
                .take_startup_cleanup_evidence()
                .map(restore_support::worker_startup_cleanup_evidence);
        }
        restore_support::log_persistent_prompt_cache_restore_miss(
            request_id,
            prompt_token_count,
            lookup_diagnostics,
            persistent_prompt_cache_diagnostics
                .startup_cleanup_evidence
                .as_ref(),
        );
        self.persistent_prompt_cache_counters.record_cache_miss();
        PersistentPromptCacheRestoreOutcome {
            persistent_prompt_cache_token_count: 0,
            restored_token_count: 0,
            last_restored_persistent_prompt_cache_block_key: None,
            persistent_prompt_cache_diagnostics,
        }
    }

    /// Hit finalization: validates the restored span against the generation
    /// context, records the zero-reclaim total, logs the hit, stamps the
    /// counters, and builds the outcome. The resident engine performs no
    /// expert reclamation here — its experts stay resident by design.
    fn finalize_restored_prefix_hit(
        &mut self,
        hit_finalization: RestoreHitFinalization<'_>,
    ) -> Result<PersistentPromptCacheRestoreOutcome, InferenceEngineError> {
        let mut hit_finalization = hit_finalization;
        let _remaining_context_token_count = hit_finalization
            .total_context_tokens
            .checked_sub(hit_finalization.restored_token_count)
            .ok_or_else(|| {
                super::fatal_engine_error(
                    "persistent prompt-cache restore exceeded the generation context",
                )
            })?;
        let target_expert_payload_bytes_reclaimed_after_restore = 0_u64;
        let target_expert_payload_bytes_reclaimed_during_restore = hit_finalization
            .target_expert_payload_bytes_reclaimed_before_restore
            .saturating_add(target_expert_payload_bytes_reclaimed_after_restore);
        hit_finalization
            .persistent_prompt_cache_diagnostics
            .expert_bytes_reclaimed_for_restore =
            target_expert_payload_bytes_reclaimed_during_restore;
        restore_support::log_persistent_prompt_cache_restore_hit(
            hit_finalization.request_id,
            hit_finalization.prompt_token_count,
            hit_finalization.restored_token_count,
            hit_finalization.complete_block_count,
            hit_finalization.restored_partial_tail_block_key,
            hit_finalization.lookup_diagnostics,
            target_expert_payload_bytes_reclaimed_during_restore,
        );
        let persistent_prompt_cache_token_count =
            u32::try_from(hit_finalization.restored_token_count).map_err(|_| {
                super::fatal_engine_error(
                    "persistent prompt-cache token count exceeds the u32 range",
                )
            })?;
        self.persistent_prompt_cache_counters
            .record_cache_hit(hit_finalization.restored_token_count);
        if hit_finalization.restored_partial_tail_block_key.is_some() {
            self.persistent_prompt_cache_counters
                .record_partial_tail_hit();
        }
        Ok(PersistentPromptCacheRestoreOutcome {
            persistent_prompt_cache_token_count,
            restored_token_count: hit_finalization.restored_token_count,
            last_restored_persistent_prompt_cache_block_key: hit_finalization
                .last_restored_persistent_prompt_cache_block_key,
            persistent_prompt_cache_diagnostics: hit_finalization
                .persistent_prompt_cache_diagnostics,
        })
    }

    /// Loads the enumerated prefix, reconstructs the concatenated full-attention
    /// state, absorbs the newest recurrent snapshot, materializes everything on
    /// the GPU, and releases the transient buffers.
    #[allow(clippy::too_many_arguments)]
    fn load_and_reconstruct_restored_prefix(
        &self,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        restored_persistent_prompt_cache_block_keys: Vec<PersistentPromptCacheBlockKey>,
        restored_token_count: usize,
        complete_block_count: usize,
        restored_partial_tail_block_key: Option<&PersistentPromptCacheBlockKey>,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<PersistentPromptCacheBlockKey>, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(
                super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error,
            )?;
        let restored_kv_block_load = restore_support::load_restored_kv_blocks(
            model.runtime(),
            persistent_prompt_cache,
            restored_persistent_prompt_cache_block_keys,
            complete_block_count,
            performance_attribution,
        )?;
        let last_restored_persistent_prompt_cache_block_key =
            restored_kv_block_load.last_restored_persistent_prompt_cache_block_key;
        restore_support::restore_concatenated_full_attention_state(
            model.runtime(),
            request_decoder_state,
            restored_kv_block_load.restored_persistent_prompt_cache_kv_block_tensors,
            restored_token_count,
            complete_block_count,
            performance_attribution,
        )?;
        // Recurrent state needs the newest boundary at the restored end, which
        // is the tail's snapshot when a tail was restored and otherwise the
        // last complete block's snapshot.
        let recurrent_snapshot_block_key = restored_partial_tail_block_key
            .or(last_restored_persistent_prompt_cache_block_key.as_ref())
            .ok_or_else(|| {
                super::fatal_engine_error("persistent prompt-cache restore lost snapshot key")
            })?;
        restore_support::absorb_newest_recurrent_snapshot(
            model.runtime(),
            persistent_prompt_cache,
            request_decoder_state,
            recurrent_snapshot_block_key,
            complete_block_count,
            performance_attribution,
        )?;
        restore_support::materialize_and_release_restored_state(
            model.runtime(),
            request_decoder_state,
            complete_block_count,
            performance_attribution,
        )?;
        Ok(last_restored_persistent_prompt_cache_block_key)
    }

    /// Matches the prompt's chain against the retained live state and, when the
    /// retained state wins, installs it as the restore outcome directly.
    #[allow(clippy::too_many_arguments)]
    fn try_adopt_retained_live_prefix(
        &mut self,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        prompt_token_ids: &[u32],
        block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
        disk_restored_token_count: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<InMemoryPrefixAdoptionResolution, InferenceEngineError> {
        if !self.chunking.experimental_in_memory_prefix_adoption_enabled {
            return Ok(InMemoryPrefixAdoptionResolution::NotAdopted(None));
        }
        if self.retained_live_prefix.is_none() {
            return Ok(InMemoryPrefixAdoptionResolution::NotAdopted(Some(
                WorkerInMemoryPrefixAdoptionOutcome::NoRetainedState,
            )));
        }
        let retained_live_prefix = match self.retained_live_prefix.take() {
            Some(retained_live_prefix) => retained_live_prefix,
            None => unreachable!("the holder was just proven present"),
        };
        let retained_lookup = PersistentPromptCachePrefixLookup::for_retained_complete_block_keys(
            &persistent_prompt_cache.model_contract,
            prompt_token_ids,
            block_causal_inputs,
            &retained_live_prefix.complete_block_keys,
            |block_hash| persistent_prompt_cache.has_recurrent_snapshot(block_hash),
        );
        Ok(
            match super::in_memory_adoption_decision::decide_in_memory_prefix_adoption(
                retained_lookup.restored_token_count(),
                disk_restored_token_count,
            ) {
                super::in_memory_adoption_decision::InMemoryPrefixAdoptionDecision::AdoptRetainedState => {
                    let adopted_restore_outcome = self.install_adopted_live_prefix(
                        persistent_prompt_cache,
                        retained_live_prefix,
                        retained_lookup,
                        request_decoder_state,
                        performance_attribution,
                    )?;
                    InMemoryPrefixAdoptionResolution::Adopted(Box::new(adopted_restore_outcome))
                }
                // The retained state lost to the disk store or diverged from
                // this prompt's chain: drop it. Its arrays return to the
                // allocator cache here rather than serving a conversation
                // they no longer match.
                super::in_memory_adoption_decision::InMemoryPrefixAdoptionDecision::ChainMismatch => {
                    InMemoryPrefixAdoptionResolution::NotAdopted(Some(
                        WorkerInMemoryPrefixAdoptionOutcome::ChainMismatch,
                    ))
                }
                super::in_memory_adoption_decision::InMemoryPrefixAdoptionDecision::RetainedStateShorter => {
                    InMemoryPrefixAdoptionResolution::NotAdopted(Some(
                        WorkerInMemoryPrefixAdoptionOutcome::RetainedStateShorter,
                    ))
                }
            },
        )
    }
}
