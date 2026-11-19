//! Restores the longest complete persistent astro SSD prompt cache prefix into live Qwen state.
//!
//! Lookup is allocation-free. After a complete chain is identified, all sequence
//! blocks and the newest required boundary are loaded, reconstructed, synchronized,
//! and released before memory admission is repeated for only the remaining context.
//!
//! The lookup diagnostics, block-key derivation, prefix enumeration, and the
//! load-and-reconstruct pipeline are shared with the resident engine in
//! `crate::astro_ssd_prompt_cache::restore_support`; this module keeps the streaming
//! engine's memory-admission and expert-retention policy.

use astronomical_ipc_protocol::{RequestId, WorkerAstroSsdPromptCacheRequestDiagnostics};

use crate::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;
use crate::astro_ssd_prompt_cache::prefix_lookup::AstroSsdPromptCachePrefixLookup;
use crate::astro_ssd_prompt_cache::restore_support;
use crate::{
    AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheDiskStore, InferenceEngineError,
    PerformanceAttribution, PerformanceOperation,
};

use super::Qwen3_5EngineState;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;

/// The `astro_ssd_prompt_cache`-specific portion of a newly admitted request's starting state.
///
/// `restored_token_count` drives the prefill cursor, while the u32 field is
/// reported in the public generation usage contract. Keeping the final block
/// key lets later cold-prefilled blocks extend the same content-addressed
/// chain instead of creating an unrelated root block.
pub(super) struct AstroSsdPromptCacheRestoreOutcome {
    pub(super) astro_ssd_prompt_cache_restored_token_count: u32,
    pub(super) restored_token_count: usize,
    pub(super) last_restored_astro_ssd_prompt_cache_block_key: Option<AstroSsdPromptCacheBlockKey>,
    pub(super) astro_ssd_prompt_cache_diagnostics: WorkerAstroSsdPromptCacheRequestDiagnostics,
}

/// The inputs of a restore-hit finalization: everything the counter stamping,
/// re-admission, hit logging, and outcome construction need.
struct RestoreHitFinalization<'a> {
    request_id: &'a RequestId,
    prompt_token_count: usize,
    total_context_tokens: usize,
    restored_token_count: usize,
    complete_block_count: usize,
    restored_partial_tail_block_key: Option<&'a AstroSsdPromptCacheBlockKey>,
    lookup_diagnostics: &'a crate::AstroSsdPromptCacheLookupDiagnostics,
    target_expert_payload_bytes_reclaimed_before_restore: u64,
    astro_ssd_prompt_cache_diagnostics: WorkerAstroSsdPromptCacheRequestDiagnostics,
    last_restored_astro_ssd_prompt_cache_block_key: Option<AstroSsdPromptCacheBlockKey>,
}

impl Qwen3_5EngineState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_astro_ssd_prompt_cache_prefix(
        &mut self,
        request_id: RequestId,
        astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
        prompt_token_ids: &[u32],
        block_causal_inputs: &[AstroSsdPromptCacheBlockCausalInput],
        total_context_tokens: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<AstroSsdPromptCacheRestoreOutcome, InferenceEngineError> {
        // First decide the longest usable prefix using only the store's small
        // in-memory index. No MLX array is allocated until a whole contiguous
        // prefix is known to be available.
        let lookup_result = performance_attribution.measure_operation(
            PerformanceOperation::AstroSsdPromptCachePrefixLookup,
            |_performance_attribution| {
                AstroSsdPromptCachePrefixLookup::for_prompt_with_partial_tail_block_recovery(
                    &astro_ssd_prompt_cache.model_contract,
                    prompt_token_ids,
                    block_causal_inputs,
                    |block_hash| astro_ssd_prompt_cache.has_kv_block(block_hash),
                    |block_hash| astro_ssd_prompt_cache.has_recurrent_snapshot(block_hash),
                )
            },
        );
        let restored_token_count = lookup_result.restored_token_count();
        let restored_partial_tail_block_key =
            lookup_result.restored_partial_tail_block_key().cloned();
        let lookup_diagnostics = lookup_result.diagnostics();
        let astro_ssd_prompt_cache_diagnostics =
            restore_support::astro_ssd_prompt_cache_request_diagnostics(
                astro_ssd_prompt_cache.model_contract.block_token_count(),
                lookup_diagnostics,
                restored_token_count,
                lookup_diagnostics.restored_partial_tail_block_token_count(),
            );
        if restored_token_count == 0 {
            // A miss is normal, not an error. The cold outcome keeps the
            // caller's fallback path identical for short prompts, first
            // requests, and a directory with no matching prefix.
            return Ok(self.cold_restore_miss_outcome(
                &request_id,
                prompt_token_ids.len(),
                astro_ssd_prompt_cache,
                lookup_diagnostics,
                astro_ssd_prompt_cache_diagnostics,
            ));
        }

        let complete_block_count =
            restored_token_count / astro_ssd_prompt_cache.model_contract.block_token_count();
        let prefix_enumeration = restore_support::enumerate_restored_prefix_blocks(
            astro_ssd_prompt_cache,
            prompt_token_ids,
            block_causal_inputs,
            restored_token_count,
            restored_partial_tail_block_key.as_ref(),
        )?;
        let astro_ssd_prompt_cache_restore_temporary_workspace_bytes = usize::try_from(
            prefix_enumeration
                .total_sequence_block_file_size_bytes
                .max(prefix_enumeration.recurrent_snapshot_file_size_bytes),
        )
        .unwrap_or(usize::MAX);
        let target_expert_payload_bytes_reclaimed_before_restore = self
            .lease_astro_ssd_prompt_cache_restore_workspace(
                astro_ssd_prompt_cache_restore_temporary_workspace_bytes,
            )?;
        let last_restored_astro_ssd_prompt_cache_block_key = self
            .load_and_reconstruct_restored_prefix(
                astro_ssd_prompt_cache,
                prefix_enumeration.restored_astro_ssd_prompt_cache_block_keys,
                restored_token_count,
                complete_block_count,
                restored_partial_tail_block_key.as_ref(),
                request_decoder_state,
                performance_attribution,
            )?;
        self.finalize_restored_prefix_hit(RestoreHitFinalization {
            request_id: &request_id,
            prompt_token_count: prompt_token_ids.len(),
            total_context_tokens,
            restored_token_count,
            complete_block_count,
            restored_partial_tail_block_key: restored_partial_tail_block_key.as_ref(),
            lookup_diagnostics,
            target_expert_payload_bytes_reclaimed_before_restore,
            astro_ssd_prompt_cache_diagnostics,
            last_restored_astro_ssd_prompt_cache_block_key,
        })
    }

    /// Loads the enumerated prefix, reconstructs the concatenated full-attention
    /// state, absorbs the newest recurrent snapshot, materializes everything on
    /// the GPU, releases the transient buffers, and resumes expert retention.
    #[allow(clippy::too_many_arguments)]
    fn load_and_reconstruct_restored_prefix(
        &self,
        astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
        restored_astro_ssd_prompt_cache_block_keys: Vec<AstroSsdPromptCacheBlockKey>,
        restored_token_count: usize,
        complete_block_count: usize,
        restored_partial_tail_block_key: Option<&AstroSsdPromptCacheBlockKey>,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<AstroSsdPromptCacheBlockKey>, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let restored_kv_block_load = restore_support::load_restored_kv_blocks(
            model.runtime(),
            astro_ssd_prompt_cache,
            restored_astro_ssd_prompt_cache_block_keys,
            complete_block_count,
            performance_attribution,
        )?;
        let last_restored_astro_ssd_prompt_cache_block_key =
            restored_kv_block_load.last_restored_astro_ssd_prompt_cache_block_key;
        restore_support::restore_concatenated_full_attention_state(
            model.runtime(),
            request_decoder_state,
            restored_kv_block_load.restored_astro_ssd_prompt_cache_kv_block_tensors,
            restored_token_count,
            complete_block_count,
            performance_attribution,
        )?;
        // Recurrent state needs the newest boundary at the restored end, which
        // is the tail's snapshot when a tail was restored and otherwise the
        // last complete block's snapshot.
        let recurrent_snapshot_block_key = restored_partial_tail_block_key
            .or(last_restored_astro_ssd_prompt_cache_block_key.as_ref())
            .ok_or_else(|| {
                super::fatal_engine_error(
                    "persistent astro-ssd-prompt-cache restore lost snapshot key",
                )
            })?;
        restore_support::absorb_newest_recurrent_snapshot(
            model.runtime(),
            astro_ssd_prompt_cache,
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
        Ok(last_restored_astro_ssd_prompt_cache_block_key)
    }

    /// Cold-path miss outcome: attaches startup-cleanup evidence when the miss
    /// reason is one a startup sweep cures, logs the miss, records the miss
    /// counter, and returns the explicit cold state.
    fn cold_restore_miss_outcome(
        &mut self,
        request_id: &RequestId,
        prompt_token_count: usize,
        astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
        lookup_diagnostics: &crate::AstroSsdPromptCacheLookupDiagnostics,
        mut astro_ssd_prompt_cache_diagnostics: WorkerAstroSsdPromptCacheRequestDiagnostics,
    ) -> AstroSsdPromptCacheRestoreOutcome {
        if restore_support::miss_reason_warrants_startup_cleanup_evidence(lookup_diagnostics) {
            astro_ssd_prompt_cache_diagnostics.startup_cleanup_evidence = astro_ssd_prompt_cache
                .take_startup_cleanup_evidence()
                .map(restore_support::worker_startup_cleanup_evidence);
        }
        restore_support::log_astro_ssd_prompt_cache_restore_miss(
            request_id,
            prompt_token_count,
            lookup_diagnostics,
            astro_ssd_prompt_cache_diagnostics
                .startup_cleanup_evidence
                .as_ref(),
        );
        self.astro_ssd_prompt_cache_counters
            .record_astro_ssd_prompt_cache_miss();
        AstroSsdPromptCacheRestoreOutcome {
            astro_ssd_prompt_cache_restored_token_count: 0,
            restored_token_count: 0,
            last_restored_astro_ssd_prompt_cache_block_key: None,
            astro_ssd_prompt_cache_diagnostics,
        }
    }

    /// Hit finalization: gives the restore's leased room back now that the
    /// transient load memory is gone, records what the kept weights gave
    /// up for it, logs the hit, stamps the counters, and builds the outcome.
    fn finalize_restored_prefix_hit(
        &mut self,
        hit_finalization: RestoreHitFinalization<'_>,
    ) -> Result<AstroSsdPromptCacheRestoreOutcome, InferenceEngineError> {
        let mut hit_finalization = hit_finalization;
        if hit_finalization.restored_token_count > hit_finalization.total_context_tokens {
            return Err(super::fatal_engine_error(
                "persistent astro-ssd-prompt-cache restore exceeded the generation context",
            ));
        }
        self.give_back_astro_ssd_prompt_cache_restore_workspace();
        let target_expert_payload_bytes_reclaimed_during_restore =
            hit_finalization.target_expert_payload_bytes_reclaimed_before_restore;
        hit_finalization
            .astro_ssd_prompt_cache_diagnostics
            .expert_bytes_reclaimed_for_restore =
            target_expert_payload_bytes_reclaimed_during_restore;
        restore_support::log_astro_ssd_prompt_cache_restore_hit(
            hit_finalization.request_id,
            hit_finalization.prompt_token_count,
            hit_finalization.restored_token_count,
            hit_finalization.complete_block_count,
            hit_finalization.restored_partial_tail_block_key,
            hit_finalization.lookup_diagnostics,
            target_expert_payload_bytes_reclaimed_during_restore,
        );
        let astro_ssd_prompt_cache_restored_token_count =
            u32::try_from(hit_finalization.restored_token_count).map_err(|_| {
                super::fatal_engine_error(
                    "persistent astro-ssd-prompt-cache token count exceeds the u32 range",
                )
            })?;
        self.astro_ssd_prompt_cache_counters
            .record_astro_ssd_prompt_cache_hit(hit_finalization.restored_token_count);
        if hit_finalization.restored_partial_tail_block_key.is_some() {
            self.astro_ssd_prompt_cache_counters
                .record_partial_tail_hit();
        }
        Ok(AstroSsdPromptCacheRestoreOutcome {
            astro_ssd_prompt_cache_restored_token_count,
            restored_token_count: hit_finalization.restored_token_count,
            last_restored_astro_ssd_prompt_cache_block_key: hit_finalization
                .last_restored_astro_ssd_prompt_cache_block_key,
            astro_ssd_prompt_cache_diagnostics: hit_finalization.astro_ssd_prompt_cache_diagnostics,
        })
    }
}

impl Qwen3_5EngineState {
    /// Leases room from the memory governor for the blocks a restore loads
    /// beside the stored state they become, and returns the kept weight
    /// bytes given back for it. A restore that fails leaves the lease to
    /// the end of the request, which gives it back.
    fn lease_astro_ssd_prompt_cache_restore_workspace(
        &self,
        restore_workspace_bytes: usize,
    ) -> Result<u64, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        Ok(model
            .govern_restore_workspace(u64::try_from(restore_workspace_bytes).unwrap_or(u64::MAX)))
    }

    /// The restored blocks have joined the stored state: their room goes
    /// back to the kept weights.
    fn give_back_astro_ssd_prompt_cache_restore_workspace(&self) {
        if let Some(model) = self.model.as_ref() {
            model.govern_restore_workspace_end();
        }
    }
}
