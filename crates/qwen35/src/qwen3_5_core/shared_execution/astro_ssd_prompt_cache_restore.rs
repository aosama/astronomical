//! Restores the longest complete persistent `astro_ssd_prompt_cache` prefix into
//! live Qwen3.5 state, for either engine.
//!
//! Lookup is allocation-free. After a complete chain is identified, all sequence
//! blocks and the newest required boundary are loaded, reconstructed, synchronized,
//! and released before the rest of the prompt is read.
//!
//! The steps are the same in both engines. Where they differ, an engine
//! supplies a hook through [`Qwen3_5AstroSsdPromptCacheRestoreHooks`]: the
//! resident engine may adopt the live prefix it kept from the last reply, and
//! the streaming engine leases room from its memory governor for the blocks a
//! restore loads.

use ipc_protocol::{
    RequestId, WorkerAstroSsdPromptCacheRequestDiagnostics, WorkerInMemoryPrefixAdoptionOutcome,
};

use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::engine_errors::fatal_engine_error;
use crate::{
    AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheDiskStore,
    AstroSsdPromptCacheLookupDiagnostics, InferenceEngineError, PerformanceAttribution,
    PerformanceOperation,
};
use engine_core::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;
use engine_core::astro_ssd_prompt_cache::prefix_lookup::AstroSsdPromptCachePrefixLookup;
use engine_core::astro_ssd_prompt_cache::restore_support;

use super::astro_ssd_prompt_cache_state_install;
use super::execution_core::Qwen3_5ExecutionCoreOwner;
use super::restore_transient_workspace::restore_transient_workspace_bytes;

/// The `astro_ssd_prompt_cache`-specific portion of a newly admitted request's starting state.
///
/// `restored_token_count` drives the prefill cursor, while the u32 field is
/// reported in the public generation usage contract. Keeping the final block
/// key lets later cold-prefilled blocks extend the same content-addressed
/// chain instead of creating an unrelated root block.
pub struct AstroSsdPromptCacheRestoreOutcome {
    pub astro_ssd_prompt_cache_restored_token_count: u32,
    pub restored_token_count: usize,
    pub last_restored_astro_ssd_prompt_cache_block_key: Option<AstroSsdPromptCacheBlockKey>,
    pub astro_ssd_prompt_cache_diagnostics: WorkerAstroSsdPromptCacheRequestDiagnostics,
}

/// How an engine's live-prefix adoption resolved: the retained live prefix
/// took over the restore, or the disk path continues with an adoption outcome
/// to report. The adopted outcome is boxed to keep the cold variant from
/// inflating every resolution.
pub enum LivePrefixAdoption {
    Adopted(Box<AstroSsdPromptCacheRestoreOutcome>),
    NotAdopted(Option<WorkerInMemoryPrefixAdoptionOutcome>),
}

/// What the request asks the `astro_ssd_prompt_cache` to restore.
pub struct AstroSsdPromptCacheRestoreRequest<'a> {
    pub request_id: RequestId,
    pub astro_ssd_prompt_cache: &'a AstroSsdPromptCacheDiskStore,
    pub prompt_token_ids: &'a [u32],
    pub block_causal_inputs: &'a [AstroSsdPromptCacheBlockCausalInput],
    pub total_context_tokens: usize,
}

/// The points where an engine's restore differs. Each default does nothing,
/// so an engine overrides only what it has.
pub trait Qwen3_5AstroSsdPromptCacheRestoreHooks: Qwen3_5ExecutionCoreOwner {
    /// Tries the live prefix kept from the last reply before the disk store.
    /// `disk_restored_token_count` is what the disk store alone would restore.
    fn adopt_retained_live_prefix(
        &mut self,
        _restore_request: &AstroSsdPromptCacheRestoreRequest<'_>,
        _disk_restored_token_count: usize,
        _request_decoder_state: &mut RequestDecoderStateStack,
        _performance_attribution: &mut PerformanceAttribution,
    ) -> Result<LivePrefixAdoption, InferenceEngineError> {
        Ok(LivePrefixAdoption::NotAdopted(None))
    }

    /// Leases room for the blocks a restore loads and returns the kept
    /// weight bytes given back for it.
    fn lease_restore_workspace(
        &self,
        _restore_workspace_bytes: usize,
    ) -> Result<u64, InferenceEngineError> {
        Ok(0)
    }

    /// The restored blocks have joined the stored state.
    fn give_back_restore_workspace(&self) {}
}

/// Restores the longest prefix the `astro_ssd_prompt_cache` holds for this
/// prompt, or returns the cold outcome when it holds none.
pub fn restore_astro_ssd_prompt_cache_prefix<Engine: Qwen3_5AstroSsdPromptCacheRestoreHooks>(
    engine: &mut Engine,
    restore_request: &AstroSsdPromptCacheRestoreRequest<'_>,
    request_decoder_state: &mut RequestDecoderStateStack,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<AstroSsdPromptCacheRestoreOutcome, InferenceEngineError> {
    let astro_ssd_prompt_cache = restore_request.astro_ssd_prompt_cache;
    let prompt_token_ids = restore_request.prompt_token_ids;
    // First decide the longest usable prefix using only the store's small
    // in-memory index. No MLX array is allocated until a whole contiguous
    // prefix is known to be available.
    let lookup_result = performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCachePrefixLookup,
        |_performance_attribution| {
            AstroSsdPromptCachePrefixLookup::for_prompt_with_partial_tail_block_recovery(
                &astro_ssd_prompt_cache.model_contract,
                prompt_token_ids,
                restore_request.block_causal_inputs,
                |block_hash| astro_ssd_prompt_cache.has_kv_block(block_hash),
                |block_hash| astro_ssd_prompt_cache.has_recurrent_snapshot(block_hash),
            )
        },
    );
    let restored_token_count = lookup_result.restored_token_count();
    let restored_partial_tail_block_key = lookup_result.restored_partial_tail_block_key().cloned();
    let lookup_diagnostics = lookup_result.diagnostics();
    let mut astro_ssd_prompt_cache_diagnostics =
        restore_support::astro_ssd_prompt_cache_request_diagnostics(
            astro_ssd_prompt_cache.model_contract.block_token_count(),
            lookup_diagnostics,
            restored_token_count,
            lookup_diagnostics.restored_partial_tail_block_token_count(),
        );
    // An adopted live prefix can even rescue a disk miss, because the live
    // state may outlive blocks the store evicted.
    match engine.adopt_retained_live_prefix(
        restore_request,
        restored_token_count,
        request_decoder_state,
        performance_attribution,
    )? {
        LivePrefixAdoption::Adopted(adopted_restore_outcome) => {
            return Ok(*adopted_restore_outcome);
        }
        LivePrefixAdoption::NotAdopted(adoption_outcome) => {
            astro_ssd_prompt_cache_diagnostics.in_memory_prefix_adoption = adoption_outcome;
        }
    }
    if restored_token_count == 0 {
        // A miss is normal, not an error. The cold outcome keeps the caller's
        // fallback path identical for short prompts, first requests, and a
        // directory with no matching prefix.
        return Ok(cold_restore_miss_outcome(
            engine,
            restore_request,
            lookup_diagnostics,
            astro_ssd_prompt_cache_diagnostics,
        ));
    }
    let complete_block_count =
        restored_token_count / astro_ssd_prompt_cache.model_contract.block_token_count();
    let prefix_enumeration = restore_support::enumerate_restored_prefix_blocks(
        astro_ssd_prompt_cache,
        prompt_token_ids,
        restore_request.block_causal_inputs,
        restored_token_count,
        restored_partial_tail_block_key.as_ref(),
    )?;
    let target_expert_payload_bytes_reclaimed_before_restore =
        engine.lease_restore_workspace(restore_transient_workspace_bytes(
            prefix_enumeration.total_sequence_block_file_size_bytes,
            prefix_enumeration.recurrent_snapshot_file_size_bytes,
        ))?;
    let restored_span = RestoredSpan {
        restored_token_count,
        complete_block_count,
        restored_partial_tail_block_key: restored_partial_tail_block_key.as_ref(),
    };
    let last_restored_astro_ssd_prompt_cache_block_key = load_and_reconstruct_restored_prefix(
        engine,
        astro_ssd_prompt_cache,
        prefix_enumeration.restored_astro_ssd_prompt_cache_block_keys,
        restored_span,
        request_decoder_state,
        performance_attribution,
    )?;
    finalize_restored_prefix_hit(
        engine,
        restore_request,
        RestoreHit {
            restored_span,
            lookup_diagnostics,
            target_expert_payload_bytes_reclaimed_before_restore,
            astro_ssd_prompt_cache_diagnostics,
            last_restored_astro_ssd_prompt_cache_block_key,
        },
    )
}

/// The span of the prompt a restore covers.
#[derive(Clone, Copy)]
struct RestoredSpan<'a> {
    restored_token_count: usize,
    complete_block_count: usize,
    restored_partial_tail_block_key: Option<&'a AstroSsdPromptCacheBlockKey>,
}

/// Everything a restore hit's finalization needs beyond the request.
struct RestoreHit<'a> {
    restored_span: RestoredSpan<'a>,
    lookup_diagnostics: &'a AstroSsdPromptCacheLookupDiagnostics,
    target_expert_payload_bytes_reclaimed_before_restore: u64,
    astro_ssd_prompt_cache_diagnostics: WorkerAstroSsdPromptCacheRequestDiagnostics,
    last_restored_astro_ssd_prompt_cache_block_key: Option<AstroSsdPromptCacheBlockKey>,
}

/// Loads the enumerated prefix, reconstructs the concatenated full-attention
/// state, absorbs the newest recurrent snapshot, materializes everything on
/// the GPU, and releases the transient buffers.
fn load_and_reconstruct_restored_prefix<Engine: Qwen3_5ExecutionCoreOwner>(
    engine: &Engine,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    restored_astro_ssd_prompt_cache_block_keys: Vec<AstroSsdPromptCacheBlockKey>,
    restored_span: RestoredSpan<'_>,
    request_decoder_state: &mut RequestDecoderStateStack,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Option<AstroSsdPromptCacheBlockKey>, InferenceEngineError> {
    let model = engine.execution_core().loaded_model()?;
    let restored_kv_block_load = restore_support::load_restored_kv_blocks(
        model.runtime(),
        astro_ssd_prompt_cache,
        restored_astro_ssd_prompt_cache_block_keys,
        restored_span.complete_block_count,
        performance_attribution,
    )?;
    let last_restored_astro_ssd_prompt_cache_block_key =
        restored_kv_block_load.last_restored_astro_ssd_prompt_cache_block_key;
    astro_ssd_prompt_cache_state_install::restore_concatenated_full_attention_state(
        model.runtime(),
        request_decoder_state,
        restored_kv_block_load.restored_astro_ssd_prompt_cache_kv_block_tensors,
        restored_span.restored_token_count,
        restored_span.complete_block_count,
        performance_attribution,
    )?;
    // Recurrent state needs the newest boundary at the restored end, which
    // is the tail's snapshot when a tail was restored and otherwise the
    // last complete block's snapshot.
    let recurrent_snapshot_block_key = restored_span
        .restored_partial_tail_block_key
        .or(last_restored_astro_ssd_prompt_cache_block_key.as_ref())
        .ok_or_else(|| {
            fatal_engine_error("persistent astro-ssd-prompt-cache restore lost snapshot key")
        })?;
    astro_ssd_prompt_cache_state_install::absorb_newest_recurrent_snapshot(
        model.runtime(),
        astro_ssd_prompt_cache,
        request_decoder_state,
        recurrent_snapshot_block_key,
        restored_span.complete_block_count,
        performance_attribution,
    )?;
    astro_ssd_prompt_cache_state_install::materialize_and_release_restored_state(
        model.runtime(),
        request_decoder_state,
        restored_span.complete_block_count,
        performance_attribution,
    )?;
    Ok(last_restored_astro_ssd_prompt_cache_block_key)
}

/// Cold-path miss outcome: attaches startup-cleanup evidence when the miss
/// reason is one a startup sweep cures, logs the miss, records the miss
/// counter, and returns the explicit cold state.
fn cold_restore_miss_outcome<Engine: Qwen3_5ExecutionCoreOwner>(
    engine: &mut Engine,
    restore_request: &AstroSsdPromptCacheRestoreRequest<'_>,
    lookup_diagnostics: &AstroSsdPromptCacheLookupDiagnostics,
    mut astro_ssd_prompt_cache_diagnostics: WorkerAstroSsdPromptCacheRequestDiagnostics,
) -> AstroSsdPromptCacheRestoreOutcome {
    if restore_support::miss_reason_warrants_startup_cleanup_evidence(lookup_diagnostics) {
        astro_ssd_prompt_cache_diagnostics.startup_cleanup_evidence = restore_request
            .astro_ssd_prompt_cache
            .take_startup_cleanup_evidence()
            .map(restore_support::worker_startup_cleanup_evidence);
    }
    restore_support::log_astro_ssd_prompt_cache_restore_miss(
        &restore_request.request_id,
        restore_request.prompt_token_ids.len(),
        lookup_diagnostics,
        astro_ssd_prompt_cache_diagnostics
            .startup_cleanup_evidence
            .as_ref(),
    );
    engine
        .execution_core_mut()
        .astro_ssd_prompt_cache_counters
        .record_astro_ssd_prompt_cache_miss();
    AstroSsdPromptCacheRestoreOutcome {
        astro_ssd_prompt_cache_restored_token_count: 0,
        restored_token_count: 0,
        last_restored_astro_ssd_prompt_cache_block_key: None,
        astro_ssd_prompt_cache_diagnostics,
    }
}

/// Hit finalization: checks the restored span against the generation context,
/// gives the restore's leased room back now that the transient load memory is
/// gone, records what the kept weights gave up for it, logs the hit, stamps
/// the counters, and builds the outcome.
fn finalize_restored_prefix_hit<Engine: Qwen3_5AstroSsdPromptCacheRestoreHooks>(
    engine: &mut Engine,
    restore_request: &AstroSsdPromptCacheRestoreRequest<'_>,
    restore_hit: RestoreHit<'_>,
) -> Result<AstroSsdPromptCacheRestoreOutcome, InferenceEngineError> {
    let RestoreHit {
        restored_span,
        lookup_diagnostics,
        target_expert_payload_bytes_reclaimed_before_restore,
        mut astro_ssd_prompt_cache_diagnostics,
        last_restored_astro_ssd_prompt_cache_block_key,
    } = restore_hit;
    if restored_span.restored_token_count > restore_request.total_context_tokens {
        return Err(fatal_engine_error(
            "persistent astro-ssd-prompt-cache restore exceeded the generation context",
        ));
    }
    engine.give_back_restore_workspace();
    astro_ssd_prompt_cache_diagnostics.expert_bytes_reclaimed_for_restore =
        target_expert_payload_bytes_reclaimed_before_restore;
    restore_support::log_astro_ssd_prompt_cache_restore_hit(
        &restore_request.request_id,
        restore_request.prompt_token_ids.len(),
        restored_span.restored_token_count,
        restored_span.complete_block_count,
        restored_span.restored_partial_tail_block_key,
        lookup_diagnostics,
        target_expert_payload_bytes_reclaimed_before_restore,
    );
    let astro_ssd_prompt_cache_restored_token_count =
        u32::try_from(restored_span.restored_token_count).map_err(|_| {
            fatal_engine_error(
                "persistent astro-ssd-prompt-cache token count exceeds the u32 range",
            )
        })?;
    let astro_ssd_prompt_cache_counters =
        &mut engine.execution_core_mut().astro_ssd_prompt_cache_counters;
    astro_ssd_prompt_cache_counters
        .record_astro_ssd_prompt_cache_hit(restored_span.restored_token_count);
    if restored_span.restored_partial_tail_block_key.is_some() {
        astro_ssd_prompt_cache_counters.record_partial_tail_hit();
    }
    Ok(AstroSsdPromptCacheRestoreOutcome {
        astro_ssd_prompt_cache_restored_token_count,
        restored_token_count: restored_span.restored_token_count,
        last_restored_astro_ssd_prompt_cache_block_key,
        astro_ssd_prompt_cache_diagnostics,
    })
}
