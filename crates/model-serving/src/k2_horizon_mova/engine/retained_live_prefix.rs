//! The live KV state retained between requests for prefix adoption.
//!
//! In-memory prefix adoption (issue #1167) keeps the just-completed
//! generation's KV caches alive so the next turn of the same conversation
//! adopts them instead of re-reading the prefix blocks from the disk store.
//! The retention decision lives in the memory package; this holder is the
//! typed state the engine retains and installs. K2 Horizon MoVA is all
//! full-attention KV with no recurrent components, so adoption is a pure
//! offset rollback — no boundary snapshots are involved.

use crate::InferenceEngineError;
use crate::k2_horizon_mova::model::K2HorizonMoVAKvState;
use crate::memory::{RetainedPrefixRetentionDecision, RetainedPrefixRetentionRequirements};
use crate::persistent_cache::{
    PersistentPromptCacheBlockKey, PersistentPromptCacheCounters,
    PersistentPromptCacheModelContract, PersistentPromptCachePrefixLookupResult,
};

use super::K2HorizonMoVAActiveGeneration;

/// The retained live KV of the last completed generation.
pub(in crate::k2_horizon_mova) struct RetainedLiveKvPrefix {
    /// One KV state per decoder layer at generation end. Their offsets extend
    /// past the adoptable boundary (generated tokens); adoption truncates
    /// them to the boundary and the stale tail is overwritten by the next
    /// prefill's in-place slab writes.
    pub(super) caches: Vec<K2HorizonMoVAKvState>,
    /// The complete prompt-cache block keys the generation's capture chain
    /// covers, in chain order.
    pub(super) complete_block_keys: Vec<PersistentPromptCacheBlockKey>,
}

/// Extracts the completed generation's live KV for adoption by the next
/// request, or `None` when the retention policy drops it.
///
/// The adoptable boundary is the smaller of the prompt's complete published
/// block chain and the verified KV offset — a cancelled or failed generation
/// must not retain uninitialized slab regions behind its processed offset.
pub(super) fn take_retained_live_prefix(
    completed_generation: K2HorizonMoVAActiveGeneration,
    model_contract: &PersistentPromptCacheModelContract,
    adoption_enabled: bool,
    active_memory_limit_bytes: usize,
) -> Option<RetainedLiveKvPrefix> {
    let block_token_count = model_contract.block_token_count();
    if block_token_count == 0 {
        return None;
    }
    let verified_offset_tokens = completed_generation
        .caches
        .iter()
        .map(K2HorizonMoVAKvState::offset_tokens)
        .min()?;
    let verified_offset_count = usize::try_from(verified_offset_tokens.max(0)).unwrap_or(0);
    let prompt_complete_block_count =
        completed_generation.prompt_token_ids.len() / block_token_count;
    let adoptable_block_count =
        (verified_offset_count / block_token_count).min(prompt_complete_block_count);
    let adoptable_token_count = adoptable_block_count * block_token_count;
    let retained_state_payload_bytes = completed_generation
        .caches
        .iter()
        .map(K2HorizonMoVAKvState::payload_byte_count)
        .fold(0_u64, u64::saturating_add);
    let retention_requirements = RetainedPrefixRetentionRequirements {
        adoption_enabled,
        adoptable_token_count,
        retained_state_payload_bytes,
        effective_mlx_memory_ceiling_bytes: u64::try_from(active_memory_limit_bytes)
            .unwrap_or(u64::MAX),
    };
    if retention_requirements.decide() != RetainedPrefixRetentionDecision::Retain {
        return None;
    }
    let mut complete_block_keys = Vec::with_capacity(adoptable_block_count);
    let mut parent_block_key: Option<PersistentPromptCacheBlockKey> = None;
    for block_index in 0..adoptable_block_count {
        let block_start = block_index * block_token_count;
        let block_end = block_start + block_token_count;
        let block_key = match parent_block_key.as_ref() {
            Some(parent_block_key) => parent_block_key
                .for_child_block(&completed_generation.prompt_token_ids[block_start..block_end]),
            None => PersistentPromptCacheBlockKey::for_root_block(
                model_contract,
                &completed_generation.prompt_token_ids[block_start..block_end],
            ),
        }
        .ok()?;
        parent_block_key = Some(block_key.clone());
        complete_block_keys.push(block_key);
    }
    Some(RetainedLiveKvPrefix {
        caches: completed_generation.caches,
        complete_block_keys,
    })
}

/// Installs an adopting retained prefix: truncates every layer to the
/// adopted boundary (the stale KV tail is overwritten by the next prefill's
/// in-place slab writes) and records the adoption counters. Returns the
/// adopted token count for the request's cached-prefix bookkeeping.
pub(super) fn install_retained_live_prefix(
    retained_live_prefix: RetainedLiveKvPrefix,
    retained_lookup: &PersistentPromptCachePrefixLookupResult,
    caches: &mut Vec<K2HorizonMoVAKvState>,
    last_published_block_key: &mut Option<PersistentPromptCacheBlockKey>,
    persistent_prompt_cache_counters: &mut PersistentPromptCacheCounters,
) -> Result<u32, InferenceEngineError> {
    let adopted_token_count = retained_lookup.restored_token_count();
    let adopted_token_count_i32 =
        i32::try_from(adopted_token_count).map_err(|_| InferenceEngineError::Fatal {
            reason: "K2 Horizon MoVA adopted prefix token count overflowed".to_owned(),
        })?;
    *caches = retained_live_prefix.caches;
    for cache in caches.iter_mut() {
        cache
            .truncate_to_offset(adopted_token_count_i32)
            .map_err(|truncate_error| InferenceEngineError::Fatal {
                reason: format!("K2 Horizon MoVA adopted KV state is invalid: {truncate_error}"),
            })?;
    }
    *last_published_block_key = retained_lookup
        .last_restored_persistent_prompt_cache_block_key()
        .cloned();
    persistent_prompt_cache_counters.record_cache_hit(adopted_token_count);
    persistent_prompt_cache_counters.record_in_memory_prefix_adoption_hit();
    Ok(u32::try_from(adopted_token_count).unwrap_or(u32::MAX))
}
