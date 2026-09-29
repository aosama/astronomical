//! Brute-force probe for one restorable partial tail block on top of a matched prefix.
//!
//! Partial tails are stored off the complete-block chain: they hold fewer
//! tokens than one full block and are never chain parents. The store does not
//! index them by parent, so lookup reconstructs each candidate key from the
//! prompt itself and asks the store whether that exact key exists. A hash hit
//! proves both the token span and the chain position, because block hashes
//! cover the parent digest, the causal input, and every tail token.

use super::block_causal_input::PersistentPromptCacheBlockCausalInput;
use super::block_key::PersistentPromptCacheBlockKey;
use super::model_contract::PersistentPromptCacheModelContract;

/// Probes for a restorable partial tail block continuing one matched prefix.
///
/// `restored_complete_block_keys` holds the matched complete-block chain in
/// prompt order. An empty chain probes a root tail keyed from the root seed;
/// otherwise the tail continues the newest matched block. The probe walks
/// candidate token counts longest-first and returns the first key whose
/// sequence-state and recurrent-snapshot files both exist, so the longest
/// stored tail under one parent wins.
///
/// `tail_block_slot_index` is the prompt block slot the tail occupies; it
/// selects the model-owned causal input for that slot. A missing causal input
/// disables the probe (fail-soft, matching complete-block lookup).
pub(crate) fn probe_restorable_partial_tail_block(
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    prompt_tokens: &[u32],
    restored_complete_block_keys: &[PersistentPromptCacheBlockKey],
    newest_snapshot_is_chain_tip: bool,
    allow_exact_block_boundary_restore: bool,
    block_causal_inputs: Option<&[PersistentPromptCacheBlockCausalInput]>,
    persistent_prompt_cache_kv_block_exists: impl Fn(&[u8; 32]) -> bool,
    persistent_prompt_cache_recurrent_snapshot_exists: impl Fn(&[u8; 32]) -> bool,
) -> Option<PersistentPromptCacheBlockKey> {
    if !persistent_prompt_cache_model_contract.has_sequence_state() || prompt_tokens.is_empty() {
        return None;
    }
    let persistent_prompt_cache_block_token_count =
        persistent_prompt_cache_model_contract.block_token_count();
    let tail_start_tokens =
        restored_complete_block_keys.len() * persistent_prompt_cache_block_token_count;
    let tail_token_count = prompt_tokens.len().checked_sub(tail_start_tokens)?;
    if tail_token_count == 0 {
        return None;
    }
    // Generation startup must keep at least one prompt token for the forward
    // pass that produces the first logits; a complete-prefix consumer may
    // consume the whole tail. A tail is also strictly partial, so the longest
    // candidate stays under one full block.
    let retained_suffix_token_count = if allow_exact_block_boundary_restore {
        0
    } else {
        1
    };
    let maximum_tail_token_count = tail_token_count
        .saturating_sub(retained_suffix_token_count)
        .min(persistent_prompt_cache_block_token_count - 1);
    if maximum_tail_token_count == 0 {
        return None;
    }
    if !restored_complete_block_keys.is_empty() {
        // A tail is restorable only when the newest recurrent snapshot sits on
        // the chain tip: otherwise the restored state cannot reach the tail's
        // parent. And when the uncached suffix still contains complete blocks,
        // the final partial segment chains from a block this lookup did not
        // reach, so no tail under the matched tip can exist.
        if tail_token_count >= persistent_prompt_cache_block_token_count
            || !newest_snapshot_is_chain_tip
        {
            return None;
        }
    }
    let tail_block_slot_index = restored_complete_block_keys.len();
    let empty_block_causal_input = PersistentPromptCacheBlockCausalInput::empty();
    let tail_block_causal_input = match block_causal_inputs {
        None => &empty_block_causal_input,
        Some(block_causal_inputs) => block_causal_inputs.get(tail_block_slot_index)?,
    };
    let parent_persistent_prompt_cache_block_key = restored_complete_block_keys.last();
    for tail_candidate_token_count in (1..=maximum_tail_token_count).rev() {
        let tail_tokens =
            &prompt_tokens[tail_start_tokens..tail_start_tokens + tail_candidate_token_count];
        let tail_block_key = match parent_persistent_prompt_cache_block_key {
            None => PersistentPromptCacheBlockKey::for_root_block_with_causal_input(
                persistent_prompt_cache_model_contract,
                tail_tokens,
                tail_block_causal_input,
            ),
            Some(parent_persistent_prompt_cache_block_key) => {
                parent_persistent_prompt_cache_block_key
                    .for_child_block_with_causal_input(tail_tokens, tail_block_causal_input)
            }
        };
        let Ok(tail_block_key) = tail_block_key else {
            continue;
        };
        let tail_block_hash = tail_block_key.block_hash();
        if persistent_prompt_cache_kv_block_exists(&tail_block_hash)
            && persistent_prompt_cache_recurrent_snapshot_exists(&tail_block_hash)
        {
            return Some(tail_block_key);
        }
    }
    None
}
