//! The decomposition units of one persistent prompt-cache prefix lookup:
//! block counting, the short-prompt path, chain matching, the boundary scan,
//! miss-reason naming, and the restored-prefix assembly. Each unit is
//! file-private-grade (`pub(super)`) and receives the shared
//! `PersistentPromptCachePrefixLookupInputs` view instead of a long
//! parameter list.

use super::partial_tail_probe;
use super::prefix_lookup::{
    PersistentPromptCacheLookupDiagnostics, PersistentPromptCacheMissReason,
    PersistentPromptCachePrefixLookupResult, cache_miss_lookup_result,
};
use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::{PersistentPromptCacheBlockKey, PersistentPromptCacheModelContract};

/// The caller-supplied inputs of one prefix lookup. Every decomposition unit
/// receives this one coherent view instead of repeating the same long
/// parameter list of contract reference, prompt slice, policy switches, and
/// injected existence checks.
pub(super) struct PersistentPromptCachePrefixLookupInputs<
    'lookup,
    KvBlockExists,
    RecurrentSnapshotExists,
> where
    KvBlockExists: Fn(&[u8; 32]) -> bool,
    RecurrentSnapshotExists: Fn(&[u8; 32]) -> bool,
{
    pub(super) persistent_prompt_cache_model_contract: &'lookup PersistentPromptCacheModelContract,
    pub(super) prompt_tokens: &'lookup [u32],
    pub(super) block_causal_inputs: Option<&'lookup [PersistentPromptCacheBlockCausalInput]>,
    pub(super) allow_exact_block_boundary_restore: bool,
    pub(super) probe_partial_tail_blocks: bool,
    pub(super) persistent_prompt_cache_kv_block_exists: KvBlockExists,
    pub(super) persistent_prompt_cache_recurrent_snapshot_exists: RecurrentSnapshotExists,
}

/// Counts the complete prompt blocks and how many of them may be restored.
pub(super) fn complete_and_maximum_restorable_block_counts(
    prompt_token_count: usize,
    persistent_prompt_cache_block_token_count: usize,
    allow_exact_block_boundary_restore: bool,
) -> (usize, usize) {
    // The contract resolver guarantees a nonzero block length. Keeping the calculation here
    // contract-driven makes lookup agree with capture on models whose state geometry selects
    // a different boundary size than another artifact on the same machine.
    let complete_prompt_block_count =
        prompt_token_count / persistent_prompt_cache_block_token_count;
    // An exact block-boundary prompt must retain its final block for a
    // forward pass. The final prompt token produces the logits used to
    // begin decode, while restoring only recurrent state cannot substitute
    // for that model computation.
    let maximum_restorable_block_count = if allow_exact_block_boundary_restore {
        complete_prompt_block_count
    } else if prompt_token_count.is_multiple_of(persistent_prompt_cache_block_token_count) {
        complete_prompt_block_count.saturating_sub(1)
    } else {
        complete_prompt_block_count
    };
    (complete_prompt_block_count, maximum_restorable_block_count)
}

/// Produces the lookup outcome for a prompt too short for one restorable
/// block: a root partial tail when the probe finds one, otherwise a plain
/// cache miss.
pub(super) fn prompt_too_short_lookup_result<KvBlockExists, RecurrentSnapshotExists>(
    lookup_inputs: &PersistentPromptCachePrefixLookupInputs<
        '_,
        KvBlockExists,
        RecurrentSnapshotExists,
    >,
    mut lookup_diagnostics: PersistentPromptCacheLookupDiagnostics,
) -> PersistentPromptCachePrefixLookupResult
where
    KvBlockExists: Fn(&[u8; 32]) -> bool,
    RecurrentSnapshotExists: Fn(&[u8; 32]) -> bool,
{
    let persistent_prompt_cache_model_contract =
        lookup_inputs.persistent_prompt_cache_model_contract;
    let prompt_tokens = lookup_inputs.prompt_tokens;
    let block_causal_inputs = lookup_inputs.block_causal_inputs;
    let allow_exact_block_boundary_restore = lookup_inputs.allow_exact_block_boundary_restore;
    let probe_partial_tail_blocks = lookup_inputs.probe_partial_tail_blocks;
    if probe_partial_tail_blocks
        && let Some(restored_partial_tail_block_key) =
            partial_tail_probe::probe_restorable_partial_tail_block(
                persistent_prompt_cache_model_contract,
                prompt_tokens,
                &[],
                false,
                allow_exact_block_boundary_restore,
                block_causal_inputs,
                &lookup_inputs.persistent_prompt_cache_kv_block_exists,
                &lookup_inputs.persistent_prompt_cache_recurrent_snapshot_exists,
            )
    {
        // A prompt shorter than one complete block can still reuse a root
        // tail published by a previous turn with the same short prefix.
        let restored_token_count = restored_partial_tail_block_key.token_count();
        lookup_diagnostics.record_restored_partial_tail_block_token_count(restored_token_count);
        return PersistentPromptCachePrefixLookupResult {
            restored_token_count,
            remaining_tokens: prompt_tokens[restored_token_count..].to_vec(),
            last_restored_persistent_prompt_cache_block_key: None,
            restored_partial_tail_block_key: Some(restored_partial_tail_block_key),
            lookup_diagnostics,
        };
    }
    lookup_diagnostics.record_miss_reason(
        PersistentPromptCacheMissReason::PromptTooShortForPersistentPromptCache,
    );
    cache_miss_lookup_result(prompt_tokens, lookup_diagnostics)
}

/// Walks the complete prompt blocks in chain order and returns every block
/// whose key matched the store, stopping at the first gap.
pub(super) fn matched_sequence_state_block_keys<KvBlockExists, RecurrentSnapshotExists>(
    lookup_inputs: &PersistentPromptCachePrefixLookupInputs<
        '_,
        KvBlockExists,
        RecurrentSnapshotExists,
    >,
    maximum_restorable_block_count: usize,
    lookup_diagnostics: &mut PersistentPromptCacheLookupDiagnostics,
) -> Vec<PersistentPromptCacheBlockKey>
where
    KvBlockExists: Fn(&[u8; 32]) -> bool,
    RecurrentSnapshotExists: Fn(&[u8; 32]) -> bool,
{
    let persistent_prompt_cache_model_contract =
        lookup_inputs.persistent_prompt_cache_model_contract;
    let prompt_tokens = lookup_inputs.prompt_tokens;
    let block_causal_inputs = lookup_inputs.block_causal_inputs;
    let persistent_prompt_cache_kv_block_exists =
        &lookup_inputs.persistent_prompt_cache_kv_block_exists;
    let persistent_prompt_cache_block_token_count =
        persistent_prompt_cache_model_contract.block_token_count();
    // The key chain is constructed in prompt order. A hit for block N is
    // meaningful only if every predecessor also matched, which prevents
    // reuse across prompts that diverge in an earlier block.
    let mut matched_persistent_prompt_cache_block_keys =
        Vec::with_capacity(maximum_restorable_block_count);
    let mut parent_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey> = None;
    let empty_block_causal_input = PersistentPromptCacheBlockCausalInput::empty();
    for block_index in 0..maximum_restorable_block_count {
        let block_start = block_index * persistent_prompt_cache_block_token_count;
        let block_end = block_start + persistent_prompt_cache_block_token_count;
        let block_tokens = &prompt_tokens[block_start..block_end];
        let block_causal_input = match block_causal_inputs {
            None => &empty_block_causal_input,
            Some(block_causal_inputs) => {
                let Some(block_causal_input) = block_causal_inputs.get(block_index) else {
                    lookup_diagnostics
                        .record_first_missing_sequence_state_block_index(block_index, None);
                    break;
                };
                block_causal_input
            }
        };
        let persistent_prompt_cache_block_key =
            match parent_persistent_prompt_cache_block_key.as_ref() {
                None => PersistentPromptCacheBlockKey::for_root_block_with_causal_input(
                    persistent_prompt_cache_model_contract,
                    block_tokens,
                    block_causal_input,
                ),
                Some(parent_persistent_prompt_cache_block_key) => {
                    parent_persistent_prompt_cache_block_key
                        .for_child_block_with_causal_input(block_tokens, block_causal_input)
                }
            };
        let persistent_prompt_cache_block_key = match persistent_prompt_cache_block_key {
            // Invalid key construction is equivalent to a missing block here.
            // This pure decision layer intentionally remains fail-soft; the
            // engine can always process the complete prompt cold.
            Ok(persistent_prompt_cache_block_key) => persistent_prompt_cache_block_key,
            Err(_) => {
                lookup_diagnostics
                    .record_first_missing_sequence_state_block_index(block_index, None);
                break;
            }
        };
        if persistent_prompt_cache_model_contract.has_sequence_state()
            && !persistent_prompt_cache_kv_block_exists(
                &persistent_prompt_cache_block_key.block_hash(),
            )
        {
            lookup_diagnostics.record_first_missing_sequence_state_block_index(
                block_index,
                Some(persistent_prompt_cache_block_key.block_hash()),
            );
            break;
        }
        // Boundary-only models still construct the deterministic chain, but do not require
        // a sequence-state file. Their latest complete boundary snapshot is sufficient for
        // restore, whereas hybrid models require both forms of state at the same boundary.
        parent_persistent_prompt_cache_block_key = Some(persistent_prompt_cache_block_key);
        if let Some(parent_persistent_prompt_cache_block_key) =
            &parent_persistent_prompt_cache_block_key
        {
            matched_persistent_prompt_cache_block_keys
                .push(parent_persistent_prompt_cache_block_key.clone());
        }
    }
    matched_persistent_prompt_cache_block_keys
}

/// Scans the matched chain backward for the newest block whose recurrent
/// snapshot can seed continuation.
pub(super) fn newest_restorable_boundary_entry(
    has_boundary_state: bool,
    matched_persistent_prompt_cache_block_keys: &[PersistentPromptCacheBlockKey],
    persistent_prompt_cache_recurrent_snapshot_exists: impl Fn(&[u8; 32]) -> bool,
) -> Option<(usize, &PersistentPromptCacheBlockKey)> {
    // Boundary-only models still construct the deterministic chain, but do not require
    // a sequence-state file. Their latest complete boundary snapshot is sufficient for
    // restore, whereas hybrid models require both forms of state at the same boundary.
    matched_persistent_prompt_cache_block_keys
        .iter()
        .enumerate()
        .rev()
        .find(|(_block_index, persistent_prompt_cache_block_key)| {
            !has_boundary_state
                || persistent_prompt_cache_recurrent_snapshot_exists(
                    &persistent_prompt_cache_block_key.block_hash(),
                )
        })
}

/// Names the reason a matched chain could not be restored at a boundary.
pub(super) fn boundary_snapshot_miss_reason(
    has_sequence_state: bool,
    first_missing_sequence_state_block_index: Option<usize>,
) -> PersistentPromptCacheMissReason {
    match (has_sequence_state, first_missing_sequence_state_block_index) {
        (true, Some(0)) => PersistentPromptCacheMissReason::RootSequenceStateBlockMissing,
        (true, Some(_)) | (true, None) | (false, _) => {
            PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
        }
    }
}

/// Assembles the successful lookup result on top of the restored boundary,
/// probing one stored partial tail block when the caller asked for it.
pub(super) fn restored_prefix_lookup_result<KvBlockExists, RecurrentSnapshotExists>(
    lookup_inputs: &PersistentPromptCachePrefixLookupInputs<
        '_,
        KvBlockExists,
        RecurrentSnapshotExists,
    >,
    matched_persistent_prompt_cache_block_keys: &[PersistentPromptCacheBlockKey],
    recurrent_snapshot_block_index: usize,
    restored_persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
    mut lookup_diagnostics: PersistentPromptCacheLookupDiagnostics,
) -> PersistentPromptCachePrefixLookupResult
where
    KvBlockExists: Fn(&[u8; 32]) -> bool,
    RecurrentSnapshotExists: Fn(&[u8; 32]) -> bool,
{
    let persistent_prompt_cache_model_contract =
        lookup_inputs.persistent_prompt_cache_model_contract;
    let prompt_tokens = lookup_inputs.prompt_tokens;
    let persistent_prompt_cache_block_token_count =
        persistent_prompt_cache_model_contract.block_token_count();
    let restored_block_count =
        (restored_persistent_prompt_cache_block_key.block_index() as usize).saturating_add(1);
    let restored_complete_block_token_count =
        restored_block_count.saturating_mul(persistent_prompt_cache_block_token_count);
    if restored_complete_block_token_count == 0
        || restored_complete_block_token_count > prompt_tokens.len()
    {
        lookup_diagnostics
            .record_miss_reason(PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing);
        return cache_miss_lookup_result(prompt_tokens, lookup_diagnostics);
    }
    // Multi-turn prompts grow by a partial final block. When the previous
    // turn published that suffix as a tail, restore it on top of the
    // complete-block chain so only the uncached suffix is prefilled.
    let restored_partial_tail_block_key = if lookup_inputs.probe_partial_tail_blocks {
        partial_tail_probe::probe_restorable_partial_tail_block(
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            matched_persistent_prompt_cache_block_keys,
            recurrent_snapshot_block_index + 1 == matched_persistent_prompt_cache_block_keys.len(),
            lookup_inputs.allow_exact_block_boundary_restore,
            lookup_inputs.block_causal_inputs,
            &lookup_inputs.persistent_prompt_cache_kv_block_exists,
            &lookup_inputs.persistent_prompt_cache_recurrent_snapshot_exists,
        )
    } else {
        None
    };
    if let Some(restored_partial_tail_block_key) = &restored_partial_tail_block_key {
        lookup_diagnostics.record_restored_partial_tail_block_token_count(
            restored_partial_tail_block_key.token_count(),
        );
    }
    let restored_token_count = restored_complete_block_token_count
        + restored_partial_tail_block_key
            .as_ref()
            .map_or(0, PersistentPromptCacheBlockKey::token_count);
    // Return the untouched suffix rather than a block-rounded slice. The
    // final partial block and the required final token both belong to the
    // normal prefill path.
    let remaining_tokens = prompt_tokens[restored_token_count..].to_vec();
    PersistentPromptCachePrefixLookupResult {
        restored_token_count,
        remaining_tokens,
        last_restored_persistent_prompt_cache_block_key: Some(
            restored_persistent_prompt_cache_block_key.clone(),
        ),
        restored_partial_tail_block_key,
        lookup_diagnostics,
    }
}
