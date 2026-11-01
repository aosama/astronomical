//! Pure-Rust persistent prompt-cache lookup for the longest restorable prompt prefix.
//!
//! This is the decision layer that the engine calls before any MLX allocation.
//! It hashes the prompt's persistent prompt-cache blocks in chain order, finds
//! the longest contiguous key/value block prefix, then walks backward to find
//! the newest recurrent snapshot that can safely seed continuation. It also
//! enforces the safety invariant that at least one prompt token always remains
//! for forward processing, even when the prompt ends exactly on a block
//! boundary.

use super::prefix_lookup_steps::{
    PersistentPromptCachePrefixLookupInputs, boundary_snapshot_miss_reason,
    complete_and_maximum_restorable_block_counts, matched_sequence_state_block_keys,
    newest_restorable_boundary_entry, prompt_too_short_lookup_result,
    restored_prefix_lookup_result,
};
use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::{PersistentPromptCacheBlockKey, PersistentPromptCacheModelContract};

/// The exact reason a prompt prefix could not be restored from the persistent prompt cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentPromptCacheMissReason {
    /// The prompt cannot produce even one safely restorable block.
    PromptTooShortForPersistentPromptCache,
    /// The first model-derived prompt block does not match tracked sequence state.
    RootSequenceStateBlockMissing,
    /// Matched KV blocks exist, but none has a usable recurrent-state snapshot.
    BoundaryStateSnapshotMissing,
}

/// Evidence gathered while looking for the longest restorable persistent prompt-cache prefix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCacheLookupDiagnostics {
    complete_prompt_block_count: usize,
    maximum_restorable_block_count: usize,
    matched_sequence_state_block_count: usize,
    first_missing_sequence_state_block_index: Option<usize>,
    first_missing_sequence_state_block_hash: Option<[u8; 32]>,
    newest_boundary_state_snapshot_block_index: Option<usize>,
    restored_partial_tail_block_token_count: Option<usize>,
    miss_reason: Option<PersistentPromptCacheMissReason>,
}

impl PersistentPromptCacheLookupDiagnostics {
    const fn new(
        complete_prompt_block_count: usize,
        maximum_restorable_block_count: usize,
    ) -> Self {
        Self {
            complete_prompt_block_count,
            maximum_restorable_block_count,
            matched_sequence_state_block_count: 0,
            first_missing_sequence_state_block_index: None,
            first_missing_sequence_state_block_hash: None,
            newest_boundary_state_snapshot_block_index: None,
            restored_partial_tail_block_token_count: None,
            miss_reason: None,
        }
    }

    /// Returns the number of complete persistent prompt-cache blocks in the prompt.
    #[must_use]
    pub const fn complete_prompt_block_count(&self) -> usize {
        self.complete_prompt_block_count
    }

    /// Returns the number of complete blocks that are eligible for restore.
    #[must_use]
    pub const fn maximum_restorable_block_count(&self) -> usize {
        self.maximum_restorable_block_count
    }

    /// Returns how many contiguous KV blocks matched before lookup stopped.
    #[must_use]
    pub const fn matched_sequence_state_block_count(&self) -> usize {
        self.matched_sequence_state_block_count
    }

    /// Returns the first missing KV block position, when lookup stopped at a gap.
    #[must_use]
    pub const fn first_missing_sequence_state_block_index(&self) -> Option<usize> {
        self.first_missing_sequence_state_block_index
    }

    /// Returns the expected block hash for the first missing KV block, when available.
    #[must_use]
    pub const fn first_missing_sequence_state_block_hash(&self) -> Option<[u8; 32]> {
        self.first_missing_sequence_state_block_hash
    }

    /// Returns the recurrent snapshot position used as the restore boundary.
    #[must_use]
    pub const fn newest_boundary_state_snapshot_block_index(&self) -> Option<usize> {
        self.newest_boundary_state_snapshot_block_index
    }

    /// Returns the token count of the restored partial tail block, when one matched.
    #[must_use]
    pub const fn restored_partial_tail_block_token_count(&self) -> Option<usize> {
        self.restored_partial_tail_block_token_count
    }

    /// Returns why lookup failed, when it failed.
    #[must_use]
    pub const fn miss_reason(&self) -> Option<PersistentPromptCacheMissReason> {
        self.miss_reason
    }

    /// Test fixture: a diagnostics record with the fields the restore-support
    /// mapping consumes set explicitly, so the IPC contract can be tested
    /// without driving a full prefix lookup against a disk store.
    #[cfg(all(test, feature = "direct-mlx"))]
    pub(crate) fn for_tests(
        complete_prompt_block_count: usize,
        maximum_restorable_block_count: usize,
        matched_sequence_state_block_count: usize,
        miss_reason: Option<PersistentPromptCacheMissReason>,
    ) -> Self {
        Self {
            complete_prompt_block_count,
            maximum_restorable_block_count,
            matched_sequence_state_block_count,
            first_missing_sequence_state_block_index: None,
            first_missing_sequence_state_block_hash: None,
            newest_boundary_state_snapshot_block_index: None,
            restored_partial_tail_block_token_count: None,
            miss_reason,
        }
    }

    pub(super) fn record_matched_sequence_state_block_count(
        &mut self,
        matched_sequence_state_block_count: usize,
    ) {
        self.matched_sequence_state_block_count = matched_sequence_state_block_count;
    }

    pub(super) fn record_first_missing_sequence_state_block_index(
        &mut self,
        missing_sequence_state_block_index: usize,
        missing_sequence_state_block_hash: Option<[u8; 32]>,
    ) {
        self.first_missing_sequence_state_block_index = Some(missing_sequence_state_block_index);
        self.first_missing_sequence_state_block_hash = missing_sequence_state_block_hash;
    }

    fn record_newest_boundary_state_snapshot_block_index(
        &mut self,
        boundary_state_snapshot_block_index: usize,
    ) {
        self.newest_boundary_state_snapshot_block_index = Some(boundary_state_snapshot_block_index);
    }

    pub(super) fn record_restored_partial_tail_block_token_count(
        &mut self,
        restored_partial_tail_block_token_count: usize,
    ) {
        self.restored_partial_tail_block_token_count =
            Some(restored_partial_tail_block_token_count);
    }

    pub(super) fn record_miss_reason(&mut self, miss_reason: PersistentPromptCacheMissReason) {
        self.miss_reason = Some(miss_reason);
    }
}

/// The longest restorable persistent prompt-cache prefix, determined without MLX.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCachePrefixLookupResult {
    pub(super) restored_token_count: usize,
    // This is copied so callers can retain the lookup outcome after the input
    // request buffer has been released. It is the exact suffix the engine must
    // still forward through the model.
    pub(super) remaining_tokens: Vec<u32>,
    pub(super) last_restored_persistent_prompt_cache_block_key:
        Option<PersistentPromptCacheBlockKey>,
    pub(super) restored_partial_tail_block_key: Option<PersistentPromptCacheBlockKey>,
    pub(super) lookup_diagnostics: PersistentPromptCacheLookupDiagnostics,
}

impl PersistentPromptCachePrefixLookupResult {
    /// Returns the number of prompt tokens that can be restored from the persistent prompt cache.
    #[must_use]
    pub fn restored_token_count(&self) -> usize {
        self.restored_token_count
    }

    /// Returns the prompt tokens that still need forward processing.
    #[must_use]
    pub fn remaining_tokens(&self) -> &[u32] {
        &self.remaining_tokens
    }

    /// Returns the persistent prompt-cache block identity of the last matched block, if any.
    ///
    /// The engine uses this to chain the next block it saves during prefill:
    /// `last_restored_persistent_prompt_cache_block_key.for_child_block(next_block_tokens)`.
    #[must_use]
    pub fn last_restored_persistent_prompt_cache_block_key(
        &self,
    ) -> Option<&PersistentPromptCacheBlockKey> {
        self.last_restored_persistent_prompt_cache_block_key
            .as_ref()
    }

    /// Returns the partial tail block restored on top of the last complete block, if any.
    ///
    /// A tail holds fewer tokens than one full block and is never a chain
    /// parent: the engine must keep chaining new complete blocks from
    /// `last_restored_persistent_prompt_cache_block_key`, which stays on the
    /// last complete block even when a tail was restored.
    #[must_use]
    pub fn restored_partial_tail_block_key(&self) -> Option<&PersistentPromptCacheBlockKey> {
        self.restored_partial_tail_block_key.as_ref()
    }

    /// Returns evidence describing how the lookup reached its result.
    #[must_use]
    pub const fn diagnostics(&self) -> &PersistentPromptCacheLookupDiagnostics {
        &self.lookup_diagnostics
    }
}

/// Walks a prompt and queries the persistent prompt cache to find the longest prefix.
pub struct PersistentPromptCachePrefixLookup;

impl PersistentPromptCachePrefixLookup {
    /// Computes the longest restorable prefix of one prompt.
    ///
    /// The caller supplies separate KV-block and recurrent-snapshot predicates
    /// to query the persistent prompt cache without coupling this pure-Rust
    /// layer to MLX or the filesystem.
    /// The result never consumes the final prompt token, so the engine
    /// always has at least one token to feed forward.
    pub fn for_prompt(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        prompt_tokens: &[u32],
        persistent_prompt_cache_kv_block_exists: impl Fn(&[u8; 32]) -> bool,
        persistent_prompt_cache_recurrent_snapshot_exists: impl Fn(&[u8; 32]) -> bool,
    ) -> PersistentPromptCachePrefixLookupResult {
        Self::for_prompt_with_block_causal_inputs_and_boundary_policy(
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            None,
            false,
            false,
            persistent_prompt_cache_kv_block_exists,
            persistent_prompt_cache_recurrent_snapshot_exists,
        )
    }

    /// Computes the longest restorable complete prefix, including an exact block boundary.
    ///
    /// This variant is for a private decoder state owner that will continue with
    /// another model-side operation. Unlike generation startup, it does not need
    /// to retain a final prompt token for logits production.
    pub fn for_complete_prefix(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        prompt_tokens: &[u32],
        persistent_prompt_cache_kv_block_exists: impl Fn(&[u8; 32]) -> bool,
        persistent_prompt_cache_recurrent_snapshot_exists: impl Fn(&[u8; 32]) -> bool,
    ) -> PersistentPromptCachePrefixLookupResult {
        Self::for_prompt_with_block_causal_inputs_and_boundary_policy(
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            None,
            true,
            false,
            persistent_prompt_cache_kv_block_exists,
            persistent_prompt_cache_recurrent_snapshot_exists,
        )
    }

    /// Computes the longest restorable prefix with model-owned causal input for each block.
    pub fn for_prompt_with_block_causal_inputs(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        prompt_tokens: &[u32],
        block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
        persistent_prompt_cache_kv_block_exists: impl Fn(&[u8; 32]) -> bool,
        persistent_prompt_cache_recurrent_snapshot_exists: impl Fn(&[u8; 32]) -> bool,
    ) -> PersistentPromptCachePrefixLookupResult {
        Self::for_prompt_with_block_causal_inputs_and_boundary_policy(
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            Some(block_causal_inputs),
            false,
            false,
            persistent_prompt_cache_kv_block_exists,
            persistent_prompt_cache_recurrent_snapshot_exists,
        )
    }

    /// Computes the longest prefix covered by the retained live decoder state.
    ///
    /// In-memory prefix adoption (issue #1167) matches the new prompt's hash
    /// chain against the block keys the last completed generation published.
    /// The retained state covers complete blocks only — its recurrent state
    /// sits at the generation-end position, so the adopted boundary always
    /// absorbs the boundary snapshot from the disk store, which is why the
    /// recurrent predicate stays caller-supplied. No partial tail is probed:
    /// tails are disk-store facts, and the exact-boundary rule matches the
    /// disk restore so the adopted prefix always keeps a token to forward.
    pub fn for_retained_complete_block_keys(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        prompt_tokens: &[u32],
        block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
        retained_complete_block_keys: &[PersistentPromptCacheBlockKey],
        persistent_prompt_cache_recurrent_snapshot_exists: impl Fn(&[u8; 32]) -> bool,
    ) -> PersistentPromptCachePrefixLookupResult {
        let retained_block_hashes = retained_complete_block_keys
            .iter()
            .map(|retained_block_key| retained_block_key.block_hash())
            .collect::<Vec<_>>();
        Self::for_prompt_with_block_causal_inputs_and_boundary_policy(
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            if block_causal_inputs.is_empty() {
                None
            } else {
                Some(block_causal_inputs)
            },
            false,
            false,
            |block_hash| retained_block_hashes.contains(block_hash),
            persistent_prompt_cache_recurrent_snapshot_exists,
        )
    }

    /// Computes the longest restorable prefix, additionally reusing one stored partial tail block.
    ///
    /// Multi-turn prompts grow by a partial final block between turns. When the
    /// previous turn published that partial suffix as a tail block, this lookup
    /// restores it on top of the matched complete-block chain so the next turn
    /// prefills only the uncached suffix. The result keeps the generation
    /// invariant of never consuming the final prompt token, and the tail is
    /// reported separately from the complete-block chain because tails are
    /// never chain parents.
    pub fn for_prompt_with_partial_tail_block_recovery(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        prompt_tokens: &[u32],
        block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
        persistent_prompt_cache_kv_block_exists: impl Fn(&[u8; 32]) -> bool,
        persistent_prompt_cache_recurrent_snapshot_exists: impl Fn(&[u8; 32]) -> bool,
    ) -> PersistentPromptCachePrefixLookupResult {
        // An empty causal-input slice means "no model-owned causal input" for
        // the existing constructors; preserve that meaning here too.
        let block_causal_inputs = if block_causal_inputs.is_empty() {
            None
        } else {
            Some(block_causal_inputs)
        };
        Self::for_prompt_with_block_causal_inputs_and_boundary_policy(
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            block_causal_inputs,
            false,
            true,
            persistent_prompt_cache_kv_block_exists,
            persistent_prompt_cache_recurrent_snapshot_exists,
        )
    }

    fn for_prompt_with_block_causal_inputs_and_boundary_policy<
        KvBlockExists,
        RecurrentSnapshotExists,
    >(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        prompt_tokens: &[u32],
        block_causal_inputs: Option<&[PersistentPromptCacheBlockCausalInput]>,
        allow_exact_block_boundary_restore: bool,
        probe_partial_tail_blocks: bool,
        persistent_prompt_cache_kv_block_exists: KvBlockExists,
        persistent_prompt_cache_recurrent_snapshot_exists: RecurrentSnapshotExists,
    ) -> PersistentPromptCachePrefixLookupResult
    where
        KvBlockExists: Fn(&[u8; 32]) -> bool,
        RecurrentSnapshotExists: Fn(&[u8; 32]) -> bool,
    {
        let (complete_prompt_block_count, maximum_restorable_block_count) =
            complete_and_maximum_restorable_block_counts(
                prompt_tokens.len(),
                persistent_prompt_cache_model_contract.block_token_count(),
                allow_exact_block_boundary_restore,
            );
        let mut lookup_diagnostics = PersistentPromptCacheLookupDiagnostics::new(
            complete_prompt_block_count,
            maximum_restorable_block_count,
        );
        let lookup_inputs = PersistentPromptCachePrefixLookupInputs {
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            block_causal_inputs,
            allow_exact_block_boundary_restore,
            probe_partial_tail_blocks,
            persistent_prompt_cache_kv_block_exists,
            persistent_prompt_cache_recurrent_snapshot_exists,
        };
        if maximum_restorable_block_count == 0 {
            return prompt_too_short_lookup_result(&lookup_inputs, lookup_diagnostics);
        }
        let matched_persistent_prompt_cache_block_keys = matched_sequence_state_block_keys(
            &lookup_inputs,
            maximum_restorable_block_count,
            &mut lookup_diagnostics,
        );
        if persistent_prompt_cache_model_contract.has_sequence_state() {
            lookup_diagnostics.record_matched_sequence_state_block_count(
                matched_persistent_prompt_cache_block_keys.len(),
            );
        }
        let newest_recurrent_snapshot_entry = newest_restorable_boundary_entry(
            persistent_prompt_cache_model_contract.has_boundary_state(),
            &matched_persistent_prompt_cache_block_keys,
            &lookup_inputs.persistent_prompt_cache_recurrent_snapshot_exists,
        );
        let Some((recurrent_snapshot_block_index, restored_persistent_prompt_cache_block_key)) =
            newest_recurrent_snapshot_entry
        else {
            let miss_reason = boundary_snapshot_miss_reason(
                persistent_prompt_cache_model_contract.has_sequence_state(),
                lookup_diagnostics.first_missing_sequence_state_block_index(),
            );
            lookup_diagnostics.record_miss_reason(miss_reason);
            return cache_miss_lookup_result(prompt_tokens, lookup_diagnostics);
        };
        lookup_diagnostics
            .record_newest_boundary_state_snapshot_block_index(recurrent_snapshot_block_index);
        restored_prefix_lookup_result(
            &lookup_inputs,
            &matched_persistent_prompt_cache_block_keys,
            recurrent_snapshot_block_index,
            restored_persistent_prompt_cache_block_key,
            lookup_diagnostics,
        )
    }
}

pub(super) fn cache_miss_lookup_result(
    prompt_tokens: &[u32],
    lookup_diagnostics: PersistentPromptCacheLookupDiagnostics,
) -> PersistentPromptCachePrefixLookupResult {
    PersistentPromptCachePrefixLookupResult {
        restored_token_count: 0,
        remaining_tokens: prompt_tokens.to_vec(),
        last_restored_persistent_prompt_cache_block_key: None,
        restored_partial_tail_block_key: None,
        lookup_diagnostics,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "prefix_lookup_tests.rs"]
mod prefix_lookup_tests;
