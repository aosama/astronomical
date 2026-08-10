//! Pure-Rust persistent prompt-cache lookup for the longest restorable prompt prefix.
//!
//! This is the decision layer that the engine calls before any MLX allocation.
//! It hashes the prompt's persistent prompt-cache blocks in chain order, finds
//! the longest contiguous key/value block prefix, then walks backward to find
//! the newest recurrent snapshot that can safely seed continuation. It also
//! enforces the safety invariant that at least one prompt token always remains
//! for forward processing, even when the prompt ends exactly on a block
//! boundary.

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

    /// Returns why lookup failed, when it failed.
    #[must_use]
    pub const fn miss_reason(&self) -> Option<PersistentPromptCacheMissReason> {
        self.miss_reason
    }

    fn record_matched_sequence_state_block_count(
        &mut self,
        matched_sequence_state_block_count: usize,
    ) {
        self.matched_sequence_state_block_count = matched_sequence_state_block_count;
    }

    fn record_first_missing_sequence_state_block_index(
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

    fn record_miss_reason(&mut self, miss_reason: PersistentPromptCacheMissReason) {
        self.miss_reason = Some(miss_reason);
    }
}

/// The longest restorable persistent prompt-cache prefix, determined without MLX.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCachePrefixLookupResult {
    restored_token_count: usize,
    // This is copied so callers can retain the lookup outcome after the input
    // request buffer has been released. It is the exact suffix the engine must
    // still forward through the model.
    remaining_tokens: Vec<u32>,
    last_restored_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey>,
    lookup_diagnostics: PersistentPromptCacheLookupDiagnostics,
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
        Self::for_prompt_with_image_digests_and_boundary_policy(
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            &[],
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
        Self::for_prompt_with_image_digests_and_boundary_policy(
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            &[],
            true,
            persistent_prompt_cache_kv_block_exists,
            persistent_prompt_cache_recurrent_snapshot_exists,
        )
    }

    /// Computes the longest restorable prefix while binding ordered image identities.
    pub fn for_prompt_with_image_digests(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        prompt_tokens: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        persistent_prompt_cache_kv_block_exists: impl Fn(&[u8; 32]) -> bool,
        persistent_prompt_cache_recurrent_snapshot_exists: impl Fn(&[u8; 32]) -> bool,
    ) -> PersistentPromptCachePrefixLookupResult {
        Self::for_prompt_with_image_digests_and_boundary_policy(
            persistent_prompt_cache_model_contract,
            prompt_tokens,
            ordered_image_sha256_digests,
            false,
            persistent_prompt_cache_kv_block_exists,
            persistent_prompt_cache_recurrent_snapshot_exists,
        )
    }

    fn for_prompt_with_image_digests_and_boundary_policy(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        prompt_tokens: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        allow_exact_block_boundary_restore: bool,
        persistent_prompt_cache_kv_block_exists: impl Fn(&[u8; 32]) -> bool,
        persistent_prompt_cache_recurrent_snapshot_exists: impl Fn(&[u8; 32]) -> bool,
    ) -> PersistentPromptCachePrefixLookupResult {
        let persistent_prompt_cache_block_token_count =
            persistent_prompt_cache_model_contract.block_token_count();
        // The contract resolver guarantees a nonzero block length. Keeping the calculation here
        // contract-driven makes lookup agree with capture on models whose state geometry selects
        // a different boundary size than another artifact on the same machine.
        let complete_prompt_block_count =
            prompt_tokens.len() / persistent_prompt_cache_block_token_count;
        // An exact block-boundary prompt must retain its final block for a
        // forward pass. The final prompt token produces the logits used to
        // begin decode, while restoring only recurrent state cannot substitute
        // for that model computation.
        let maximum_restorable_block_count = if allow_exact_block_boundary_restore {
            complete_prompt_block_count
        } else if prompt_tokens
            .len()
            .is_multiple_of(persistent_prompt_cache_block_token_count)
        {
            complete_prompt_block_count.saturating_sub(1)
        } else {
            complete_prompt_block_count
        };
        let mut lookup_diagnostics = PersistentPromptCacheLookupDiagnostics::new(
            complete_prompt_block_count,
            maximum_restorable_block_count,
        );
        if maximum_restorable_block_count == 0 {
            lookup_diagnostics.record_miss_reason(
                PersistentPromptCacheMissReason::PromptTooShortForPersistentPromptCache,
            );
            return cache_miss_lookup_result(prompt_tokens, lookup_diagnostics);
        }
        // The key chain is constructed in prompt order. A hit for block N is
        // meaningful only if every predecessor also matched, which prevents
        // reuse across prompts that diverge in an earlier block.
        let mut matched_persistent_prompt_cache_block_keys =
            Vec::with_capacity(maximum_restorable_block_count);
        let mut parent_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey> =
            None;
        for block_index in 0..maximum_restorable_block_count {
            let block_start = block_index * persistent_prompt_cache_block_token_count;
            let block_end = block_start + persistent_prompt_cache_block_token_count;
            let block_tokens = &prompt_tokens[block_start..block_end];
            let persistent_prompt_cache_block_key =
                match parent_persistent_prompt_cache_block_key.as_ref() {
                    None => PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
                        persistent_prompt_cache_model_contract,
                        block_tokens,
                        ordered_image_sha256_digests,
                    ),
                    Some(parent_persistent_prompt_cache_block_key) => {
                        parent_persistent_prompt_cache_block_key.for_child_block(block_tokens)
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
        if persistent_prompt_cache_model_contract.has_sequence_state() {
            lookup_diagnostics.record_matched_sequence_state_block_count(
                matched_persistent_prompt_cache_block_keys.len(),
            );
        }
        let newest_recurrent_snapshot_entry = matched_persistent_prompt_cache_block_keys
            .iter()
            .enumerate()
            .rev()
            .find(|(_block_index, persistent_prompt_cache_block_key)| {
                !persistent_prompt_cache_model_contract.has_boundary_state()
                    || persistent_prompt_cache_recurrent_snapshot_exists(
                        &persistent_prompt_cache_block_key.block_hash(),
                    )
            });
        let Some((recurrent_snapshot_block_index, restored_persistent_prompt_cache_block_key)) =
            newest_recurrent_snapshot_entry
        else {
            let miss_reason = match (
                persistent_prompt_cache_model_contract.has_sequence_state(),
                lookup_diagnostics.first_missing_sequence_state_block_index(),
            ) {
                (true, Some(0)) => PersistentPromptCacheMissReason::RootSequenceStateBlockMissing,
                (true, Some(_)) | (true, None) | (false, _) => {
                    PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
                }
            };
            lookup_diagnostics.record_miss_reason(miss_reason);
            return cache_miss_lookup_result(prompt_tokens, lookup_diagnostics);
        };
        lookup_diagnostics
            .record_newest_boundary_state_snapshot_block_index(recurrent_snapshot_block_index);
        let restored_block_count =
            (restored_persistent_prompt_cache_block_key.block_index() as usize).saturating_add(1);
        let restored_token_count =
            restored_block_count.saturating_mul(persistent_prompt_cache_block_token_count);
        if restored_token_count == 0 || restored_token_count > prompt_tokens.len() {
            lookup_diagnostics
                .record_miss_reason(PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing);
            return cache_miss_lookup_result(prompt_tokens, lookup_diagnostics);
        }
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
            lookup_diagnostics,
        }
    }
}

fn cache_miss_lookup_result(
    prompt_tokens: &[u32],
    lookup_diagnostics: PersistentPromptCacheLookupDiagnostics,
) -> PersistentPromptCachePrefixLookupResult {
    PersistentPromptCachePrefixLookupResult {
        restored_token_count: 0,
        remaining_tokens: prompt_tokens.to_vec(),
        last_restored_persistent_prompt_cache_block_key: None,
        lookup_diagnostics,
    }
}
