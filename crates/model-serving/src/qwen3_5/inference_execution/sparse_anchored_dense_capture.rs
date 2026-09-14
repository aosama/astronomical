//! Dense prompt-cache capture for the conversation tail that follows a SpecPrefill restore.
//!
//! SpecPrefill restores a *compact* decoder state: the sparse target slab contains only the
//! selected prompt rows, so its physical row offsets no longer match prompt positions. A
//! request that restores that state and then processes the remaining conversation densely
//! (the target-only path taken whenever the uncached suffix is shorter than the configured
//! minimum) cannot append ordinary token-aligned cache blocks: the ordinary chain assumes
//! row index == prompt position, and the rows between the restored prefix and the first
//! block boundary were never materialized densely at all.
//!
//! This module owns the arithmetic that makes that tail cacheable anyway (issue #659):
//! dense blocks are published in a chain *anchored* at the restored sparse prefix, with
//! block boundaries measured from the anchor instead of from prompt position zero, and with
//! their slab bytes read at the anchor-relative physical offset. The chain root binds the
//! exact sparse state identity, so a different selection over the same tokens can never
//! restore these blocks.

use super::super::super::persistent_cache::PersistentPromptCacheBlockKey;
use crate::PersistentPromptCacheBlockCausalInput;
use crate::PersistentPromptCacheModelContract;

/// Per-request facts needed to publish and restore the sparse-anchored dense tail chain.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::qwen3_5) struct SparseAnchoredDenseCaptureContext {
    /// Identity of the restored sparse target state, binding this chain to one exact selection.
    pub(in crate::qwen3_5) sparse_target_state_identity: [u8; 32],
    /// Prompt token count represented by the restored sparse prefix; the chain anchor.
    pub(in crate::qwen3_5) anchor_prompt_token_count: usize,
    /// Physical row count of the restored compact slab, i.e. the slab offset of the anchor.
    pub(in crate::qwen3_5) anchor_compact_row_count: usize,
    /// Newest anchored block this request published or restored, used as the next parent.
    pub(in crate::qwen3_5) last_published_block_key: Option<PersistentPromptCacheBlockKey>,
    /// Anchored blocks restored from SSD at request start, reported through diagnostics.
    pub(in crate::qwen3_5) restored_block_count: usize,
}

impl SparseAnchoredDenseCaptureContext {
    #[must_use]
    pub(in crate::qwen3_5) const fn new(
        sparse_target_state_identity: [u8; 32],
        anchor_prompt_token_count: usize,
        anchor_compact_row_count: usize,
    ) -> Self {
        Self {
            sparse_target_state_identity,
            anchor_prompt_token_count,
            anchor_compact_row_count,
            last_published_block_key: None,
            restored_block_count: 0,
        }
    }

    /// Returns the physical slab row offset holding one prompt position.
    ///
    /// Returns `None` for positions inside the compact prefix itself: those rows are
    /// selection-bound and never addressable as dense prompt-aligned bytes.
    #[must_use]
    pub(in crate::qwen3_5) fn compact_row_offset_for_prompt_position(
        &self,
        prompt_position: usize,
    ) -> Option<usize> {
        prompt_position
            .checked_sub(self.anchor_prompt_token_count)
            .and_then(|anchor_relative_offset| {
                self.anchor_compact_row_count
                    .checked_add(anchor_relative_offset)
            })
    }

    /// Builds the chain root key for the first dense block published after the anchor.
    pub(in crate::qwen3_5) fn root_block_key(
        &self,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        block_tokens: &[u32],
        block_causal_input: &PersistentPromptCacheBlockCausalInput,
    ) -> Result<PersistentPromptCacheBlockKey, crate::PersistentPromptCacheBlockKeyError> {
        PersistentPromptCacheBlockKey::for_sparse_anchored_root_block_with_causal_input(
            persistent_prompt_cache_model_contract,
            &self.sparse_target_state_identity,
            block_tokens,
            block_causal_input,
        )
    }

    /// Builds the next key in the anchored chain, or the root when none is published yet.
    pub(in crate::qwen3_5) fn next_block_key(
        &self,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        block_tokens: &[u32],
        block_causal_input: &PersistentPromptCacheBlockCausalInput,
    ) -> Result<PersistentPromptCacheBlockKey, crate::PersistentPromptCacheBlockKeyError> {
        match self.last_published_block_key.as_ref() {
            None => self.root_block_key(
                persistent_prompt_cache_model_contract,
                block_tokens,
                block_causal_input,
            ),
            Some(parent_block_key) => {
                parent_block_key.for_child_block_with_causal_input(block_tokens, block_causal_input)
            }
        }
    }
}
