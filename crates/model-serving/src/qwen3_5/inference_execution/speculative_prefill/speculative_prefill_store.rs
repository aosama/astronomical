//! Defines exact reuse identities and tiny worker-local hot stores.
//!
//! Two kinds of data are cached in memory:
//!
//! - selections: absolute target positions for an exact selectable token suffix;
//! - drafter prefixes: MLX decoder allocation checkpoints for a complete prompt prefix.
//!
//! Both stores are deliberately tiny because checkpoints retain live MLX payload
//! and because persistent storage handles reuse across processes. Keys include
//! every policy/model fact that can change selected positions or decoder meaning.

use crate::{
    PersistentSpeculativePrefillSelectionContract, PersistentSpeculativePrefillTargetStateContract,
    Qwen3_5ExecutionError, RequestDecoderStateStack, RequestDecoderStateStackAllocationCheckpoint,
};

use super::super::Qwen3_5EngineState;

// These are hard caps on worker-local acceleration state, not user-facing model
// memory budgets. Insertion evicts one existing entry when a new identity arrives.
const SPECULATIVE_PREFILL_SELECTION_STORE_ENTRY_LIMIT: usize = 4;
const SPECULATIVE_PREFILL_DRAFT_PREFIX_STORE_ENTRY_LIMIT: usize = 2;

/// Complete identity for worker-local SpecPrefill reuse.
///
/// Optional fields make key construction total while a model is being activated;
/// matching helpers reject keys that do not equal the engine's resolved identity.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct Qwen3_5SpeculativePrefillStoreKey {
    /// Configured drafter identity and validated revision.
    pub(crate) draft_model_id: Option<String>,
    pub(crate) draft_model_revision: Option<String>,
    /// Canonical target/drafter token-to-identifier mapping contract.
    pub(crate) token_identifier_mapping_digest: Option<[u8; 32]>,
    /// Selection policy inputs; changing any one invalidates reuse.
    pub(crate) keep_percentage: u32,
    pub(crate) selection_chunk_token_count: u32,
    pub(crate) mandatory_trailing_token_count: u32,
    pub(crate) lookahead_token_count: u32,
    pub(crate) importance_pooling_kernel_token_count: u32,
    /// Logical decoder/selection position represented by `token_ids`.
    pub(crate) position_tokens: u32,
    /// Exact prompt content represented by the cached value.
    pub(crate) token_ids: Vec<u32>,
}

/// Live drafter decoder checkpoint plus its attributable payload size.
pub(crate) struct Qwen3_5SpeculativePrefillDraftPrefixStoreEntry {
    pub(crate) allocation_checkpoint: RequestDecoderStateStackAllocationCheckpoint,
    pub(crate) payload_bytes: u64,
}

impl Qwen3_5EngineState {
    /// Builds the durable sparse-target-state identity from fully loaded engine state.
    ///
    /// `None` means activation has not resolved every required identity component;
    /// callers must not persist under a partial contract.
    pub(crate) fn speculative_prefill_target_state_contract(
        &self,
    ) -> Option<PersistentSpeculativePrefillTargetStateContract> {
        Some(PersistentSpeculativePrefillTargetStateContract::new(
            self.model_id.clone()?,
            self.model_revision.clone()?,
            self.speculative_prefill.draft_model_id.clone()?,
            self.speculative_prefill_draft_model_revision.clone()?,
            self.speculative_prefill_token_identifier_mapping_digest?,
            self.speculative_prefill.keep_percentage,
            self.speculative_prefill.selection_chunk_token_count,
            self.speculative_prefill.mandatory_trailing_token_count,
            self.speculative_prefill.lookahead_token_count,
            self.speculative_prefill
                .importance_pooling_kernel_token_count,
        ))
    }

    /// Builds the durable selection identity for one selectable prompt range.
    pub(crate) fn speculative_prefill_selection_contract(
        &self,
        position_tokens: u32,
        prompt_token_count: usize,
    ) -> Option<PersistentSpeculativePrefillSelectionContract> {
        Some(PersistentSpeculativePrefillSelectionContract::new(
            self.model_id.clone()?,
            self.model_revision.clone()?,
            self.speculative_prefill.draft_model_id.clone()?,
            self.speculative_prefill_draft_model_revision.clone()?,
            self.speculative_prefill_token_identifier_mapping_digest?,
            self.speculative_prefill.keep_percentage,
            self.speculative_prefill.selection_chunk_token_count,
            self.speculative_prefill.mandatory_trailing_token_count,
            self.speculative_prefill.lookahead_token_count,
            self.speculative_prefill
                .importance_pooling_kernel_token_count,
            position_tokens,
            u32::try_from(prompt_token_count).ok()?,
        ))
    }

    /// Builds the exact worker-memory key used by both hot stores.
    pub(crate) fn speculative_prefill_store_key(
        &self,
        position_tokens: u32,
        token_ids: Vec<u32>,
    ) -> Qwen3_5SpeculativePrefillStoreKey {
        Qwen3_5SpeculativePrefillStoreKey {
            draft_model_id: self.speculative_prefill.draft_model_id.clone(),
            draft_model_revision: self.speculative_prefill_draft_model_revision.clone(),
            token_identifier_mapping_digest: self
                .speculative_prefill_token_identifier_mapping_digest,
            keep_percentage: self.speculative_prefill.keep_percentage,
            selection_chunk_token_count: self.speculative_prefill.selection_chunk_token_count,
            mandatory_trailing_token_count: self.speculative_prefill.mandatory_trailing_token_count,
            lookahead_token_count: self.speculative_prefill.lookahead_token_count,
            importance_pooling_kernel_token_count: self
                .speculative_prefill
                .importance_pooling_kernel_token_count,
            position_tokens,
            token_ids,
        }
    }

    /// Inserts one selection into the bounded worker hot store.
    ///
    /// Replacing an existing identity does not evict another entry. For a new
    /// identity at capacity, removal of the map's first key is sufficient: this
    /// is a tiny cap, not an ordered least-recently-used policy.
    pub(crate) fn store_speculative_prefill_selection(
        &self,
        selection_store_key: Qwen3_5SpeculativePrefillStoreKey,
        selected_token_positions: Vec<usize>,
    ) {
        let mut selection_store = self.speculative_prefill_selection_store.borrow_mut();
        if selection_store.len() >= SPECULATIVE_PREFILL_SELECTION_STORE_ENTRY_LIMIT
            && !selection_store.contains_key(&selection_store_key)
            && let Some(previous_selection_store_key) = selection_store.keys().next().cloned()
        {
            selection_store.remove(&previous_selection_store_key);
        }
        selection_store.insert(selection_store_key, selected_token_positions);
    }

    /// Restores one exact drafter checkpoint while leaving a reusable clone in the store.
    ///
    /// Checkpoints are move-only ownership handles. The entry is removed before
    /// restoration, then a new checkpoint is captured from the restored state and
    /// reinserted so future exact requests can reuse the same logical prefix.
    pub(crate) fn restore_speculative_prefill_draft_prefix_checkpoint(
        &self,
        draft_prefix_store_key: &Qwen3_5SpeculativePrefillStoreKey,
        draft_request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<bool, Qwen3_5ExecutionError> {
        let Some(draft_prefix_store_entry) = self
            .speculative_prefill_draft_prefix_store
            .borrow_mut()
            .remove(draft_prefix_store_key)
        else {
            return Ok(false);
        };
        let draft_prefix_payload_bytes = draft_prefix_store_entry.payload_bytes;
        draft_request_decoder_state
            .restore_allocation_checkpoint(draft_prefix_store_entry.allocation_checkpoint)?;
        let reusable_draft_prefix_allocation_checkpoint =
            draft_request_decoder_state.allocation_checkpoint()?;
        self.speculative_prefill_draft_prefix_store
            .borrow_mut()
            .insert(
                draft_prefix_store_key.clone(),
                Qwen3_5SpeculativePrefillDraftPrefixStoreEntry {
                    allocation_checkpoint: reusable_draft_prefix_allocation_checkpoint,
                    payload_bytes: draft_prefix_payload_bytes,
                },
            );
        Ok(true)
    }

    /// Finds and restores the longest strict worker-memory prefix of a prompt.
    ///
    /// A strict prefix (`cached_len < prompt_len`) guarantees scoring has a suffix
    /// to advance. Exact full-prompt selection reuse is handled by the separate
    /// selection store before this decoder-state lookup.
    pub(crate) fn restore_longest_speculative_prefill_draft_prefix_checkpoint(
        &self,
        prompt_token_ids: &[u32],
        draft_request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Option<usize>, Qwen3_5ExecutionError> {
        let longest_matching_draft_prefix_store_key = self
            .speculative_prefill_draft_prefix_store
            .borrow()
            .keys()
            .filter(|draft_prefix_store_key| {
                self.speculative_prefill_store_key_matches_draft_identity(draft_prefix_store_key)
                    && draft_prefix_store_key.token_ids.len() < prompt_token_ids.len()
                    && prompt_token_ids.starts_with(&draft_prefix_store_key.token_ids)
            })
            .max_by_key(|draft_prefix_store_key| draft_prefix_store_key.token_ids.len())
            .cloned();
        let Some(longest_matching_draft_prefix_store_key) = longest_matching_draft_prefix_store_key
        else {
            return Ok(None);
        };
        let restored_prefix_token_count = longest_matching_draft_prefix_store_key.token_ids.len();
        self.restore_speculative_prefill_draft_prefix_checkpoint(
            &longest_matching_draft_prefix_store_key,
            draft_request_decoder_state,
        )?;
        Ok(Some(restored_prefix_token_count))
    }

    /// Inserts one live drafter prefix checkpoint into the bounded worker store.
    pub(crate) fn store_speculative_prefill_draft_prefix_checkpoint(
        &self,
        draft_prefix_store_key: Qwen3_5SpeculativePrefillStoreKey,
        draft_prefix_allocation_checkpoint: RequestDecoderStateStackAllocationCheckpoint,
        draft_prefix_payload_bytes: u64,
    ) {
        let mut draft_prefix_store = self.speculative_prefill_draft_prefix_store.borrow_mut();
        if draft_prefix_store.len() >= SPECULATIVE_PREFILL_DRAFT_PREFIX_STORE_ENTRY_LIMIT
            && !draft_prefix_store.contains_key(&draft_prefix_store_key)
            && let Some(previous_draft_prefix_store_key) = draft_prefix_store.keys().next().cloned()
        {
            draft_prefix_store.remove(&previous_draft_prefix_store_key);
        }
        draft_prefix_store.insert(
            draft_prefix_store_key,
            Qwen3_5SpeculativePrefillDraftPrefixStoreEntry {
                allocation_checkpoint: draft_prefix_allocation_checkpoint,
                payload_bytes: draft_prefix_payload_bytes,
            },
        );
    }

    /// Verifies that a cached checkpoint was produced by the exact active drafter policy.
    ///
    /// The final position/length equality also rejects malformed keys whose
    /// decoder position does not describe the stored token prefix.
    fn speculative_prefill_store_key_matches_draft_identity(
        &self,
        draft_prefix_store_key: &Qwen3_5SpeculativePrefillStoreKey,
    ) -> bool {
        draft_prefix_store_key.draft_model_id == self.speculative_prefill.draft_model_id
            && draft_prefix_store_key.draft_model_revision
                == self.speculative_prefill_draft_model_revision
            && draft_prefix_store_key.token_identifier_mapping_digest
                == self.speculative_prefill_token_identifier_mapping_digest
            && draft_prefix_store_key.keep_percentage == self.speculative_prefill.keep_percentage
            && draft_prefix_store_key.selection_chunk_token_count
                == self.speculative_prefill.selection_chunk_token_count
            && draft_prefix_store_key.mandatory_trailing_token_count
                == self.speculative_prefill.mandatory_trailing_token_count
            && draft_prefix_store_key.lookahead_token_count
                == self.speculative_prefill.lookahead_token_count
            && draft_prefix_store_key.importance_pooling_kernel_token_count
                == self
                    .speculative_prefill
                    .importance_pooling_kernel_token_count
            && usize::try_from(draft_prefix_store_key.position_tokens).ok()
                == Some(draft_prefix_store_key.token_ids.len())
    }
}
