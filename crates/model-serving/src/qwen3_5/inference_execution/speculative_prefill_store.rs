use crate::{
    PersistentSpeculativePrefillSelectionContract, PersistentSpeculativePrefillTargetStateContract,
    Qwen3_5ExecutionError, RequestDecoderStateStack, RequestDecoderStateStackAllocationCheckpoint,
};

use super::Qwen3_5EngineState;

const SPECULATIVE_PREFILL_SELECTION_STORE_ENTRY_LIMIT: usize = 4;
const SPECULATIVE_PREFILL_DRAFT_PREFIX_STORE_ENTRY_LIMIT: usize = 2;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct Qwen3_5SpeculativePrefillStoreKey {
    pub(super) draft_model_id: Option<String>,
    pub(super) draft_model_revision: Option<String>,
    pub(super) token_identifier_mapping_digest: Option<[u8; 32]>,
    pub(super) keep_percentage: u32,
    pub(super) selection_chunck_token_count: u32,
    pub(super) mandatory_trailing_token_count: u32,
    pub(super) lookahead_token_count: u32,
    pub(super) importance_pooling_kernel_token_count: u32,
    pub(super) position_tokens: u32,
    pub(super) token_ids: Vec<u32>,
}

pub(crate) struct Qwen3_5SpeculativePrefillDraftPrefixStoreEntry {
    pub(crate) allocation_checkpoint: RequestDecoderStateStackAllocationCheckpoint,
    pub(crate) payload_bytes: u64,
}

impl Qwen3_5EngineState {
    pub(super) fn speculative_prefill_target_state_contract(
        &self,
    ) -> Option<PersistentSpeculativePrefillTargetStateContract> {
        Some(PersistentSpeculativePrefillTargetStateContract::new(
            self.model_id.clone()?,
            self.model_revision.clone()?,
            self.speculative_prefill.draft_model_id.clone()?,
            self.speculative_prefill_draft_model_revision.clone()?,
            self.speculative_prefill_token_identifier_mapping_digest?,
            self.speculative_prefill.keep_percentage,
            self.speculative_prefill.selection_chunck_token_count,
            self.speculative_prefill.mandatory_trailing_token_count,
            self.speculative_prefill.lookahead_token_count,
            self.speculative_prefill
                .importance_pooling_kernel_token_count,
        ))
    }

    pub(super) fn speculative_prefill_selection_contract(
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
            self.speculative_prefill.selection_chunck_token_count,
            self.speculative_prefill.mandatory_trailing_token_count,
            self.speculative_prefill.lookahead_token_count,
            self.speculative_prefill
                .importance_pooling_kernel_token_count,
            position_tokens,
            u32::try_from(prompt_token_count).ok()?,
        ))
    }

    pub(super) fn speculative_prefill_store_key(
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
            selection_chunck_token_count: self.speculative_prefill.selection_chunck_token_count,
            mandatory_trailing_token_count: self.speculative_prefill.mandatory_trailing_token_count,
            lookahead_token_count: self.speculative_prefill.lookahead_token_count,
            importance_pooling_kernel_token_count: self
                .speculative_prefill
                .importance_pooling_kernel_token_count,
            position_tokens,
            token_ids,
        }
    }

    pub(super) fn store_speculative_prefill_selection(
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

    pub(super) fn restore_speculative_prefill_draft_prefix_checkpoint(
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

    pub(super) fn restore_longest_speculative_prefill_draft_prefix_checkpoint(
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

    pub(super) fn store_speculative_prefill_draft_prefix_checkpoint(
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
            && draft_prefix_store_key.selection_chunck_token_count
                == self.speculative_prefill.selection_chunck_token_count
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
