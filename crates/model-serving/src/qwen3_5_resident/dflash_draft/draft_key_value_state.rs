//! Per-layer append-only key-value state for the DFlash draft.
//!
//! The draft's six layers each own one append-only key-value state, the same
//! primitive the target's full-attention layers use. Sliding-window semantics
//! are enforced by the attention mask at read time, not by the storage: the
//! windowed layers keep the full context resident (bounded in practice by
//! memory governance) and simply do not attend beyond their window. This is
//! the deliberate storage choice for this integration — memory-bound but
//! simple, and the draft's cache is small next to the target's. The
//! ring-buffer state already in the tree is the measured follow-up for
//! long-context sessions: it would replace these states windowed-layer by
//! windowed-layer without changing any caller, once its logical truncate
//! primitive exists.
//!
//! Rollback of rejected speculation is `truncate_to_offset`, the same
//! transactional primitive the target's decode path uses.

use crate::decoder_cache::FullAttentionKeyValueState;
use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

/// Capacity growth granularity for draft key-value slabs, in tokens. The
/// draft is small enough that a coarse step wastes little and keeps slab
/// reallocation rare during steady decode.
const DRAFT_KEY_VALUE_STATE_GROWTH_TOKENS: i32 = 256;

/// The six per-layer key-value states plus one shared logical offset.
pub struct DFlashDraftKeyValueStates {
    layer_states: Vec<FullAttentionKeyValueState>,
}

impl DFlashDraftKeyValueStates {
    #[must_use]
    pub fn empty(layer_count: usize) -> Self {
        let layer_states = (0..layer_count)
            .map(|_| {
                FullAttentionKeyValueState::empty_with_validated_growth_tokens(
                    DRAFT_KEY_VALUE_STATE_GROWTH_TOKENS,
                )
            })
            .collect();
        Self { layer_states }
    }

    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layer_states.len()
    }

    /// The logical context length every layer agrees on.
    #[must_use]
    pub fn offset_tokens(&self) -> i32 {
        self.layer_states
            .first()
            .map_or(0, FullAttentionKeyValueState::offset_tokens)
    }

    pub fn layer_state_mut(
        &mut self,
        layer_index: usize,
    ) -> Option<&mut FullAttentionKeyValueState> {
        self.layer_states.get_mut(layer_index)
    }

    /// Rolls every layer back to one shared checkpoint offset after a
    /// speculation round whose suffix was rejected.
    pub fn truncate_to_offset(
        &mut self,
        checkpoint_offset_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        for layer_state in &mut self.layer_states {
            layer_state.truncate_to_offset(checkpoint_offset_tokens)?;
        }
        Ok(())
    }

    /// Total logical key-value payload bytes, for admission budgets.
    #[must_use]
    pub fn payload_byte_count(&self) -> u64 {
        self.layer_states
            .iter()
            .map(FullAttentionKeyValueState::payload_byte_count)
            .sum()
    }
}

impl Default for DFlashDraftKeyValueStates {
    fn default() -> Self {
        Self::empty(0)
    }
}
