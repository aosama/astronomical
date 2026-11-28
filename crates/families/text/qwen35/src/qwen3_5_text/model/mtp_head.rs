//! The multi-token-prediction head on the engine: the state a request keeps
//! for it, and the forward that drafts with whichever kind the checkpoint
//! ships.
//!
//! The head drafts the tokens after the one the target just chose. Given a
//! token and the target's state at the position before it, it returns
//! logits for the token after it and its own state, which stands in for the
//! target's when it drafts one token further. It keeps its own attention
//! keys and values, in positions that count the head's own rows.
//! Flash-Next's head (`flash_next_mtp_head.rs`) reads the target's residual
//! lanes; a dense Qwen 3.5 head (`dense_mtp_head.rs`) reads its normalized
//! final hidden state.

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntimeError;
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::model_math::mtp_head_kind::{MtpHead, resolved_mtp_draft_token_count};
use crate::qwen3_5_text::model::Qwen3_5Model;
use crate::qwen4_exp::request_state::IndexerLayerState;
use crate::{FullAttentionKeyValueState, PerformanceAttribution};

/// One request's memory of the head: what its attention has seen.
pub struct MtpHeadState {
    pub(super) key_value_state: FullAttentionKeyValueState,
    pub(super) indexer_state: Option<IndexerLayerState>,
}

impl MtpHeadState {
    pub fn empty(full_attention_kv_state_growth_tokens: i32) -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            key_value_state: FullAttentionKeyValueState::empty_with_growth_tokens(
                full_attention_kv_state_growth_tokens,
            )?,
            indexer_state: None,
        })
    }

    /// Rows the head has forwarded and kept.
    #[must_use]
    pub fn row_count(&self) -> i32 {
        self.key_value_state.offset_tokens()
    }

    /// Forgets every row after the first `kept_row_count`: the rows a draft
    /// added on the head's own guesses, before the target's lanes for the
    /// kept tokens replace them.
    pub fn truncate_rows(
        &mut self,
        kept_row_count: i32,
        tokens_per_indexer_block: i32,
    ) -> Result<(), MlxRuntimeError> {
        self.key_value_state.truncate_to_offset(kept_row_count)?;
        if let Some(indexer_state) = self.indexer_state.as_mut() {
            indexer_state.keys.truncate_rows(kept_row_count);
            if let Some(block_keys) = indexer_state.block_keys.as_mut() {
                block_keys.truncate_rows(kept_row_count / tokens_per_indexer_block.max(1));
            }
        }
        Ok(())
    }

    /// The arrays a forward leaves to evaluate with its outputs.
    pub fn append_evaluation_arrays<'state>(&'state self, arrays: &mut Vec<&'state MlxArray>) {
        arrays.extend(self.key_value_state.keys_state());
        arrays.extend(self.key_value_state.values_state());
        if let Some(indexer_state) = &self.indexer_state {
            arrays.push(indexer_state.keys.storage());
            if let Some(block_keys) = &indexer_state.block_keys {
                arrays.push(block_keys.storage());
            }
        }
    }
}

/// What one head forward returns, both lazy.
pub struct MtpHeadOutput {
    /// `[1, tokens, vocabulary]` float32 logits for the token after each
    /// input token.
    pub logits: MlxArray,
    /// The stand-in for the target's state one draft further: Flash-Next's
    /// head returns its lanes before its final mixer (`[1, tokens, lanes *
    /// hidden]`), a dense head its normalized hidden state (`[1, tokens,
    /// hidden]`).
    pub states: MlxArray,
}

impl Qwen3_5Model {
    /// The checkpoint's head, when it ships one.
    #[must_use]
    pub fn mtp_head(&self) -> Option<&MtpHead> {
        self.optional_accelerators.mtp_head.as_ref()
    }

    /// The most tokens a round drafts: the configured count, or the head's
    /// own when the configuration leaves it open; none without a head.
    #[must_use]
    pub fn mtp_draft_token_count(&self) -> u32 {
        resolved_mtp_draft_token_count(
            self.chunking.multi_token_prediction_draft_tokens,
            self.mtp_head().map(MtpHead::kind),
        )
    }

    /// Sets `reserve_bytes` aside for a drafter that is loading and plans
    /// the kept weights around it, giving back what no longer fits; zero
    /// ends the reservation.
    pub fn reserve_room_for_a_loading_drafter(&self, reserve_bytes: u64) {
        self.governed_memory
            .drafter_load_reserve_bytes
            .set(reserve_bytes);
        self.open_memory_governor();
    }

    /// The model's geometry as the memory ledger should see it: while
    /// replies draft, the head's weights are core weights, always in memory,
    /// and the `astro_expert_cache` gets what is left after them.
    #[must_use]
    pub fn geometry_with_drafting_head(
        &self,
        mut model_geometry: crate::ModelMemoryGeometry,
    ) -> crate::ModelMemoryGeometry {
        if self.mtp_draft_token_count() > 0
            && let Some(head) = self.mtp_head()
        {
            model_geometry.model_core_payload_bytes = model_geometry
                .model_core_payload_bytes
                .saturating_add(head.payload_byte_count());
        }
        // A hosted DFlash drafter never pages either; one that is loading
        // is counted by the room set aside for it.
        let drafter_bytes = match self.optional_accelerators.draft_runtime.as_deref() {
            Some(draft_runtime) => draft_runtime.hosted_weight_bytes,
            None => self.governed_memory.drafter_load_reserve_bytes.get(),
        };
        model_geometry.model_core_payload_bytes = model_geometry
            .model_core_payload_bytes
            .saturating_add(drafter_bytes);
        model_geometry
    }

    /// Runs the head over `token_indices` (`[1, tokens]` int32), each paired
    /// with the state at the position before it (`preceding_states`, one
    /// row a token): the target's rows as `target_rows_for_head` gives them,
    /// or the head's own from a previous draft.
    pub fn forward_mtp_head(
        &self,
        head: &MtpHead,
        token_indices: &MlxArray,
        preceding_states: &MlxArray,
        head_state: &mut MtpHeadState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MtpHeadOutput, TextExecutionError> {
        let token_count = token_indices.shape().get(1).copied().unwrap_or(0);
        if token_count == 0 || preceding_states.shape().get(1).copied() != Some(token_count) {
            return Err(TextExecutionError::InvalidInput {
                description: "the head needs one row of state for each of its tokens",
            });
        }
        match head {
            MtpHead::FlashNext(flash_next_head) => self.forward_flash_next_mtp_head(
                flash_next_head,
                token_indices,
                preceding_states,
                head_state,
                performance_attribution,
            ),
            MtpHead::DenseQwen3_5(dense_head) => {
                self.forward_dense_mtp_head(dense_head, token_indices, preceding_states, head_state)
            }
        }
    }

    /// The target's rows (`[1, rows, width]`, before its final norm, as a
    /// forward or a verify pass leaves them) as the head reads them:
    /// Flash-Next's head reads the lanes as they are; a dense head reads the
    /// state after the final norm, the one the output head reads.
    pub fn target_rows_for_head(
        &self,
        head: &MtpHead,
        target_rows: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        match head {
            MtpHead::FlashNext(_) => Ok(target_rows.retain()?),
            MtpHead::DenseQwen3_5(_) => self.final_normalized_states(target_rows),
        }
    }
}
