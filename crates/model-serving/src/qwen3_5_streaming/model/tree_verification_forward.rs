//! The streaming model as a draft tree's target: a layer's feed-forward over
//! the tree's rows finds its experts through the `astro_expert_cache` and the
//! pager, as a prompt span of that many tokens does, so a drafted round runs
//! whether a layer's experts are all in memory or partly on storage. The pass
//! itself is the shared core's (`draft_tree_verification`), the same one the
//! resident model runs.

use mlx_bridge::MlxArray;

use super::streaming_model::Qwen3_5StreamingModel;
use crate::qwen3_5_core::model::draft_tree_verification::{
    DraftTreeTarget, Qwen3_5DraftTreeVerification, commit_draft_tree, verify_draft_tree,
};
use crate::qwen3_5_core::model::{Qwen3_5DecoderLayerAttentionOutput, Qwen3_5ModelBase};
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::Qwen3_5MoEPagedExecutionMode;
use crate::speculative_decoding::tree_rows::TreeRows;
use crate::{PerformanceAttribution, RequestDecoderStateStack};

impl DraftTreeTarget for Qwen3_5StreamingModel {
    fn draft_tree_model_base(&self) -> &Qwen3_5ModelBase {
        self
    }

    fn forward_tree_rows_feed_forward(
        &self,
        attention_output: &Qwen3_5DecoderLayerAttentionOutput,
        row_count: i32,
        layer_index: usize,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        self.forward_decoder_layer_feed_forward(
            attention_output,
            row_count,
            layer_index,
            decoder_layer_weights,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            &mut PerformanceAttribution::disabled(),
        )
    }
}

impl Qwen3_5StreamingModel {
    /// Verifies `tree_rows`, whose anchor sits at `anchor_position` (every
    /// token before it is already in `request_decoder_state`).
    pub fn verify_draft_tree(
        &self,
        tree_rows: &TreeRows,
        anchor_position: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        target_hidden_state_tap_layer_ids: &[i32],
    ) -> Result<Qwen3_5DraftTreeVerification, Qwen3_5ExecutionError> {
        verify_draft_tree(
            self,
            tree_rows,
            anchor_position,
            request_decoder_state,
            target_hidden_state_tap_layer_ids,
        )
    }

    /// Advances every layer over `kept_rows` (tree rows, anchor first), as if
    /// only that chain had been fed after the context.
    pub fn commit_draft_tree(
        &self,
        request_decoder_state: &mut RequestDecoderStateStack,
        verification: Qwen3_5DraftTreeVerification,
        kept_rows: &[usize],
    ) -> Result<(), Qwen3_5ExecutionError> {
        commit_draft_tree(self, request_decoder_state, verification, kept_rows)
    }
}
