//! The streaming model as a draft tree's target.
//!
//! A layer's feed-forward over the tree's rows finds its experts through
//! the `astro_expert_cache`, as a reply token's does: the rows are reply
//! rows, so a layer held in part reads only the experts its table lacks,
//! into the table. The pass itself is the shared core's
//! (`draft_tree_verification`), the same one the resident model runs.
//!
//! The pass is tried first with each partly held layer looking its routes
//! up inside the graph (`in_graph_table_serving`), so a round is one
//! evaluation however many layers are held in part; a tree's rows seldom
//! share experts, and stopping at the host once a layer is what made
//! drafted rounds lose on a streamed model before. A pass a lookup refuses
//! is run again the exact way from the state it began with.

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use super::streaming_model::Qwen3_5StreamingModel;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model::draft_tree_verification::{
    DraftTreeTarget, Qwen3_5DraftTreeVerification, TreeLayerFeedForward, TreeLayerWeights,
};
use crate::qwen3_5_streaming::{PagedRouteValidationOutcome, Qwen3_5MoEPagedExecutionMode};
use crate::{PerformanceAttribution, RequestDecoderStateStack};
use engine_core::speculative_decoding::tree_rows::TreeRows;

impl DraftTreeTarget for Qwen3_5StreamingModel {
    fn draft_tree_model_base(&self) -> &Qwen3_5ModelBase {
        self
    }

    /// A sparse model's layers are bound; a dense model's come from its
    /// streamer, kept or read from the SSD for this pass.
    fn tree_layer_weights(
        &self,
        layer_index: usize,
    ) -> Result<Box<dyn TreeLayerWeights + '_>, TextExecutionError> {
        Ok(Box::new(self.forward_layer_weights(layer_index)?))
    }

    /// Each row finds its experts as a reply token does.
    fn tree_layer_feed_forward(
        &self,
        feed_forward: TreeLayerFeedForward<'_>,
    ) -> Result<MlxArray, TextExecutionError> {
        let TreeLayerFeedForward {
            attention_output,
            row_count,
            layer_index,
            decoder_layer_weights,
        } = feed_forward;
        self.forward_decoder_layer_feed_forward(
            attention_output,
            row_count,
            layer_index,
            decoder_layer_weights,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            &mut PerformanceAttribution::disabled(),
        )
    }

    fn tree_per_row_logits(
        &self,
        hidden_states: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        self.normalized_vocabulary_logits_for_every_row(hidden_states)
    }

    /// A streamed pass is one evaluation: a layer group started while the
    /// next is built would run its lookups while this thread reads the same
    /// tables (`in_graph_route_lookup`).
    fn starts_tree_layers_asynchronously(&self) -> bool {
        false
    }
}

impl Qwen3_5StreamingModel {
    /// Verifies `tree_rows` and evaluates the result: the logits after
    /// every row and the requested taps are ready to read when this
    /// returns, and every route the pass took was served from its table.
    pub fn verify_draft_tree_resolving_expert_routes(
        &self,
        tree_rows: &TreeRows,
        anchor_position: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        target_hidden_state_tap_layer_ids: &[i32],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5DraftTreeVerification, TextExecutionError> {
        self.serve_reply_rows_resolving_expert_routes(
            tree_rows.row_count(),
            request_decoder_state,
            |request_decoder_state| {
                self.verify_and_evaluate_draft_tree(
                    tree_rows,
                    anchor_position,
                    request_decoder_state,
                    target_hidden_state_tap_layer_ids,
                    performance_attribution,
                )
            },
        )
    }

    /// One pass and its one evaluation. `None` when an in-graph lookup
    /// refused a layer, so the pass's outputs are wrong.
    fn verify_and_evaluate_draft_tree(
        &self,
        tree_rows: &TreeRows,
        anchor_position: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        target_hidden_state_tap_layer_ids: &[i32],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<Qwen3_5DraftTreeVerification>, TextExecutionError> {
        // The rows are reply tokens, whatever their number.
        self.paged_forward_missing_route_collector
            .set_serves_reply_rows(true);
        let verification = self.verify_draft_tree(
            tree_rows,
            anchor_position,
            request_decoder_state,
            target_hidden_state_tap_layer_ids,
        );
        self.paged_forward_missing_route_collector
            .set_serves_reply_rows(false);
        let verification = verification?;
        let mut evaluation_arrays = Vec::with_capacity(1 + verification.hidden_state_taps.len());
        evaluation_arrays.push(&verification.per_row_logits);
        evaluation_arrays.extend(verification.hidden_state_taps.iter());
        let validation_outcome = self
            .evaluate_arrays_resolving_paged_routes(&evaluation_arrays, performance_attribution)?;
        drop(evaluation_arrays);
        Ok(match validation_outcome {
            PagedRouteValidationOutcome::CompleteHit => Some(verification),
            PagedRouteValidationOutcome::MissedExperts => None,
        })
    }
}
