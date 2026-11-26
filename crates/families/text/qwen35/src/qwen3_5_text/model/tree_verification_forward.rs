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

use super::streaming_model::Qwen3_5TextModel;
use crate::qwen3_5_core::configuration::Qwen3_5FeedForwardArchitecture;
use crate::qwen3_5_core::draft_tree_decode::draft_round_target::{DraftRoundTarget, RoundTree};
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model::draft_tree_verification::{
    DraftTreeTarget, Qwen3_5DraftTreeVerification, TreeLayerFeedForward, TreeLayerWeights,
};
use crate::qwen3_5_text::{PagedRouteValidationOutcome, Qwen3_5MoEPagedExecutionMode};
use crate::{PerformanceAttribution, RequestDecoderStateStack};
use engine_core::speculative_decoding::tree_rows::TreeRows;

impl DraftTreeTarget for Qwen3_5TextModel {
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

    /// A mixture-of-experts pass is one evaluation: a layer group started
    /// while the next is built would run its lookups while this thread reads
    /// the same tables (`in_graph_route_lookup`). A dense model's layers run
    /// no host step inside the graph, so the GPU starts on each few layers
    /// while the next are built: on the dense 27B with every layer kept,
    /// building the 64-layer pass before the GPU began cost a drafted round
    /// about 8 ms of its 96. A measurement that profiles the pass asks for
    /// it whole, since a profiler sees only the graph it is handed.
    fn starts_tree_layers_asynchronously(&self) -> bool {
        self.config.feed_forward_architecture() == Qwen3_5FeedForwardArchitecture::Dense
            && !self
                .optional_accelerators
                .tree_pass_built_whole_for_measurement
    }
}

impl Qwen3_5TextModel {
    /// Asks the forwards that follow to keep the hidden rows after
    /// `tap_layer_ids` for a hosted drafter, at most the newest `row_limit`
    /// a tap; `take_draft_taps` collects them.
    pub fn begin_draft_tap_capture(&self, tap_layer_ids: &[i32], row_limit: usize) {
        self.paged_forward_missing_route_collector
            .draft_tap_capture
            .begin(tap_layer_ids, row_limit);
    }

    /// The rows kept since `begin_draft_tap_capture`: the position of the
    /// first, and one `[1, rows, hidden]` array a tap. `None` when none
    /// were asked for or a tap fell short.
    pub fn take_draft_taps(&self) -> Result<Option<(i32, Vec<MlxArray>)>, TextExecutionError> {
        let Some(captured_taps) = self
            .paged_forward_missing_route_collector
            .draft_tap_capture
            .take()
        else {
            return Ok(None);
        };
        let mut tap_rows = Vec::with_capacity(captured_taps.span_rows_by_tap.len());
        for span_rows in &captured_taps.span_rows_by_tap {
            let span_row_references: Vec<&MlxArray> = span_rows.iter().collect();
            tap_rows.push(self.runtime.concatenate_axis(&span_row_references, 1)?);
        }
        Ok(Some((captured_taps.first_position, tap_rows)))
    }

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

impl DraftRoundTarget for Qwen3_5TextModel {
    /// A streamed round's tree is evaluated where it is verified: its routes
    /// are resolved in that evaluation, and a refused pass is run again.
    fn verify_round_tree(
        &self,
        round_tree: RoundTree<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Qwen3_5DraftTreeVerification, TextExecutionError> {
        self.verify_draft_tree_resolving_expert_routes(
            round_tree.tree_rows,
            round_tree.anchor_position,
            request_decoder_state,
            round_tree.tap_layer_ids,
            &mut PerformanceAttribution::disabled(),
        )
    }

    /// Declares the tree kernels and converts a dense model's output head
    /// and the layers kept now to register order; a layer kept later
    /// converts as it is read. The tree kernels are what make a drafted
    /// round possible, so a model with nothing kept, or a sparse one, still
    /// drafts with 0 converted.
    fn prepare_for_draft_trees(
        &mut self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<usize>, TextExecutionError> {
        let Some(converted_projection_count) =
            self.convert_projections_to_register_order(performance_attribution)?
        else {
            return Ok(None);
        };
        self.enable_draft_tree_verification()?;
        self.enable_row_exact_tree_attention()?;
        Ok(Some(converted_projection_count))
    }

    /// The kept layers and experts fill the limit by the time a drafter
    /// loads, so the room is never already there: a dense 27B at a 12 GB
    /// limit held 11.1 GB and its drafter, which writes code 2.7 times
    /// faster where the model streams, was refused at every limit under
    /// 18 GB. The governor plans the kept weights around the drafter
    /// instead and they are given back to make the room.
    fn make_room_for_a_drafter(&mut self, drafter_load_bytes: u64) -> bool {
        self.reserve_room_for_a_loading_drafter(drafter_load_bytes);
        true
    }
}
