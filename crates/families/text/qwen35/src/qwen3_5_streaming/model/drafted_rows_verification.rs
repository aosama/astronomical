//! Verifying a drafted block on the streamed Flash-Next target, and keeping
//! only the rows the target agrees with.
//!
//! One forward runs the last chosen token and the drafts after it. Shared
//! experts are read from storage once for the whole block and every layer
//! is built once, which is where drafting saves time on a streamed model.
//! The forward leaves every row in the request's state, so it also captures
//! what a rewind needs: each linear-attention layer's state after every row
//! but the last, and the Flash-Next state before the forward. Keeping fewer
//! rows then moves the state back without running the target again.

use mlx_bridge::MlxArray;
use text_generation::AstroSsdPromptCacheBoundaryCheckpoint;
use text_generation::decoder_state::family_decoder_state::FamilyDecoderState;
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::model::target_forward_graph::Qwen3_5TargetForwardGraph;
use crate::qwen3_5_core::model_math::forward_contract::decoder_state_arrays;
use crate::qwen3_5_streaming::{PagedRouteValidationOutcome, Qwen3_5MoEPagedExecutionMode};
use crate::qwen4_exp::drafted_rows_rewind::KeptDraftedRows;
use crate::qwen4_exp::request_state::Qwen4ExpRequestState;
use crate::{PerformanceAttribution, RequestDecoderStateStack};

use super::Qwen3_5Model;

/// One verified block: what the target says after every row, and what
/// rewinding to fewer rows needs.
pub struct DraftedRowsVerification {
    /// `[1, rows, vocabulary]` float32: the target's logits after each row.
    pub per_row_logits: MlxArray,
    /// `[1, rows, lanes * hidden]`: the target's final lanes after each row,
    /// which the draft head continues from.
    pub lanes: MlxArray,
    /// Entry `n - 1` holds the linear-attention state after the first `n` rows.
    row_boundaries: Vec<AstroSsdPromptCacheBoundaryCheckpoint>,
    family_state_before: Option<Box<dyn FamilyDecoderState>>,
    token_ids: Vec<u32>,
    starting_position_tokens: u32,
}

/// What one forward of a drafted block produced, before it is known to keep.
struct ForwardedDraftedRows {
    per_row_logits: MlxArray,
    lanes: MlxArray,
    row_boundaries: Vec<AstroSsdPromptCacheBoundaryCheckpoint>,
}

impl Qwen3_5Model {
    /// Forwards `token_ids` (the last chosen token, then the drafts) from
    /// `starting_position_tokens` and evaluates the result.
    ///
    /// The block is tried first with each partly held layer looking its
    /// routes up inside the graph, as a single reply token is
    /// (`in_graph_table_serving`): a block's rows seldom share experts, so
    /// stopping at the host once a layer cost a round more than it cost a
    /// token. A block a lookup refuses is forwarded again the exact way.
    pub fn verify_drafted_rows(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<DraftedRowsVerification, TextExecutionError> {
        if token_ids.len() < 2 {
            return Err(TextExecutionError::InvalidInput {
                description: "a drafted block verifies the last chosen token and at least one draft",
            });
        }
        let family_state_before = request_decoder_state.family_state_snapshot()?;
        let forwarded_rows = self.serve_reply_rows_resolving_expert_routes(
            token_ids.len(),
            request_decoder_state,
            |request_decoder_state| {
                self.forward_drafted_rows(
                    token_ids,
                    starting_position_tokens,
                    request_decoder_state,
                    performance_attribution,
                )
            },
        )?;
        Ok(DraftedRowsVerification {
            per_row_logits: forwarded_rows.per_row_logits,
            lanes: forwarded_rows.lanes,
            row_boundaries: forwarded_rows.row_boundaries,
            family_state_before,
            token_ids: token_ids.to_vec(),
            starting_position_tokens,
        })
    }

    /// One forward of the block and its one evaluation. `None` when an
    /// in-graph lookup refused a layer, so the block's outputs are wrong.
    fn forward_drafted_rows(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<ForwardedDraftedRows>, TextExecutionError> {
        let mut row_boundary_collector = self.drafted_row_boundary_collector(token_ids.len())?;
        // The rows are reply tokens, whatever their number.
        self.paged_forward_missing_route_collector
            .set_serves_reply_rows(true);
        let target_forward_graph = self.build_target_forward_graph(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            Some(&mut row_boundary_collector),
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            performance_attribution,
        );
        self.paged_forward_missing_route_collector
            .set_serves_reply_rows(false);
        let target_forward_output = target_forward_graph?;
        let lanes = target_forward_output
            .pre_final_normalization_hidden_states()
            .retain()?;
        let per_row_logits = self.lane_mixed_vocabulary_logits_for_every_row(&lanes)?;
        // The logits and the lanes join the state's evaluation: a second
        // evaluation would be a second wait every round, and the last
        // layer's lookup, which only they depend on, has to run before the
        // forward is judged.
        let mut evaluation_arrays = decoder_state_arrays(request_decoder_state)?;
        evaluation_arrays.extend(row_boundary_collector.evaluation_arrays());
        evaluation_arrays.push(&per_row_logits);
        evaluation_arrays.push(&lanes);
        let validation_outcome = self
            .evaluate_arrays_resolving_paged_routes(&evaluation_arrays, performance_attribution)?;
        drop(evaluation_arrays);
        if validation_outcome == PagedRouteValidationOutcome::MissedExperts {
            return Ok(None);
        }
        Ok(Some(ForwardedDraftedRows {
            per_row_logits,
            lanes,
            row_boundaries: row_boundary_collector.complete()?,
        }))
    }

    /// Leaves the request's state as if only the first `kept_row_count` rows
    /// of the verified block had been forwarded.
    pub fn keep_drafted_rows(
        &self,
        verification: DraftedRowsVerification,
        kept_row_count: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<(), TextExecutionError> {
        let forwarded_row_count = verification.token_ids.len();
        if kept_row_count == 0 || kept_row_count > forwarded_row_count {
            return Err(TextExecutionError::InvalidInput {
                description: "a drafted block keeps at least its first row and at most all of them",
            });
        }
        if kept_row_count == forwarded_row_count {
            return Ok(());
        }
        let DraftedRowsVerification {
            mut row_boundaries,
            family_state_before,
            token_ids,
            starting_position_tokens,
            ..
        } = verification;
        let kept_token_offset = i32::try_from(starting_position_tokens as usize + kept_row_count)
            .map_err(|_| TextExecutionError::InvalidInput {
            description: "the kept rows end beyond the MLX integer range",
        })?;
        let kept_row_boundary =
            row_boundaries
                .get_mut(kept_row_count - 1)
                .ok_or(TextExecutionError::InvalidInput {
                    description: "the verified block captured no boundary for its kept rows",
                })?;
        request_decoder_state.rewind_to_kept_row_boundary(
            &mut kept_row_boundary.recurrent_snapshot_tensors,
            kept_token_offset,
        )?;
        self.rewind_flash_next_state_to_kept_rows(
            family_state_before.as_deref(),
            &token_ids[..kept_row_count],
            forwarded_row_count,
            request_decoder_state,
        )
    }

    fn rewind_flash_next_state_to_kept_rows(
        &self,
        family_state_before: Option<&dyn FamilyDecoderState>,
        kept_token_ids: &[u32],
        forwarded_row_count: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<(), TextExecutionError> {
        let missing_state = TextExecutionError::InvalidInput {
            description: "a verified Flash-Next block has the Flash-Next state on both sides",
        };
        let (Some(architecture), Some(ngram_table)) = (
            self.config.qwen4_exp_architecture(),
            self.ngram_table.as_ref(),
        ) else {
            return Err(missing_state);
        };
        let Some(state_before) = family_state_before
            .and_then(|family_state| family_state.as_any().downcast_ref::<Qwen4ExpRequestState>())
        else {
            return Err(missing_state);
        };
        let kept_ngram_context = ngram_table
            .hash_tables()
            .next_context(&state_before.ngram_context, kept_token_ids);
        let Some(state_after) = request_decoder_state.family_state_mut::<Qwen4ExpRequestState>()
        else {
            return Err(missing_state);
        };
        state_after.rewind_to_kept_drafted_rows(
            &self.runtime,
            state_before,
            kept_ngram_context,
            KeptDraftedRows {
                forwarded_row_count: forwarded_row_count as i32,
                kept_row_count: kept_token_ids.len() as i32,
                tokens_per_indexer_block: architecture.indexer_compress_ratio as i32,
            },
        )
    }
}
