//! Writing a reply in draft tree rounds with a hosted DFlash drafter: the
//! streaming engine's second drafter, beside the multi-token-prediction
//! head of `mtp_reply_drafting`.
//!
//! The two share one place in the decode step
//! (`advance_drafted_generation_if_admitted`), the queue discipline and the
//! emission: a round decides several reply tokens, every queued token but
//! the last has already run through the target, and the last starts the
//! next round. They differ in who proposes. Here the drafter reads the
//! target's hidden rows after a few layers (its taps), proposes a tree of
//! continuations, and the target verifies the whole tree in one pass
//! (`qwen3_5_core::draft_tree_decode`), keeping the chain its own choices
//! follow. Every emitted token is one the target chose itself.
//!
//! A reply drafts only when the drafter has rows to read. The taps are
//! captured while the prompt is read; a prompt restored whole from the
//! `astro_ssd_prompt_cache` leaves none, and that reply is written one
//! token a forward.

use std::collections::VecDeque;

use text_generation::generation_loop::decode_step::ActiveRequestAdvance;
use text_generation::generation_loop::generated_token_emission::generated_token_will_be_terminal;

use crate::InferenceEngineError;
use crate::qwen3_5_core::draft_tree_decode::draft_tree_round::{
    DraftSession, DraftTreeRoundRequest, run_draft_tree_round,
};
use crate::qwen3_5_core::draft_tree_decode::speculation_session_admission::verification_token_choice_for_strategy;

use super::Qwen3_5EngineState;
use super::draft_round_payoff::{DraftRoundPayoff, NextReplyStep};
use super::engine_request::StreamingEngineRequest;
use super::mtp_reply_drafting::{
    queued_or_pending_token, record_drafted_round, request_can_draft, target_token_choice_for,
};

/// The most rows of each tap kept from a prompt for the drafter. Taps are
/// about 50 MB a thousand tokens, held beside a memory limit the model
/// already fills, and a drafter whose context was capped at 2,048 rows kept
/// as many drafts as one with no cap (65.3 and 57.8 against 64.0 and 58.2
/// tokens a second on a first turn and a follow-up).
const STREAMED_DRAFT_TAP_ROW_LIMIT: usize = 4_096;

/// One reply's drafting with a hosted DFlash drafter.
pub(in crate::qwen3_5_streaming) struct DraftTreeReplyDrafting {
    session: DraftSession,
    /// Reply tokens decided and not yet emitted; the last has not run
    /// through the target.
    emission_queue: VecDeque<u32>,
    /// Whether this reply's rounds are paying (`draft_round_payoff`).
    payoff: DraftRoundPayoff,
    /// Reply tokens the last forward decided, for the payoff's count.
    tokens_decided_by_the_last_step: u32,
}

fn fatal(description: &'static str) -> InferenceEngineError {
    text_generation::engine_errors::fatal_engine_error(description)
}

impl Qwen3_5EngineState {
    /// Before a stretch of the prompt is read: when a drafter is hosted and
    /// the request's settings let rounds stand in for plain steps, opens
    /// the reply's drafting and asks the read to keep the drafter's taps.
    pub(super) fn begin_draft_taps_for_prompt_stretch(
        &self,
        active_request: &mut StreamingEngineRequest,
    ) {
        let Ok(model) = self.core.loaded_model() else {
            return;
        };
        let Some(draft_runtime) = model.optional_accelerators.draft_runtime.as_deref() else {
            return;
        };
        let is_held_back_by_a_contract = model
            .reply_measurement
            .writes_replies_without_the_hosted_drafter
            .get();
        if is_held_back_by_a_contract || !request_can_draft(active_request) {
            return;
        }
        if active_request
            .engine_extension
            .draft_tree_drafting
            .is_none()
        {
            active_request.engine_extension.draft_tree_drafting = Some(DraftTreeReplyDrafting {
                session: DraftSession::new(
                    target_token_choice_for(&active_request.sampling_strategy).keyed_seed,
                ),
                emission_queue: VecDeque::new(),
                payoff: DraftRoundPayoff::default(),
                tokens_decided_by_the_last_step: 0,
            });
        }
        let drafter_configuration = &draft_runtime.drafter_configuration;
        let row_limit = drafter_configuration
            .context_row_limit()
            .and_then(|context_row_limit| usize::try_from(context_row_limit).ok())
            .map_or(STREAMED_DRAFT_TAP_ROW_LIMIT, |context_row_limit| {
                context_row_limit.min(STREAMED_DRAFT_TAP_ROW_LIMIT)
            });
        model.begin_draft_tap_capture(&drafter_configuration.target_tap_layer_ids, row_limit);
    }

    /// After a stretch of the prompt was read: hands the taps it kept to
    /// the reply's drafter. A stretch that kept none, or whose taps could
    /// not be handed on, ends the reply's drafting: it is then written one
    /// token a forward, which is slower and never wrong, so reading the
    /// prompt does not fail for it.
    pub(super) fn hand_prompt_stretch_taps_to_the_drafter(
        &self,
        active_request: &mut StreamingEngineRequest,
    ) {
        if active_request
            .engine_extension
            .draft_tree_drafting
            .is_none()
        {
            return;
        }
        if let Err(tap_error) = self.append_prompt_stretch_taps(active_request) {
            tracing::warn!(%tap_error, "the drafter could not be given a prompt's rows");
            active_request.engine_extension.draft_tree_drafting = None;
        }
    }

    fn append_prompt_stretch_taps(
        &self,
        active_request: &mut StreamingEngineRequest,
    ) -> Result<(), InferenceEngineError> {
        let model = self.core.loaded_model()?;
        let draft_runtime = model
            .optional_accelerators
            .draft_runtime
            .as_deref()
            .ok_or_else(|| fatal("a reply opened drafting without a hosted drafter"))?;
        let Some((first_position, tap_rows)) = model.take_draft_taps()? else {
            active_request.engine_extension.draft_tree_drafting = None;
            return Ok(());
        };
        let Some(drafting) = active_request.engine_extension.draft_tree_drafting.as_mut() else {
            return Ok(());
        };
        drafting.session.append_tap_rows(
            model,
            draft_runtime.drafter_configuration.context_row_limit(),
            &tap_rows,
            first_position,
        )?;
        drafting.session.evaluate_pending_tap_rows(model)?;
        Ok(())
    }

    /// A drafted step when the reply drafts with a hosted drafter and this
    /// step allows it; `None` hands the step on.
    pub(super) fn advance_draft_tree_generation_if_admitted(
        &mut self,
        active_request: &mut StreamingEngineRequest,
        forced_thinking_transition_token_id: Option<u32>,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        let Some(drafting) = active_request.engine_extension.draft_tree_drafting.as_mut() else {
            return Ok(None);
        };
        if drafting.emission_queue.len() > 1 {
            let queued_token_id = drafting
                .emission_queue
                .pop_front()
                .ok_or_else(|| fatal("a drafted reply had no queued token to emit"))?;
            return self
                .emit_drafted_token(active_request, queued_token_id)
                .map(Some);
        }
        let can_run_a_round = drafting.payoff.writes_plain_tokens()
            || (forced_thinking_transition_token_id.is_none()
                && drafting.session.has_pending_tap_rows()
                && request_can_draft(active_request));
        if !can_run_a_round {
            return self
                .leave_draft_tree_drafting(active_request)
                .map(|()| None);
        }
        match self.next_step_of_the_reply(active_request)? {
            Some(NextReplyStep::Round) => {
                let Some(bonus_token_id) = next_token_to_run(active_request)? else {
                    return Ok(None);
                };
                self.run_draft_tree_round_step(active_request, bonus_token_id)
                    .map(Some)
            }
            Some(NextReplyStep::PlainToken) | None => Ok(None),
        }
    }

    /// What the reply's next forward should be, once it has a token to
    /// run; the payoff's count closes the step before it here. A reply
    /// that goes back to plain tokens keeps only its count: the token no
    /// forward has run becomes the plain step's pending token and the
    /// drafter's rows are released.
    fn next_step_of_the_reply(
        &self,
        active_request: &mut StreamingEngineRequest,
    ) -> Result<Option<NextReplyStep>, InferenceEngineError> {
        let model = self.core.loaded_model()?;
        let has_a_pending_token = active_request.pending_generated_token.is_some();
        let Some(drafting) = active_request.engine_extension.draft_tree_drafting.as_mut() else {
            return Ok(None);
        };
        if drafting.emission_queue.is_empty() && !has_a_pending_token {
            return Ok(None);
        }
        let wrote_plain_tokens = drafting.payoff.writes_plain_tokens();
        let tokens_decided_by_the_last_step = if wrote_plain_tokens {
            1
        } else {
            std::mem::take(&mut drafting.tokens_decided_by_the_last_step)
        };
        let next_reply_step = drafting.payoff.next_step(
            std::time::Instant::now(),
            tokens_decided_by_the_last_step,
            &model.reply_measurement.draft_round_costs,
        );
        if next_reply_step == NextReplyStep::PlainToken && !wrote_plain_tokens {
            tracing::info!(
                break_even_tokens_a_round = ?model
                    .reply_measurement
                    .draft_round_costs
                    .break_even_tokens_a_round(),
                "the reply's draft rounds decided too few tokens to pay; it goes on plain"
            );
            drafting.session = DraftSession::new(
                target_token_choice_for(&active_request.sampling_strategy).keyed_seed,
            );
            if let Some(unrun_token_id) = drafting.emission_queue.pop_front() {
                active_request.pending_generated_token =
                    Some(pending_token_array(model, unrun_token_id)?);
            }
        }
        Ok(Some(next_reply_step))
    }

    /// Ends the reply's drafting where the plain step can take over: only
    /// the token no forward has run is left, and it goes back to being the
    /// pending token the plain step reads, emits and runs.
    fn leave_draft_tree_drafting(
        &self,
        active_request: &mut StreamingEngineRequest,
    ) -> Result<(), InferenceEngineError> {
        let Some(mut drafting) = active_request.engine_extension.draft_tree_drafting.take() else {
            return Ok(());
        };
        if let Some(unrun_token_id) = drafting.emission_queue.pop_front() {
            active_request.pending_generated_token = Some(pending_token_array(
                self.core.loaded_model()?,
                unrun_token_id,
            )?);
        }
        Ok(())
    }

    /// Runs one round from `bonus_token_id`, queues what it decided and
    /// emits the bonus token.
    fn run_draft_tree_round_step(
        &mut self,
        active_request: &mut StreamingEngineRequest,
        bonus_token_id: u32,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        if generated_token_will_be_terminal(
            &self.core.end_of_sequence_token_ids,
            active_request,
            bonus_token_id,
        ) {
            // The reply ends with this token; nothing after it is needed.
            return self.emit_drafted_token(active_request, bonus_token_id);
        }
        let memory_baseline = self.begin_forward_memory_measurement()?;
        let model = self.core.loaded_model()?;
        active_request.reclaim_outgrown_attention_slabs(model.runtime())?;
        let draft_runtime = model
            .optional_accelerators
            .draft_runtime
            .as_deref()
            .ok_or_else(|| fatal("a draft tree round ran without a hosted drafter"))?;
        let mut drafting = active_request
            .engine_extension
            .draft_tree_drafting
            .take()
            .ok_or_else(|| fatal("a draft tree round ran without the reply's drafting"))?;
        let round_request = DraftTreeRoundRequest {
            bonus_token_id,
            context_length_tokens: active_request.next_position_tokens,
            target_token_choice: verification_token_choice_for_strategy(
                &active_request.sampling_strategy,
            )
            .ok_or_else(|| fatal("a draft tree round needs a sampling it can reproduce"))?,
        };
        // The anchor and every node of the tree are reply rows.
        model.begin_decode_astro_expert_cache_forward_of_rows(
            draft_runtime.tree_options.node_budget + 1,
        );
        let round_outcome = run_draft_tree_round(
            model,
            &mut active_request.request_decoder_state,
            draft_runtime,
            &mut drafting.session,
            &round_request,
        );
        model.end_decode_astro_expert_cache_forward();
        let round_outcome = round_outcome?;
        model
            .reply_measurement
            .stage_tally
            .record_draft_tree_round(&round_outcome.stages);
        // The bonus token is emitted now; the rest wait in the queue, the
        // last of them the target's own next choice, which no forward ran.
        let decided_token_ids = &round_outcome.emitted_token_ids[1..];
        active_request.advance_position(decided_token_ids.len())?;
        drafting
            .emission_queue
            .extend(decided_token_ids.iter().copied());
        drafting.tokens_decided_by_the_last_step = decided_token_ids.len() as u32;
        active_request.engine_extension.draft_tree_drafting = Some(drafting);
        record_drafted_round(
            active_request,
            round_outcome.drafted_token_count,
            round_outcome.accepted_proposal_count,
        );
        self.measure_round_and_emit(
            active_request,
            memory_baseline.active_memory_bytes,
            bonus_token_id,
        )
    }
}

/// A decided token as the plain step expects its pending token: `[1, 1]`.
fn pending_token_array(
    model: &crate::qwen3_5_streaming::model::Qwen3_5Model,
    token_id: u32,
) -> Result<mlx_bridge::MlxArray, InferenceEngineError> {
    model
        .runtime()
        .array_from_u32(&[token_id], &[1, 1])
        .map_err(text_generation::engine_errors::fatal_runtime_error)
}

/// The token the next round starts from: the queue's last, or the one the
/// prompt's read or a plain step left pending; `None` before the reply's
/// first token exists.
fn next_token_to_run(
    active_request: &mut StreamingEngineRequest,
) -> Result<Option<u32>, InferenceEngineError> {
    let queued_token_id = active_request
        .engine_extension
        .draft_tree_drafting
        .as_mut()
        .and_then(|drafting| drafting.emission_queue.pop_front());
    queued_or_pending_token(queued_token_id, active_request)
}
