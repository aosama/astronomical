//! Writing a reply with the model's multi-token-prediction head: the
//! streaming engine's drafted decode step.
//!
//! A round decides several reply tokens at once, and the decode step emits
//! one token a call, so the tokens a round decided wait in a queue. Every
//! queued token but the last has already run through the target; emitting
//! one is bookkeeping. The last is the target's own next choice, which no
//! forward has run yet: reaching it emits it and starts the next round,
//! whose verify forward runs it. The queue therefore never empties while
//! the reply drafts, and a reply that drafts never falls through to the
//! plain step with a token the target has not seen at the right position.
//!
//! Drafting starts from the reply token the plain step left pending, with
//! one forward that keeps the target's lanes for the head, and steps aside
//! for whatever the round cannot honour: a forced thinking transition, a
//! structured reply, a repetition penalty or image input.

use std::collections::VecDeque;

use text_generation::generation_loop::decode_step::ActiveRequestAdvance;
use text_generation::generation_loop::generated_token_emission::{
    build_generated_token_emission, generated_token_will_be_terminal,
    synchronize_generated_token_id,
};

use crate::qwen3_5_streaming::model::mtp_drafting::{
    MtpDraftingSession, MtpRoundStart, TargetTokenChoice, TopKTopPSampling,
};
use crate::{InferenceEngineError, TextSamplingStrategy};

use super::Qwen3_5EngineState;
use super::completed_forward_memory::measure_completed_reply_forward;
use super::engine_request::StreamingEngineRequest;

/// One reply's drafting.
pub(in crate::qwen3_5_streaming) struct MtpReplyDrafting {
    /// `None` until the first forward that keeps the target's lanes.
    session: Option<MtpDraftingSession>,
    /// Reply tokens decided and not yet emitted; the last has not run
    /// through the target.
    emission_queue: VecDeque<u32>,
    target_token_choice: TargetTokenChoice,
}

/// The request's sampling as the verify forward reproduces it.
fn target_token_choice_for(sampling_strategy: &TextSamplingStrategy) -> TargetTokenChoice {
    match sampling_strategy {
        TextSamplingStrategy::HighestLogit => TargetTokenChoice::highest_scoring(),
        TextSamplingStrategy::TopKTopP {
            temperature_thousandths,
            top_k,
            top_p_thousandths,
            seed,
        } => TargetTokenChoice {
            sampling: Some(TopKTopPSampling {
                temperature_thousandths: *temperature_thousandths,
                top_k: *top_k,
                top_p_thousandths: *top_p_thousandths,
            }),
            keyed_seed: crate::qwen3_5_core::text::resolve_sampling_seed(
                *seed,
                crate::sampling_seed::current_time_millis_since_unix_epoch,
            ),
        },
    }
}

impl Qwen3_5EngineState {
    /// A drafted step when the engine drafts and this step allows it;
    /// `None` hands the step to the plain path.
    pub(super) fn advance_mtp_drafted_generation_if_admitted(
        &mut self,
        active_request: &mut StreamingEngineRequest,
        forced_thinking_transition_token_id: Option<u32>,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        let maximum_draft_count = self.mtp_maximum_draft_count();
        if maximum_draft_count == 0 {
            return Ok(None);
        }
        let committed_tokens_are_queued = active_request
            .engine_extension
            .mtp_reply_drafting
            .as_ref()
            .is_some_and(|drafting| drafting.emission_queue.len() > 1);
        if committed_tokens_are_queued {
            return self.emit_queued_mtp_token(active_request).map(Some);
        }
        if forced_thinking_transition_token_id.is_some() || !request_can_draft(active_request) {
            // Only the token no forward has run is left, so the plain step
            // can take over at this position; drafting starts again from
            // the token it leaves pending.
            active_request.engine_extension.mtp_reply_drafting = None;
            return Ok(None);
        }
        let Some(last_chosen_token_id) = next_token_to_run(active_request)? else {
            return Ok(None);
        };
        self.run_mtp_step(active_request, last_chosen_token_id, maximum_draft_count)
            .map(Some)
    }

    /// The most tokens a round drafts: the configured count when the loaded
    /// model ships a head, otherwise zero.
    fn mtp_maximum_draft_count(&self) -> usize {
        self.core.model.as_ref().map_or(0, |model| {
            if model.mtp_head().is_some() {
                model.chunking.multi_token_prediction_draft_tokens as usize
            } else {
                0
            }
        })
    }

    /// Emits the next queued token, which already ran through the target.
    fn emit_queued_mtp_token(
        &mut self,
        active_request: &mut StreamingEngineRequest,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        let queued_token_id = active_request
            .engine_extension
            .mtp_reply_drafting
            .as_mut()
            .and_then(|drafting| drafting.emission_queue.pop_front())
            .ok_or_else(|| {
                text_generation::engine_errors::fatal_engine_error(
                    "a drafted reply had no queued token to emit",
                )
            })?;
        self.emit_mtp_token(active_request, queued_token_id)
    }

    fn emit_mtp_token(
        &mut self,
        active_request: &mut StreamingEngineRequest,
        reply_token_id: u32,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        Ok(build_generated_token_emission(
            self.core.loaded_model()?,
            &self.core.end_of_sequence_token_ids,
            active_request,
            reply_token_id,
            None,
        )?
        .into_active_request_advance())
    }

    /// Runs `last_chosen_token_id` through the target, with drafts after it
    /// once the head has lanes to start from, queues what the forward
    /// decided, and emits the token.
    fn run_mtp_step(
        &mut self,
        active_request: &mut StreamingEngineRequest,
        last_chosen_token_id: u32,
        maximum_draft_count: usize,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        if generated_token_will_be_terminal(
            &self.core.end_of_sequence_token_ids,
            active_request,
            last_chosen_token_id,
        ) {
            // The reply ends with this token; nothing after it is needed.
            return self.emit_mtp_token(active_request, last_chosen_token_id);
        }
        let memory_baseline = self.begin_forward_memory_measurement()?;
        let model = self.core.loaded_model()?;
        active_request.reclaim_outgrown_attention_slabs(model.runtime())?;
        let mut drafting = active_request
            .engine_extension
            .mtp_reply_drafting
            .take()
            .unwrap_or_else(|| MtpReplyDrafting {
                session: None,
                emission_queue: VecDeque::new(),
                target_token_choice: target_token_choice_for(&active_request.sampling_strategy),
            });
        model.begin_decode_astro_expert_cache_forward();
        let step_outcome = run_target_step(
            model,
            active_request,
            &mut drafting,
            last_chosen_token_id,
            maximum_draft_count,
        );
        model.end_decode_astro_expert_cache_forward();
        let decided_token_ids = step_outcome?;
        active_request.advance_position(decided_token_ids.len())?;
        drafting.emission_queue.extend(decided_token_ids);
        active_request.engine_extension.mtp_reply_drafting = Some(drafting);
        measure_completed_reply_forward(
            model,
            active_request,
            memory_baseline.active_memory_bytes,
        )?;
        self.emit_mtp_token(active_request, last_chosen_token_id)
    }
}

/// Whether the request's own settings let a round stand in for plain steps.
fn request_can_draft(active_request: &StreamingEngineRequest) -> bool {
    active_request.structured_generation.is_none()
        && active_request.repetition_penalty_thousandths == 1_000
        && !active_request.has_visual_inputs
}

/// The token the next forward runs: the queue's last, or the one the plain
/// step left pending; `None` before the reply's first token exists.
fn next_token_to_run(
    active_request: &mut StreamingEngineRequest,
) -> Result<Option<u32>, InferenceEngineError> {
    let queued_token_id = active_request
        .engine_extension
        .mtp_reply_drafting
        .as_mut()
        .and_then(|drafting| drafting.emission_queue.pop_front());
    if queued_token_id.is_some() {
        return Ok(queued_token_id);
    }
    let Some(pending_generated_token) = active_request.pending_generated_token.take() else {
        return Ok(None);
    };
    synchronize_generated_token_id(active_request, &pending_generated_token).map(Some)
}

/// One target forward from `last_chosen_token_id`: the first keeps the
/// target's lanes and starts the head, every later one is a drafted round.
/// Returns the tokens it decided, the last of which has not run yet; their
/// count is also how many rows the target kept.
fn run_target_step(
    model: &crate::qwen3_5_streaming::model::Qwen3_5Model,
    active_request: &mut StreamingEngineRequest,
    drafting: &mut MtpReplyDrafting,
    last_chosen_token_id: u32,
    maximum_draft_count: usize,
) -> Result<Vec<u32>, InferenceEngineError> {
    let starting_position_tokens = active_request.next_position_tokens;
    let Some(session) = drafting.session.as_mut() else {
        let target_forward_output = model
            .forward_chunk_with_pre_final_normalization_hidden_states(
                &[last_chosen_token_id],
                starting_position_tokens,
                &mut active_request.request_decoder_state,
            )?;
        let lanes = target_forward_output.pre_final_normalization_hidden_states();
        let next_token_ids = model.target_token_picks(
            &model.lane_mixed_vocabulary_logits_for_every_row(lanes)?,
            starting_position_tokens,
            drafting.target_token_choice,
        )?;
        drafting.session = Some(model.begin_mtp_drafting(
            &[last_chosen_token_id],
            lanes,
            next_token_ids[0],
            &mut active_request.performance_attribution,
        )?);
        return Ok(next_token_ids);
    };
    let round_outcome = model.advance_mtp_drafting_round(
        session,
        MtpRoundStart {
            last_chosen_token_id,
            starting_position_tokens,
            maximum_draft_count,
        },
        &mut active_request.request_decoder_state,
        drafting.target_token_choice,
        &mut active_request.performance_attribution,
    )?;
    let attribution = &mut active_request.performance_attribution;
    attribution.record_counter(crate::PerformanceCounter::SpeculativeRoundCount, 1);
    attribution.record_counter(
        crate::PerformanceCounter::SpeculativeProposedTokenCount,
        round_outcome.drafted_token_count as u64,
    );
    attribution.record_counter(
        crate::PerformanceCounter::SpeculativeAcceptedTokenCount,
        round_outcome.accepted_draft_count as u64,
    );
    Ok(round_outcome.reply_token_ids)
}
