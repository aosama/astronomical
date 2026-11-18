//! Draft rounds in the streaming engine's decode loop.
//!
//! A drafting session advances one token a call like any other, but a call
//! that finds its queue empty runs one draft round (the shared core's
//! `run_draft_tree_round`): the drafter proposes a tree, the target verifies
//! every row in one pass through the `astro_expert_cache` and the pager, and
//! the chain the target's own picks follow is committed. The round's tokens
//! then reach the caller one an advance, with no forward in between.
//!
//! A round is measured as one forward over its rows, so the memory governor
//! learns the working buffers a round needs and keeps room for them, as it
//! does for a plain reply step.

use astronomical_ipc_protocol::RequestId;

use super::advance_generation::ActiveRequestAdvance;
use super::completed_forward_memory::{
    CompletedForwardMemoryObservation, MeasuredForward,
    capture_completed_forward_memory_observation, measure_completed_forward,
};
use super::engine_request::Qwen3_5EngineRequest;
use super::generated_token_emission;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::InferenceEngineError;
use crate::qwen3_5_core::draft_tree_decode::draft_tree_round::{
    DraftTreeRoundOutcome, DraftTreeRoundRequest, run_draft_tree_round,
};
use crate::qwen3_5_core::draft_tree_decode::speculation_session_admission::verification_token_choice_for_strategy;

impl Qwen3_5EngineState {
    /// One drafting advance: emits one queued committed token, or, at an
    /// empty queue, runs one draft round and emits its first token.
    pub(super) fn advance_speculative_generation(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        // The queue holds committed tokens in front and one uncommitted token
        // at its tail: the last round's own next pick, which the next
        // round's verify commits as its anchor.
        let draft_request_state = &mut active_request.engine_extension.draft_request_state;
        if draft_request_state.emission_queue.len() > 1 {
            let committed_token_id = draft_request_state
                .emission_queue
                .pop_front()
                .ok_or_else(|| fatal_engine_error("the drafting queue emptied between checks"))?;
            return self.emit_drafted_token(active_request, committed_token_id, None);
        }
        let bonus_token_id = match draft_request_state.emission_queue.pop_front() {
            Some(last_round_pick_token_id) => last_round_pick_token_id,
            None => {
                // The token the prompt (or a classic step) left pending is on
                // the GPU only; reading it once starts the first round.
                let pending_bonus_token = active_request
                    .pending_generated_token
                    .take()
                    .ok_or_else(|| {
                        fatal_engine_error(
                            "a drafting advance requires a queued pick or a pending token",
                        )
                    })?;
                generated_token_emission::synchronize_generated_token_id(
                    active_request,
                    &pending_bonus_token,
                )?
            }
        };
        self.run_one_draft_round(request_id, active_request, bonus_token_id)
    }

    /// Runs one round from the token that seeds it, queues the round's other
    /// tokens, and emits the seed.
    fn run_one_draft_round(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        bonus_token_id: u32,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        if active_request.generated_token_count == 0
            && active_request.first_decode_forward_elapsed_millis.is_none()
        {
            // The prompt already produced this token, as on the classic path.
            active_request.first_decode_forward_elapsed_millis = Some(0);
        }
        let memory_baseline = self.begin_forward_memory_measurement()?;
        let round_started_at = std::time::Instant::now();
        let round_outcome = self.verified_draft_round(active_request, bonus_token_id)?;
        active_request
            .engine_extension
            .draft_request_state
            .round_tally
            .record_round(
                // The round's last token is its own next pick, which the
                // next round commits and counts as its anchor.
                round_outcome.emitted_token_ids.len().saturating_sub(1),
                round_started_at.elapsed(),
            );
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let completed_forward_memory = measure_completed_forward(
            model,
            MeasuredForward::reply_round(round_outcome.drafted_token_count + 1),
            active_request.request_decoder_state.payload_byte_count(),
            memory_baseline.active_memory_bytes,
            &mut active_request.performance_attribution,
        )?;
        active_request
            .engine_extension
            .mlx_buffer_cache_reply_record
            .record_token(
                u64::try_from(
                    completed_forward_memory
                        .mlx_memory_snapshot
                        .mlx_buffer_cache_bytes(),
                )
                .unwrap_or(u64::MAX),
            );
        record_round_counters(request_id, active_request, &round_outcome);
        for queued_token_id in round_outcome.emitted_token_ids.iter().skip(1) {
            active_request
                .engine_extension
                .draft_request_state
                .emission_queue
                .push_back(*queued_token_id);
        }
        self.emit_drafted_token(
            active_request,
            bonus_token_id,
            Some(&completed_forward_memory),
        )
    }

    /// The round itself: draft, verify every row in one target pass, commit
    /// the kept chain, and advance the request's position over it.
    fn verified_draft_round(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        bonus_token_id: u32,
    ) -> Result<DraftTreeRoundOutcome, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let draft_runtime = self.speculative_draft_runtime.as_ref().ok_or_else(|| {
            fatal_engine_error("a drafting advance requires the engine's hosted drafter")
        })?;
        let mut draft_session = active_request
            .engine_extension
            .draft_request_state
            .draft_session
            .take()
            .ok_or_else(|| fatal_engine_error("a draft round requires the request's session"))?;
        let context_length_tokens = active_request.next_position_tokens;
        let round_request = DraftTreeRoundRequest {
            bonus_token_id,
            context_length_tokens,
            target_token_choice: verification_token_choice_for_strategy(
                &active_request.sampling_strategy,
            )
            .ok_or_else(|| {
                fatal_engine_error("a draft round requires a reproducible sampling strategy")
            })?,
        };
        model.begin_decode_astro_expert_cache_forward();
        let round_result = run_draft_tree_round(
            model,
            &mut active_request.request_decoder_state,
            draft_runtime,
            &mut draft_session,
            &round_request,
        );
        model.end_decode_astro_expert_cache_forward();
        let round_outcome = round_result.map_err(qwen3_5_runtime_error)?;
        active_request
            .engine_extension
            .draft_request_state
            .draft_session = Some(draft_session);
        active_request.advance_position(
            round_outcome
                .context_length_after_round
                .saturating_sub(context_length_tokens) as usize,
        )?;
        Ok(round_outcome)
    }

    /// Emits one token whose id the round already produced on the host.
    fn emit_drafted_token(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        generated_token_id: u32,
        completed_forward_memory: Option<&CompletedForwardMemoryObservation>,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let will_be_terminal =
            self.generated_token_will_be_terminal(active_request, generated_token_id);
        let terminal_forward_memory = match (will_be_terminal, completed_forward_memory) {
            (true, None) => Some(capture_completed_forward_memory_observation(model)?),
            _ => None,
        };
        let generated_token_emission = self.build_generated_token_emission(
            model,
            active_request,
            generated_token_id,
            completed_forward_memory.or(terminal_forward_memory.as_ref()),
        )?;
        if will_be_terminal || generated_token_emission.is_terminal {
            return Ok(ActiveRequestAdvance::Complete(
                generated_token_emission.generated_token,
            ));
        }
        Ok(ActiveRequestAdvance::Continue(
            generated_token_emission.generated_token,
        ))
    }
}

fn record_round_counters(
    request_id: RequestId,
    active_request: &mut Qwen3_5EngineRequest,
    round_outcome: &DraftTreeRoundOutcome,
) {
    let performance_attribution = &mut active_request.performance_attribution;
    performance_attribution.record_counter(crate::PerformanceCounter::SpeculativeRoundCount, 1);
    performance_attribution.record_counter(
        crate::PerformanceCounter::SpeculativeProposedTokenCount,
        round_outcome.drafted_token_count as u64,
    );
    performance_attribution.record_counter(
        crate::PerformanceCounter::SpeculativeAcceptedTokenCount,
        round_outcome.accepted_proposal_count as u64,
    );
    tracing::debug!(
        request_id = request_id.value(),
        accepted_proposal_count = round_outcome.accepted_proposal_count,
        drafted_token_count = round_outcome.drafted_token_count,
        "a draft round kept a chain of its tree on the streaming engine"
    );
}
