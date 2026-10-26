//! The speculative decode advance: drain committed blocks, seed from the tail.
//!
//! An admitted session reaches this advance before any classic decode
//! machinery runs. The emission queue holds tokens one accepted round owes
//! the caller: every entry except the tail is already committed to the
//! target's decoder state, so draining them is pure emission through the
//! ordinary machinery — stop conditions, thinking-end markers, and penalty
//! history keep their exact classic semantics. The single tail entry is the
//! round's correction token: the target's own next preference, committed by
//! no forward yet. When the drain reaches it, that token both emits as this
//! call's output and seeds the next draft-and-verify round, whose verify
//! forward commits it — so the queue never empties while the session lives,
//! and an admitted session can never fall through to the classic advance
//! (a fall-through would re-forward a stale token at the wrong position;
//! the serving journey locks this contract end to end).

use astronomical_ipc_protocol::RequestId;

use crate::GeneratedToken;
use crate::qwen3_5_resident::dflash_draft::draft_key_value_state::DFlashDraftKeyValueStates;
use crate::qwen3_5_resident::inference_execution::decode_lifecycle::advance_generation::ActiveRequestAdvance;
use crate::qwen3_5_resident::inference_execution::decode_lifecycle::generated_token_emission;
use crate::qwen3_5_resident::inference_execution::memory_governance::completed_forward_memory::capture_completed_forward_memory_observation;

use super::super::engine_request::Qwen3_5EngineRequest;
use super::super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use super::speculation_round::{SpeculationRoundRequest, run_speculative_round};

impl Qwen3_5EngineState {
    /// One speculative advance step. `Ok(None)` means the session just
    /// measured itself out of speculation: the seed token moved to the
    /// classic pending slot, the session suspended, and the caller must
    /// take the classic advance path for this and every later token.
    pub(in crate::qwen3_5_resident::inference_execution) fn advance_speculative_generation(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<Option<ActiveRequestAdvance>, crate::InferenceEngineError> {
        // The queue holds committed tokens in front and exactly one
        // uncommitted token at the tail: the last round's correction, which
        // the target endorsed but no forward has committed yet. Draining a
        // committed token is pure emission — its decoder-state entry
        // already exists. Reaching the tail means the tail itself becomes
        // this call's emission AND the bonus of the next round, whose
        // verify forward commits it. An admitted session therefore never
        // falls through to the classic advance: falling through with a
        // drained queue would re-forward a stale token at the wrong
        // position and corrupt the request. The one sanctioned exit is the
        // low-acceptance guard below, which hands the uncommitted seed to
        // the classic pending slot first, so the classic path commits it
        // exactly as a verify forward would have.
        if active_request.speculative_emission_queue.len() > 1 {
            let committed_generated_token = active_request
                .speculative_emission_queue
                .pop_front()
                .expect("the queue length was just proven above one");
            return self
                .emit_one_speculative_token(active_request, committed_generated_token)
                .map(Some);
        }
        // A full window of rounds below the break-even acceptance floor
        // means further rounds cost more than they commit — high-temperature
        // sessions flatten the target distribution until the draft rarely
        // matches. The seed (queue tail or pending) becomes the classic
        // pending token; committed queue tokens are all emitted already.
        if super::speculation_session_admission::should_suspend_speculation_for_poor_acceptance(
            &active_request
                .speculative_acceptance_window
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        ) {
            let seed_generated_token = if let Some(correction_token) =
                active_request.speculative_emission_queue.pop_front()
            {
                correction_token
            } else {
                active_request.pending_generated_token.take().ok_or_else(|| {
                        fatal_engine_error(
                            "a speculative advance requires a queued correction or pending bonus token",
                        )
                    })?
            };
            active_request.pending_generated_token = Some(seed_generated_token);
            active_request.suspend_speculative_session("poor_acceptance");
            return Ok(None);
        }
        let bonus_generated_token =
            if let Some(correction_token) = active_request.speculative_emission_queue.pop_front() {
                correction_token
            } else {
                active_request
                .pending_generated_token
                .take()
                .ok_or_else(|| {
                    fatal_engine_error(
                        "a speculative advance requires a queued correction or pending bonus token",
                    )
                })?
            };
        self.run_one_speculative_round(request_id, active_request, bonus_generated_token)
            .map(Some)
    }

    /// Drains one queued token through the classic emission machinery;
    /// used for every committed token between rounds.
    fn emit_one_speculative_token(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        current_generated_token: mlx_c_rust::MlxArray,
    ) -> Result<ActiveRequestAdvance, crate::InferenceEngineError> {
        let current_generated_token_id = generated_token_emission::synchronize_generated_token_id(
            active_request,
            &current_generated_token,
        )?;
        self.emit_one_speculative_token_id(
            active_request,
            current_generated_token,
            current_generated_token_id,
        )
    }

    /// Emits one already-synchronized token, so a caller that read the id
    /// for its own decisions (the round's bonus) never synchronizes twice.
    /// The device array stays a parameter so its storage keeps living until
    /// the emission has consumed it.
    fn emit_one_speculative_token_id(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        _current_generated_token: mlx_c_rust::MlxArray,
        current_generated_token_id: u32,
    ) -> Result<ActiveRequestAdvance, crate::InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error)?;
        if self.generated_token_will_be_terminal(active_request, current_generated_token_id) {
            let completed_forward_memory = if self.adaptive_ram_growth_guard_enabled {
                Some(capture_completed_forward_memory_observation(model)?)
            } else {
                None
            };
            let generated_token_emission = self.build_generated_token_emission(
                model,
                active_request,
                current_generated_token_id,
                completed_forward_memory.as_ref(),
            )?;
            return Ok(ActiveRequestAdvance::Complete(
                generated_token_emission.generated_token,
            ));
        }
        let generated_token_emission = self.build_generated_token_emission(
            model,
            active_request,
            current_generated_token_id,
            None,
        )?;
        if generated_token_emission.is_terminal {
            return Ok(ActiveRequestAdvance::Complete(
                generated_token_emission.generated_token,
            ));
        }
        Ok(ActiveRequestAdvance::Continue(
            generated_token_emission.generated_token,
        ))
    }

    /// Runs one draft-and-verify round from the bonus token that seeds it,
    /// then emits that bonus through the classic emission machinery.
    fn run_one_speculative_round(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        bonus_generated_token: mlx_c_rust::MlxArray,
    ) -> Result<ActiveRequestAdvance, crate::InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error)?;
        let draft_runtime = self.speculative_draft_runtime.as_ref().ok_or_else(|| {
            fatal_engine_error("speculative advance requested without a resident draft runtime")
        })?;
        // The prefill-seeded bonus reports a truthful zero for the first
        // decode-forward latency, exactly as the classic pending path does.
        if super::super::decode_lifecycle::decode_step_decisions::pending_successor_reports_zero_first_decode_time(
            active_request.generated_token_count,
            active_request.first_decode_forward_elapsed_millis,
        ) {
            active_request.first_decode_forward_elapsed_millis = Some(0);
        }
        let bonus_token_id = generated_token_emission::synchronize_generated_token_id(
            active_request,
            &bonus_generated_token,
        )?;
        // A block never drafts past the session's output budget minus the
        // bonus: the terminal checks below own the final tokens.
        let remaining_output_tokens = active_request
            .maximum_output_tokens
            .saturating_sub(active_request.generated_token_count);
        let speculation_token_count = draft_runtime
            .speculation_block_length
            .min(remaining_output_tokens.saturating_sub(1) as usize)
            .max(1);
        let context_length_tokens = active_request.next_position_tokens;
        let mut draft_key_value_states: DFlashDraftKeyValueStates = active_request
            .speculative_draft_key_value_states
            .take()
            .ok_or_else(|| {
                fatal_engine_error(
                    "a speculative round requires the request's draft key-value states",
                )
            })?;
        let round_request = SpeculationRoundRequest {
            request_id,
            bonus_token_id,
            speculation_token_count,
            context_length_tokens,
            // Admission guarantees the strategy is reproducible; the error
            // arm is the defensive boundary if a future strategy lands.
            target_token_choice:
                super::speculation_session_admission::verification_token_choice_for_strategy(
                    &active_request.sampling_strategy,
                )
                .ok_or_else(|| {
                    fatal_engine_error(
                        "a speculative round requires a reproducible sampling strategy",
                    )
                })?,
        };
        let round_outcome = run_speculative_round(
            model,
            &mut active_request.request_decoder_state,
            &draft_runtime.draft.weights,
            &draft_runtime.draft_configuration,
            &mut draft_key_value_states,
            &round_request,
            &mut active_request.random_state,
            &mut active_request.performance_attribution,
        )
        .map_err(crate::InferenceEngineError::from)?;
        active_request.speculative_draft_key_value_states = Some(draft_key_value_states);
        let accepted_context_growth = round_outcome
            .context_length_after_round
            .checked_sub(context_length_tokens)
            .unwrap_or(0) as usize;
        active_request.advance_position(accepted_context_growth)?;
        active_request
            .performance_attribution
            .record_counter(crate::PerformanceCounter::SpeculativeRoundCount, 1);
        active_request.performance_attribution.record_counter(
            crate::PerformanceCounter::SpeculativeProposedTokenCount,
            speculation_token_count as u64,
        );
        active_request.performance_attribution.record_counter(
            crate::PerformanceCounter::SpeculativeAcceptedTokenCount,
            round_outcome.accepted_proposal_count as u64,
        );
        // Feed the rolling acceptance window the low-acceptance guard
        // reads; the cap keeps the window at the admission module's length.
        active_request.speculative_acceptance_window.push_back((
            speculation_token_count as u64,
            round_outcome.accepted_proposal_count as u64,
        ));
        while active_request.speculative_acceptance_window.len()
            > super::speculation_session_admission::ACCEPTANCE_WINDOW_ROUND_COUNT
        {
            active_request.speculative_acceptance_window.pop_front();
        }
        tracing::debug!(
            request_id = request_id.value(),
            accepted_proposal_count = round_outcome.accepted_proposal_count,
            speculation_token_count,
            "speculative round accepted its drafted block prefix"
        );
        // Queue every round token except the bonus, which emits now. The
        // tail entry is the round's correction token: the target's own next
        // preference, absent from the decoder state until the next round's
        // verify commits it.
        let runtime = model.runtime();
        for queued_token_id in round_outcome.emitted_token_ids.iter().skip(1) {
            let queued_generated_token = runtime
                .array_from_u32(&[*queued_token_id], &[1, 1])
                .map_err(qwen3_5_runtime_error)?;
            active_request
                .speculative_emission_queue
                .push_back(queued_generated_token);
        }
        self.emit_one_speculative_token_id(active_request, bonus_generated_token, bonus_token_id)
    }
}
