//! The speculative decode advance: drain accepted blocks, draft new ones.
//!
//! An admitted session reaches this advance before any classic decode
//! machinery runs. Two shapes exist. When a previous round still owes
//! tokens, the queue drains one token per call through the ordinary
//! emission machinery, so every per-token side effect — stop conditions,
//! thinking-end markers, penalty history — keeps its exact classic
//! semantics. When the queue is empty, the seeded bonus token drives one
//! draft-and-verify round; the round's first token emits now and the rest
//! queue, with the round's correction token as the tail and next bonus.

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
    pub(in crate::qwen3_5_resident::inference_execution) fn advance_speculative_generation(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<ActiveRequestAdvance, crate::InferenceEngineError> {
        if let Some(current_generated_token) = active_request.speculative_emission_queue.pop_front()
        {
            return self.emit_one_speculative_token(active_request, current_generated_token);
        }
        self.run_one_speculative_round(request_id, active_request)
    }

    /// Drains one queued token through the classic emission machinery.
    fn emit_one_speculative_token(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        current_generated_token: mlx_c_rust::MlxArray,
    ) -> Result<ActiveRequestAdvance, crate::InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error)?;
        let current_generated_token_id = generated_token_emission::synchronize_generated_token_id(
            active_request,
            &current_generated_token,
        )?;
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

    /// Runs one draft-and-verify round from the seeded bonus token.
    fn run_one_speculative_round(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<ActiveRequestAdvance, crate::InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error)?;
        let draft_runtime = self.speculative_draft_runtime.as_ref().ok_or_else(|| {
            fatal_engine_error("speculative advance requested without a resident draft runtime")
        })?;
        let pending_bonus_token =
            active_request
                .pending_generated_token
                .take()
                .ok_or_else(|| {
                    fatal_engine_error("a speculative round requires its seeded bonus token")
                })?;
        // The prefill-seeded bonus reports a truthful zero for the first
        // decode-forward latency, exactly as the classic pending path does.
        if super::super::decode_lifecycle::decode_step_decisions::pending_successor_reports_zero_first_decode_time(
            active_request.generated_token_count,
            active_request.first_decode_forward_elapsed_millis,
        ) {
            active_request.first_decode_forward_elapsed_millis = Some(0);
        }
        let bonus_token_id = pending_bonus_token
            .item_u32()
            .map_err(qwen3_5_runtime_error)?;
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
        };
        let round_outcome = run_speculative_round(
            model,
            &mut active_request.request_decoder_state,
            &draft_runtime.draft.weights,
            &draft_runtime.draft_configuration,
            &mut draft_key_value_states,
            &round_request,
            &mut active_request.performance_attribution,
        )
        .map_err(crate::InferenceEngineError::from)?;
        active_request.speculative_draft_key_value_states = Some(draft_key_value_states);
        let accepted_context_growth = round_outcome
            .context_length_after_round
            .checked_sub(context_length_tokens)
            .unwrap_or(0) as usize;
        active_request.advance_position(accepted_context_growth)?;
        tracing::debug!(
            request_id = request_id.value(),
            accepted_proposal_count = round_outcome.accepted_proposal_count,
            speculation_token_count,
            "speculative round accepted its drafted block prefix"
        );
        // Queue every round token except the first — the bonus emits now.
        let runtime = model.runtime();
        for queued_token_id in round_outcome.emitted_token_ids.iter().skip(1) {
            let queued_generated_token = runtime
                .array_from_u32(&[*queued_token_id], &[1, 1])
                .map_err(qwen3_5_runtime_error)?;
            active_request
                .speculative_emission_queue
                .push_back(queued_generated_token);
        }
        let bonus_generated_token = runtime
            .array_from_u32(&[bonus_token_id], &[1, 1])
            .map_err(qwen3_5_runtime_error)?;
        self.emit_one_speculative_token(active_request, bonus_generated_token)
    }
}
