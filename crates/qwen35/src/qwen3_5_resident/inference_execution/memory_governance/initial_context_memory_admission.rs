//! Whether a request's whole context fits, asked before the first forward.

use crate::qwen3_5_resident::model::request_errors;
use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceAttributionOutcome,
    StreamingEngineRequirement,
};
use astronomical_ipc_protocol::RequestId;

use super::Qwen3_5EngineState;
use super::forward_memory::fixed_weight_bytes;

impl Qwen3_5EngineState {
    pub(in crate::qwen3_5_resident::inference_execution) fn admit_initial_generation_context_or_record_rejection(
        &mut self,
        request_id: RequestId,
        configured_maximum_output_tokens: u16,
        total_context_tokens: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        match self.require_request_context_to_fit(total_context_tokens) {
            Ok(()) => {
                self.govern_request_start(self.live_retained_context_state_payload_bytes());
                Ok(())
            }
            Err(context_admission_error) => {
                self.record_generation_performance_attribution(
                    std::mem::replace(performance_attribution, PerformanceAttribution::disabled()),
                    PerformanceAttributionOutcome::Rejected,
                    request_id,
                    configured_maximum_output_tokens,
                    None,
                    Some("generation context admission rejected"),
                );
                Err(context_admission_error)
            }
        }
    }

    /// Whether a request whose context reaches `context_token_count` tokens
    /// fits beside every weight, asked of the memory governor before the
    /// request changes any state.
    ///
    /// The stored state is taken as the larger of what MLX holds beyond the
    /// weights now and what the whole context will occupy. What is held now
    /// is either a prefix this request continues from, which the context
    /// figure already contains, or state the request replaces.
    pub(in crate::qwen3_5_resident::inference_execution) fn require_request_context_to_fit(
        &self,
        context_token_count: usize,
    ) -> Result<(), InferenceEngineError> {
        let whole_context_bytes = context_token_count
            .checked_mul(self.context_memory_reservation_bytes_per_token)
            .ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::generation_context_memory_reservation_overflowed_error)?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error)?;
        let memory_governor = model.memory_governor.borrow();
        let Some(memory_governor) = memory_governor.as_ref() else {
            return Ok(());
        };
        let active_bytes = model
            .runtime()
            .memory_snapshot()
            .map_err(super::qwen3_5_runtime_error)?
            .active_memory_bytes() as u64;
        let stored_state_bytes = active_bytes
            .saturating_sub(fixed_weight_bytes(model))
            .max(whole_context_bytes as u64);
        let prompt_chunk_accounts =
            memory_governor.accounts_while(self.serving_prompt_chunk_phase(), stored_state_bytes);
        if prompt_chunk_accounts.shortfall_bytes == 0 {
            return Ok(());
        }
        Err(request_errors::resident_capacity_error(
            StreamingEngineRequirement::RequestContextShortfall {
                shortfall_bytes: usize::try_from(prompt_chunk_accounts.shortfall_bytes)
                    .unwrap_or(usize::MAX),
            },
            "generation context exceeds available GPU wired memory",
        ))
    }
}
