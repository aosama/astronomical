//! Initial generation-context admission for one Qwen3.5 request.
//!
//! Workspace composition and demote-or-admit live in `memory/`. This module
//! measures request facts and records a rejection when the request cannot fit.

use crate::qwen3_5::model::memory_admission::invalid_request_error;
use crate::{
    InferenceEngineError, MemoryPhase, PerformanceAttribution, PerformanceAttributionOutcome,
    request_context_temporary_workspace_bytes,
};
use astronomical_ipc_protocol::RequestId;

use super::{Qwen3_5EngineState, fatal_engine_error};

impl Qwen3_5EngineState {
    pub(super) fn admit_initial_generation_context_or_record_rejection(
        &mut self,
        request_id: RequestId,
        configured_maximum_output_tokens: u16,
        total_context_tokens: usize,
        prompt_token_count: usize,
        can_use_persistent_prompt_cache: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        match self.validate_initial_generation_context_memory_admission(
            total_context_tokens,
            prompt_token_count,
            can_use_persistent_prompt_cache,
            performance_attribution,
        ) {
            Ok(()) => Ok(()),
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

    pub(super) fn validate_initial_generation_context_memory_admission(
        &mut self,
        total_context_tokens: usize,
        prompt_token_count: usize,
        can_use_persistent_prompt_cache: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        let context_growth_bytes = total_context_tokens
            .checked_mul(self.context_memory_reservation_bytes_per_token)
            .ok_or_else(|| {
                invalid_request_error("generation context memory reservation overflowed")
            })?;
        let (
            prefill_activation_workspace_bytes,
            complete_layer_scratch_bytes,
            complete_experts_are_resident,
        ) = {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let complete_experts_are_resident = model.resident_expert_weights.is_some();
            // Layer-weight activation heuristics and the SSD stream slot already
            // live inside the seated active snapshot. Adding them again stacks
            // exclusive paper peaks on top of RAM that is already allocated.
            if complete_experts_are_resident {
                (0, 0, true)
            } else {
                let ram_budget = model.mlx_ram_budget();
                // The activation reserve belongs to one forward, so it is
                // sized by the planned chunk bound, never the total context
                // length. Sizing it from the context multiplied a chunk-shaped
                // learned observation into a reserve several times the
                // ceiling and rejected every later request (issue #690); the
                // planned chunk size is the same operation scope `plan()`
                // already resolves (issue #644).
                let planned_prefill_operation_token_count = u64::try_from(
                    self.prompt_processing_chunk_sizer
                        .maximum_prompt_processing_chunk_size_tokens(),
                )
                .unwrap_or(u64::MAX);
                let prefill_activation_workspace_bytes = usize::try_from(
                    ram_budget.activation_headroom_bytes(
                        MemoryPhase::Prefill,
                        planned_prefill_operation_token_count,
                    ),
                )
                .map_err(|_| {
                    invalid_request_error("prefill activation workspace exceeds the platform range")
                })?;
                let complete_layer_scratch_bytes = usize::try_from(
                    ram_budget
                        .model_geometry()
                        .largest_complete_expert_layer_bytes,
                )
                .map_err(|_| {
                    invalid_request_error(
                        "complete-layer scratch reservation exceeds the platform range",
                    )
                })?;
                (
                    prefill_activation_workspace_bytes,
                    complete_layer_scratch_bytes,
                    false,
                )
            }
        };
        let temporary_workspace_reservation_bytes = request_context_temporary_workspace_bytes(
            complete_experts_are_resident,
            context_growth_bytes,
            0,
            0,
            prefill_activation_workspace_bytes,
            complete_layer_scratch_bytes,
        )
        .ok_or_else(|| {
            invalid_request_error("generation context workspace reservation overflowed")
        })?;
        crate::memory::log_generation_context_workspace_reservation(
            total_context_tokens,
            prompt_token_count,
            can_use_persistent_prompt_cache,
            self.context_memory_reservation_bytes_per_token,
            0,
            0,
            prefill_activation_workspace_bytes,
            complete_layer_scratch_bytes,
            temporary_workspace_reservation_bytes,
            0,
        );
        self.validate_context_memory_admission_with_resident_expert_demotion(
            total_context_tokens,
            temporary_workspace_reservation_bytes,
            0,
            performance_attribution,
        )?;
        Ok(())
    }
}
