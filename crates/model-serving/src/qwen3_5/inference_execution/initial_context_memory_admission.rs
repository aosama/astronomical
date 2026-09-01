//! Initial generation-context admission for one Qwen3.5 request.
//!
//! Workspace composition and demote-or-admit live in `memory/`. This module
//! measures request facts and records a rejection when the request cannot fit.

use crate::qwen3_5::model::memory_admission::invalid_request_error;
use crate::{
    InferenceEngineError, MemoryPhase, PerformanceAttribution, PerformanceAttributionOutcome,
    PerformanceCounter, persistent_context_restore_workspace_bytes,
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
    ) -> Result<u64, InferenceEngineError> {
        // Return reclaimed expert bytes so request diagnostics can explain work
        // performed specifically to reserve direct cache-publication workspace.
        match self.validate_initial_generation_context_memory_admission(
            total_context_tokens,
            prompt_token_count,
            can_use_persistent_prompt_cache,
            performance_attribution,
        ) {
            Ok(reclaimed_expert_payload_bytes) => Ok(reclaimed_expert_payload_bytes),
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
    ) -> Result<u64, InferenceEngineError> {
        let direct_publication_workspace_bytes = if can_use_persistent_prompt_cache {
            self.persistent_prompt_cache_model_contract
                .as_ref()
                .map_or(0, |model_contract| {
                    model_contract.direct_publication_workspace_bytes()
                })
        } else {
            0
        };
        let additional_maximum_expert_page_reservation_bytes =
            self.speculative_prefill_draft_maximum_expert_page_reservation_bytes();
        // Cache restore temporarily owns source tensors beside live decoder
        // state. Charge that overlap only for prompt tokens that may already
        // exist as cache blocks. The output budget is generated later and has
        // no blocks; multiplying it in double-counts future KV and demotes a
        // fitting resident model.
        let restore_overlap_workspace_bytes = if can_use_persistent_prompt_cache {
            persistent_context_restore_workspace_bytes(
                self.context_memory_reservation_bytes_per_token,
                prompt_token_count,
            )
            .ok_or_else(|| invalid_request_error("prompt-cache restore workspace overflowed"))?
        } else {
            0
        };
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
                let prefill_activation_workspace_bytes =
                    usize::try_from(ram_budget.activation_headroom_bytes(MemoryPhase::Prefill))
                        .map_err(|_| {
                            invalid_request_error(
                                "prefill activation workspace exceeds the platform range",
                            )
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
            restore_overlap_workspace_bytes,
            direct_publication_workspace_bytes,
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
            direct_publication_workspace_bytes,
            restore_overlap_workspace_bytes,
            prefill_activation_workspace_bytes,
            complete_layer_scratch_bytes,
            temporary_workspace_reservation_bytes,
            additional_maximum_expert_page_reservation_bytes,
        );
        let target_expert_payload_bytes_reclaimed_during_context_admission = self
            .validate_context_memory_admission_with_resident_expert_demotion(
                total_context_tokens,
                temporary_workspace_reservation_bytes,
                additional_maximum_expert_page_reservation_bytes,
                performance_attribution,
            )?;
        if self.speculative_prefill.enabled {
            performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
                target_expert_payload_bytes_reclaimed_during_context_admission,
            );
        }
        Ok(if can_use_persistent_prompt_cache {
            target_expert_payload_bytes_reclaimed_during_context_admission
        } else {
            0
        })
    }
}
