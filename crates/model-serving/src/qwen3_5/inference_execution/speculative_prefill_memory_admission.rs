/// Returns the target expert-retention payload that must be reclaimed before
/// drafter scoring can allocate its remaining decoder state and one expert page.
#[must_use]
pub(crate) const fn speculative_prefill_draft_scoring_reclamation_target_bytes(
    current_active_memory_bytes: usize,
    draft_scoring_reservation_bytes: usize,
    allowed_active_memory_bytes: usize,
) -> usize {
    current_active_memory_bytes
        .saturating_add(draft_scoring_reservation_bytes)
        .saturating_sub(allowed_active_memory_bytes)
}

/// Combines the independently owned drafter allocations required before its
/// scoring graph can run.
#[must_use]
pub(crate) fn speculative_prefill_draft_scoring_reservation_bytes(
    draft_decoder_state_growth_bytes: usize,
    draft_vision_payload_bytes: usize,
    draft_maximum_expert_page_reservation_bytes: usize,
    draft_temporary_workspace_bytes: usize,
) -> Option<usize> {
    draft_decoder_state_growth_bytes
        .checked_add(draft_vision_payload_bytes)?
        .checked_add(draft_maximum_expert_page_reservation_bytes)?
        .checked_add(draft_temporary_workspace_bytes)
}

#[cfg(feature = "direct-mlx")]
use super::super::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
#[cfg(feature = "direct-mlx")]
use crate::{
    InferenceEngineError, PerformanceAttribution, Qwen3_5Model,
    qwen3_5_moe::reclaim_retained_experts_for_request_memory_pressure,
};

#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    /// Evicts only sufficient pageable target expert weights before a drafter
    /// needs additional GPU state, retaining the target model core and the
    /// configured MLX ceiling.
    pub(super) fn admit_speculative_prefill_draft_scoring_memory(
        &self,
        request_id: u64,
        draft_model: &Qwen3_5Model,
        draft_request_decoder_state: &RequestDecoderStateStack,
        draft_suffix_token_count: usize,
        should_reserve_draft_vision_payload: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        let target_model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded target model"))?;
        let draft_decoder_state_growth_bytes = draft_request_decoder_state
            .projected_persistent_state_growth_bytes(
                draft_model.decoder_cache_layout(),
                draft_suffix_token_count,
            )
            .map_err(qwen3_5_runtime_error)?;
        let draft_maximum_expert_page_reservation_bytes =
            self.speculative_prefill_draft_maximum_expert_page_reservation_bytes();
        let draft_vision_payload_bytes = if should_reserve_draft_vision_payload {
            draft_model
                .vision_model()
                .map_or(0_u64, |draft_vision_model| {
                    draft_vision_model.projected_payload_bytes()
                })
        } else {
            0
        };
        let draft_vision_payload_bytes =
            usize::try_from(draft_vision_payload_bytes).map_err(|_| {
                fatal_engine_error("speculative-prefill draft vision payload exceeds usize")
            })?;
        let draft_temporary_workspace_bytes = draft_model
            .decoder_cache_layout()
            .boundary_snapshot_payload_byte_count()
            .map_err(|draft_workspace_projection_error| {
                fatal_engine_error(format!(
                    "failed to project speculative-prefill draft temporary workspace: {draft_workspace_projection_error}"
                ))
            })?;
        let draft_scoring_reservation_bytes = speculative_prefill_draft_scoring_reservation_bytes(
            draft_decoder_state_growth_bytes,
            draft_vision_payload_bytes,
            draft_maximum_expert_page_reservation_bytes,
            draft_temporary_workspace_bytes,
        )
        .ok_or_else(|| {
            fatal_engine_error("speculative-prefill draft scoring reservation overflowed")
        })?;
        let memory_snapshot_before_draft_scoring = target_model
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        let required_target_expert_reclamation_bytes =
            speculative_prefill_draft_scoring_reclamation_target_bytes(
                memory_snapshot_before_draft_scoring.active_memory_bytes(),
                draft_scoring_reservation_bytes,
                self.memory_limits.allowed_active_memory_bytes(),
            );
        if required_target_expert_reclamation_bytes == 0 {
            return Ok(());
        }

        let expert_weight_memory_cache_statistics_before_reclamation =
            target_model.expert_weight_memory_cache_statistics();
        let Some(memory_snapshot_after_target_expert_reclamation) =
            reclaim_retained_experts_for_request_memory_pressure(
                target_model,
                required_target_expert_reclamation_bytes,
            )?
        else {
            return Err(InferenceEngineError::InvalidRequest {
                reason:
                    "speculative-prefill drafter scoring cannot reclaim pageable target experts"
                        .to_owned(),
            });
        };
        let remaining_target_expert_reclamation_bytes =
            speculative_prefill_draft_scoring_reclamation_target_bytes(
                memory_snapshot_after_target_expert_reclamation.active_memory_bytes(),
                draft_scoring_reservation_bytes,
                self.memory_limits.allowed_active_memory_bytes(),
            );
        let expert_weight_memory_cache_statistics_after_reclamation =
            target_model.expert_weight_memory_cache_statistics();
        let reclaimed_target_expert_payload_bytes =
            expert_weight_memory_cache_statistics_before_reclamation
                .resident_payload_byte_count
                .saturating_sub(
                    expert_weight_memory_cache_statistics_after_reclamation
                        .resident_payload_byte_count,
                );
        performance_attribution.record_counter(
            crate::PerformanceCounter::SpeculativePrefillDraftTargetExpertReclaimedPayloadBytes,
            reclaimed_target_expert_payload_bytes,
        );
        if remaining_target_expert_reclamation_bytes > 0 {
            target_model.resume_expert_retention_after_request_memory_pressure();
            return Err(InferenceEngineError::InvalidRequest {
                reason: "speculative-prefill drafter scoring remains above the MLX memory ceiling after target expert reclamation".to_owned(),
            });
        }
        tracing::info!(
            request_id,
            active_memory_bytes_before_draft_scoring =
                memory_snapshot_before_draft_scoring.active_memory_bytes(),
            active_memory_bytes_after_target_expert_reclamation =
                memory_snapshot_after_target_expert_reclamation.active_memory_bytes(),
            draft_decoder_state_growth_bytes,
            draft_vision_payload_bytes,
            draft_maximum_expert_page_reservation_bytes,
            draft_temporary_workspace_bytes,
            draft_scoring_reservation_bytes,
            required_target_expert_reclamation_bytes,
            reclaimed_target_expert_payload_bytes,
            retained_target_expert_payload_bytes_before =
                expert_weight_memory_cache_statistics_before_reclamation
                    .resident_payload_byte_count,
            retained_target_expert_payload_bytes_after =
                expert_weight_memory_cache_statistics_after_reclamation.resident_payload_byte_count,
            "reclaimed pageable target experts before speculative-prefill drafter scoring"
        );
        Ok(())
    }

    /// Uses MLX's exact rejected allocation size to reclaim the smallest
    /// pageable target-expert payload needed for one scoring retry.
    pub(super) fn reclaim_target_experts_after_draft_scoring_allocation_rejection(
        &self,
        request_id: u64,
        active_memory_bytes: usize,
        attempted_allocation_bytes: usize,
        allowed_active_memory_bytes: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        let required_target_expert_reclamation_bytes =
            speculative_prefill_draft_scoring_reclamation_target_bytes(
                active_memory_bytes,
                attempted_allocation_bytes,
                allowed_active_memory_bytes,
            );
        if required_target_expert_reclamation_bytes == 0 {
            return Ok(());
        }
        let target_model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded target model"))?;
        let expert_weight_memory_cache_statistics_before_reclamation =
            target_model.expert_weight_memory_cache_statistics();
        let Some(memory_snapshot_after_target_expert_reclamation) =
            reclaim_retained_experts_for_request_memory_pressure(
                target_model,
                required_target_expert_reclamation_bytes,
            )?
        else {
            return Err(InferenceEngineError::InvalidRequest {
                reason:
                    "speculative-prefill drafter scoring cannot reclaim pageable target experts"
                        .to_owned(),
            });
        };
        let expert_weight_memory_cache_statistics_after_reclamation =
            target_model.expert_weight_memory_cache_statistics();
        let reclaimed_target_expert_payload_bytes =
            expert_weight_memory_cache_statistics_before_reclamation
                .resident_payload_byte_count
                .saturating_sub(
                    expert_weight_memory_cache_statistics_after_reclamation
                        .resident_payload_byte_count,
                );
        performance_attribution.record_counter(
            crate::PerformanceCounter::SpeculativePrefillDraftTargetExpertReclaimedPayloadBytes,
            reclaimed_target_expert_payload_bytes,
        );
        if speculative_prefill_draft_scoring_reclamation_target_bytes(
            memory_snapshot_after_target_expert_reclamation.active_memory_bytes(),
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
        ) > 0
        {
            target_model.resume_expert_retention_after_request_memory_pressure();
            return Err(InferenceEngineError::InvalidRequest {
                reason: "speculative-prefill drafter scoring allocation remains above the MLX memory ceiling after target expert reclamation".to_owned(),
            });
        }
        tracing::info!(
            request_id,
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
            active_memory_bytes_after_target_expert_reclamation =
                memory_snapshot_after_target_expert_reclamation.active_memory_bytes(),
            required_target_expert_reclamation_bytes,
            reclaimed_target_expert_payload_bytes,
            retained_target_expert_payload_bytes_before =
                expert_weight_memory_cache_statistics_before_reclamation
                    .resident_payload_byte_count,
            retained_target_expert_payload_bytes_after =
                expert_weight_memory_cache_statistics_after_reclamation.resident_payload_byte_count,
            "reclaimed pageable target experts after speculative-prefill drafter allocation rejection"
        );
        Ok(())
    }
}
