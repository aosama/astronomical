//! Exact memory admission for request-scoped drafter scoring.
//!
//! Target and drafter share one process-wide MLX active-memory ceiling. Before
//! loading/scoring the drafter, the engine projects every drafter allocation that
//! can overlap and reclaims only pageable target expert payload needed to fit.
//! Model-core weights, active target decoder state, and the configured ceiling
//! are never reduced as part of this operation.

#[cfg(feature = "direct-mlx")]
use super::super::super::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
use super::super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
#[cfg(feature = "direct-mlx")]
use crate::SpeculativePrefillAdmission;
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
    pub(crate) fn admit_speculative_prefill_draft_scoring_memory(
        &mut self,
        request_id: u64,
        draft_model: &Qwen3_5Model,
        draft_request_decoder_state: &RequestDecoderStateStack,
        draft_suffix_token_count: usize,
        should_reserve_draft_vision_payload: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        let draft_decoder_state_growth_bytes = draft_request_decoder_state
            .projected_persistent_state_growth_bytes(
                draft_model.decoder_cache_layout(),
                draft_suffix_token_count,
            )
            .map_err(qwen3_5_runtime_error)?;
        // Read the request-scoped model directly: the startup compatibility
        // drafter has already been dropped and is not retained on engine state.
        let draft_maximum_expert_page_reservation_bytes =
            draft_model.expert_pager.as_ref().map_or(0, |expert_pager| {
                usize::try_from(expert_pager.maximum_expert_page_bytes()).unwrap_or(usize::MAX)
            });
        let draft_vision_payload_bytes = if should_reserve_draft_vision_payload {
            // Reserve the drafter tower's projected output, not the target's:
            // hidden widths are allowed to differ.
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
        // A cache-enabled scoring forward can land on a drafter publication boundary.
        // Disabled storage performs neither checkpoint capture nor serialization.
        let (draft_boundary_checkpoint_workspace_bytes, draft_direct_publication_workspace_bytes) =
            if let Some(draft_persistent_prompt_cache) = self
                .speculative_prefill_draft_persistent_prompt_cache
                .as_ref()
            {
                let draft_boundary_checkpoint_workspace_bytes = draft_model
                .decoder_cache_layout()
                .boundary_snapshot_payload_byte_count()
                .map_err(|draft_workspace_projection_error| {
                    fatal_engine_error(format!(
                        "failed to project speculative-prefill draft temporary workspace: {draft_workspace_projection_error}"
                    ))
                })?;
                (
                    draft_boundary_checkpoint_workspace_bytes,
                    draft_persistent_prompt_cache
                        .model_contract_ref()
                        .direct_publication_workspace_bytes(),
                )
            } else {
                (0, 0)
            };
        let draft_scoring_reservation_bytes =
            SpeculativePrefillAdmission::draft_scoring_reservation_bytes(
                draft_decoder_state_growth_bytes,
                draft_vision_payload_bytes,
                draft_maximum_expert_page_reservation_bytes,
                draft_boundary_checkpoint_workspace_bytes,
                draft_direct_publication_workspace_bytes,
            )
            .ok_or_else(|| {
                fatal_engine_error("speculative-prefill draft scoring reservation overflowed")
            })?;
        let mut memory_snapshot_before_draft_scoring = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded target model"))?
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        // Only pageable target experts are reclaimable here. Target core weights,
        // active request state, and the configured ceiling remain unchanged.
        let mut required_target_expert_reclamation_bytes =
            SpeculativePrefillAdmission::required_target_expert_reclamation_bytes(
                memory_snapshot_before_draft_scoring.active_memory_bytes(),
                draft_scoring_reservation_bytes,
                self.memory_limits.allowed_active_memory_bytes(),
            );
        tracing::info!(
            request_id,
            draft_suffix_token_count,
            active_memory_bytes_before_draft_scoring =
                memory_snapshot_before_draft_scoring.active_memory_bytes(),
            allowed_active_memory_bytes = self.memory_limits.allowed_active_memory_bytes(),
            draft_decoder_state_growth_bytes,
            draft_vision_payload_bytes,
            draft_maximum_expert_page_reservation_bytes,
            draft_boundary_checkpoint_workspace_bytes,
            draft_direct_publication_workspace_bytes,
            draft_scoring_reservation_bytes,
            required_target_expert_reclamation_bytes,
            "projected speculative-prefill drafter scoring memory"
        );
        if required_target_expert_reclamation_bytes == 0 {
            // Avoid touching retention policy when current active memory already
            // has sufficient headroom for the complete overlap projection.
            return Ok(());
        }

        // Complete target ownership is all-or-nothing. Demote only after the
        // exact scoring reservation proves that keeping it cannot fit.
        let target_expert_statistics_before_reclamation = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded target model"))?
            .expert_weight_memory_cache_statistics();
        let target_has_complete_resident_expert_owner = self
            .model
            .as_ref()
            .is_some_and(|target_model| target_model.resident_expert_weights.is_some());
        if target_has_complete_resident_expert_owner {
            self.model
                .as_mut()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded target model"))?
                .demote_resident_experts_to_paging(
                    crate::qwen3_5_moe::Qwen3_5ExpertResidencyTransitionReason::SpeculativePrefillDraftLoading,
                    performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
            memory_snapshot_before_draft_scoring = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded target model"))?
                .runtime()
                .memory_snapshot()
                .map_err(qwen3_5_runtime_error)?;
            required_target_expert_reclamation_bytes =
                SpeculativePrefillAdmission::required_target_expert_reclamation_bytes(
                    memory_snapshot_before_draft_scoring.active_memory_bytes(),
                    draft_scoring_reservation_bytes,
                    self.memory_limits.allowed_active_memory_bytes(),
                );
            let target_expert_statistics_after_complete_owner_demotion = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded target model"))?
                .expert_weight_memory_cache_statistics();
            performance_attribution.record_counter(
                crate::PerformanceCounter::SpeculativePrefillDraftTargetExpertReclaimedPayloadBytes,
                target_expert_statistics_before_reclamation
                    .resident_payload_byte_count
                    .saturating_sub(
                        target_expert_statistics_after_complete_owner_demotion
                            .resident_payload_byte_count,
                    ),
            );
            if required_target_expert_reclamation_bytes == 0 {
                tracing::info!(
                    request_id,
                    draft_scoring_reservation_bytes,
                    active_memory_bytes_after_target_demotion =
                        memory_snapshot_before_draft_scoring.active_memory_bytes(),
                    "demoted complete target experts only after drafter scoring required the capacity"
                );
                return Ok(());
            }
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
            // No reclaimable pageable owner exists. Do not evict model core or
            // target request state to force a configured optimization to run.
            return Err(InferenceEngineError::InvalidRequest {
                reason:
                    "speculative-prefill drafter scoring cannot reclaim pageable target experts"
                        .to_owned(),
            });
        };
        let remaining_target_expert_reclamation_bytes =
            SpeculativePrefillAdmission::required_target_expert_reclamation_bytes(
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
            // Reclamation changed retention policy but still could not prove the
            // projection fits. Resume normal target retention before failing.
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
            draft_boundary_checkpoint_workspace_bytes,
            draft_direct_publication_workspace_bytes,
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
}
