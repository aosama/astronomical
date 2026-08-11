//! Exact memory admission for request-scoped drafter scoring.
//!
//! Target and drafter share one process-wide MLX active-memory ceiling. Before
//! loading/scoring the drafter, the engine projects every drafter allocation that
//! can overlap and reclaims only pageable target expert payload needed to fit.
//! Model-core weights, active target decoder state, and the configured ceiling
//! are never reduced as part of this operation.

/// Returns the target expert-retention payload that must be reclaimed before
/// drafter scoring can allocate its remaining decoder state and one expert page.
#[must_use]
pub(crate) const fn speculative_prefill_draft_scoring_reclamation_target_bytes(
    current_active_memory_bytes: usize,
    draft_scoring_reservation_bytes: usize,
    allowed_active_memory_bytes: usize,
) -> usize {
    // Saturating arithmetic makes the formula total: an already-over-limit
    // snapshot requests at least the excess, while a fitting projection returns
    // exactly zero instead of underflowing.
    current_active_memory_bytes
        .saturating_add(draft_scoring_reservation_bytes)
        .saturating_sub(allowed_active_memory_bytes)
}

/// Combines the independently owned drafter allocations required before its
/// scoring graph can run.
///
/// Decoder growth and visual payload are long-lived for scoring. The expert page,
/// boundary checkpoint, and direct-publication workspace are transient but may
/// overlap at a cache boundary, so admission must reserve all five categories.
#[must_use]
pub(crate) fn speculative_prefill_draft_scoring_reservation_bytes(
    draft_decoder_state_growth_bytes: usize,
    draft_vision_payload_bytes: usize,
    draft_maximum_expert_page_reservation_bytes: usize,
    draft_boundary_checkpoint_workspace_bytes: usize,
    draft_direct_publication_workspace_bytes: usize,
) -> Option<usize> {
    // Overflow means admission cannot prove the request fits and must fail
    // closed. `Option` keeps the pure formula independent of engine error types.
    draft_decoder_state_growth_bytes
        .checked_add(draft_vision_payload_bytes)?
        .checked_add(draft_maximum_expert_page_reservation_bytes)?
        .checked_add(draft_boundary_checkpoint_workspace_bytes)?
        .checked_add(draft_direct_publication_workspace_bytes)
}

#[cfg(feature = "direct-mlx")]
use super::super::super::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
use super::super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
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
        &self,
        request_id: u64,
        draft_model: &Qwen3_5Model,
        draft_request_decoder_state: &RequestDecoderStateStack,
        draft_suffix_token_count: usize,
        should_reserve_draft_vision_payload: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        // Target runtime accounting is the process-wide baseline against which
        // all request-scoped drafter ownership must fit.
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
        // Paged drafts can transiently require their largest routed expert page;
        // resident/dense drafts report zero through this helper.
        let draft_maximum_expert_page_reservation_bytes =
            self.speculative_prefill_draft_maximum_expert_page_reservation_bytes();
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
        let draft_scoring_reservation_bytes = speculative_prefill_draft_scoring_reservation_bytes(
            draft_decoder_state_growth_bytes,
            draft_vision_payload_bytes,
            draft_maximum_expert_page_reservation_bytes,
            draft_boundary_checkpoint_workspace_bytes,
            draft_direct_publication_workspace_bytes,
        )
        .ok_or_else(|| {
            fatal_engine_error("speculative-prefill draft scoring reservation overflowed")
        })?;
        let memory_snapshot_before_draft_scoring = target_model
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        // Only pageable target experts are reclaimable here. Target core weights,
        // active request state, and the configured ceiling remain unchanged.
        let required_target_expert_reclamation_bytes =
            speculative_prefill_draft_scoring_reclamation_target_bytes(
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

        // Capture relational expert statistics around reclamation. MLX active
        // memory can include graph work, while this payload delta precisely
        // identifies retained target experts released for the drafter.
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
