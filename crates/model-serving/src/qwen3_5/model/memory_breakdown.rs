use astronomical_runtime_integration::MlxArray;

use crate::MlxActiveMemoryBreakdown;
use crate::PerformanceAttribution;
use crate::PerformanceCounter;
use crate::memory::MemoryCeilingUtilization;

use super::{Qwen3_5Model, Qwen3_5VisionModel, RequestDecoderStateStack};

impl Qwen3_5Model {
    #[must_use]
    pub(crate) fn active_memory_breakdown(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        additional_context_state_payload_bytes: u64,
        mlx_active_memory_bytes: u64,
        additional_model_core_payload_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        let context_state_payload_bytes = request_decoder_state
            .payload_byte_count()
            .saturating_add(additional_context_state_payload_bytes);
        self.active_memory_breakdown_with_context_state_payload_bytes(
            context_state_payload_bytes,
            mlx_active_memory_bytes,
            additional_model_core_payload_bytes,
        )
    }

    #[must_use]
    pub(crate) fn finalized_active_memory_breakdown(
        &self,
        mlx_active_memory_bytes: u64,
        additional_model_core_payload_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        self.active_memory_breakdown_with_context_state_payload_bytes(
            0,
            mlx_active_memory_bytes,
            additional_model_core_payload_bytes,
        )
    }

    #[must_use]
    pub(crate) fn active_memory_breakdown_with_speculative_prefill_draft(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        additional_context_state_payload_bytes: u64,
        mlx_active_memory_bytes: u64,
        draft_model: &Self,
        draft_request_decoder_state: &RequestDecoderStateStack,
        draft_visual_embeddings: Option<&MlxArray>,
    ) -> MlxActiveMemoryBreakdown {
        let context_state_payload_bytes = request_decoder_state
            .payload_byte_count()
            .saturating_add(additional_context_state_payload_bytes);
        let target_model_core_payload_bytes =
            self.resident_model_payload_byte_count().saturating_add(
                self.vision_model
                    .as_ref()
                    .map_or(0, Qwen3_5VisionModel::resident_payload_bytes),
            );
        let draft_model_payload_bytes = draft_model
            .resident_model_payload_byte_count()
            .saturating_add(
                draft_model
                    .vision_model
                    .as_ref()
                    .map_or(0, Qwen3_5VisionModel::resident_payload_bytes),
            )
            .saturating_add(
                draft_model
                    .expert_weight_memory_cache_statistics()
                    .resident_payload_byte_count,
            )
            .saturating_add(draft_request_decoder_state.payload_byte_count())
            .saturating_add(
                draft_visual_embeddings.map_or(0, |draft_visual_embeddings| {
                    u64::try_from(draft_visual_embeddings.byte_count()).unwrap_or(u64::MAX)
                }),
            );
        MlxActiveMemoryBreakdown::reconcile_with_speculative_prefill_draft(
            mlx_active_memory_bytes,
            self.expert_weight_memory_cache_statistics()
                .resident_payload_byte_count,
            target_model_core_payload_bytes,
            context_state_payload_bytes,
            draft_model_payload_bytes,
        )
    }

    /// Composes the ceiling-utilization split for one measured instant (issue #510).
    ///
    /// The budget owner is the single source of the reserve arithmetic, so the
    /// published split and the engine's own admission decisions can never
    /// drift apart. Callers pass the breakdown reconciled from the same
    /// measurement, keeping every term pinned to one instant.
    pub(crate) fn memory_ceiling_utilization_for_breakdown(
        &self,
        phase: crate::MemoryPhase,
        context_token_count: u64,
        mlx_active_memory_bytes: u64,
        active_breakdown: MlxActiveMemoryBreakdown,
    ) -> MemoryCeilingUtilization {
        MemoryCeilingUtilization::compose(
            self.mlx_ram_budget
                .borrow()
                .plan(phase, context_token_count, 0),
            mlx_active_memory_bytes,
            active_breakdown,
        )
    }

    /// Records why part of the MLX ceiling is unused at one decode step (issue #507).
    ///
    /// Only the step with the largest unused headroom is kept, and all terms are
    /// recorded together from that same step. Recording each term as its own
    /// maximum would mix terms from different steps and break the identity the
    /// acceptance journey asserts. `record_maximum_counter` alone cannot express
    /// that, so the peak is read back before re-recording the whole split.
    pub(crate) fn record_memory_ceiling_utilization(
        &self,
        phase: crate::MemoryPhase,
        context_token_count: u64,
        request_decoder_state: &RequestDecoderStateStack,
        additional_context_state_payload_bytes: u64,
        mlx_active_memory_bytes: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let active_breakdown = self.active_memory_breakdown(
            request_decoder_state,
            additional_context_state_payload_bytes,
            mlx_active_memory_bytes,
            0,
        );
        let utilization = self.memory_ceiling_utilization_for_breakdown(
            phase,
            context_token_count,
            mlx_active_memory_bytes,
            active_breakdown,
        );
        let previous_peak_unused_headroom_bytes = performance_attribution
            .counter_value(PerformanceCounter::MemoryCeilingUtilizationUnusedHeadroomBytes);
        if utilization.unused_headroom_bytes < previous_peak_unused_headroom_bytes {
            return;
        }
        for (counter, amount) in [
            (
                PerformanceCounter::MemoryCeilingUtilizationCeilingBytes,
                utilization.mlx_active_memory_ceiling_bytes,
            ),
            (
                PerformanceCounter::MemoryCeilingUtilizationActiveBytes,
                utilization.active_memory_bytes,
            ),
            (
                PerformanceCounter::MemoryCeilingUtilizationUnusedHeadroomBytes,
                utilization.unused_headroom_bytes,
            ),
            (
                PerformanceCounter::MemoryCeilingUtilizationReservedModelCoreSlackBytes,
                utilization.reserved_model_core_slack_bytes,
            ),
            (
                PerformanceCounter::MemoryCeilingUtilizationReservedContextGrowthBytes,
                utilization.reserved_context_growth_bytes,
            ),
            (
                PerformanceCounter::MemoryCeilingUtilizationReservedActivationAndWorkspaceBytes,
                utilization.reserved_activation_and_workspace_bytes,
            ),
            (
                PerformanceCounter::MemoryCeilingUtilizationUnseatedExpertEntitlementBytes,
                utilization.unseated_expert_entitlement_bytes,
            ),
            (
                PerformanceCounter::MemoryCeilingUtilizationUnexplainedHeadroomBytes,
                utilization.unexplained_headroom_bytes,
            ),
            (
                PerformanceCounter::MemoryCeilingUtilizationOwnerOverrunBytes,
                utilization.owner_overrun_bytes,
            ),
        ] {
            performance_attribution.record_snapshot_counter(counter, amount);
        }
    }

    fn active_memory_breakdown_with_context_state_payload_bytes(
        &self,
        context_state_payload_bytes: u64,
        mlx_active_memory_bytes: u64,
        additional_model_core_payload_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        let paged_expert_payload_bytes = self
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let expert_payload_bytes = paged_expert_payload_bytes;
        let model_core_payload_bytes = self
            .resident_model_payload_byte_count()
            .saturating_add(
                self.vision_model
                    .as_ref()
                    .map_or(0, Qwen3_5VisionModel::resident_payload_bytes),
            )
            .saturating_add(additional_model_core_payload_bytes);
        MlxActiveMemoryBreakdown::reconcile(
            mlx_active_memory_bytes,
            expert_payload_bytes,
            model_core_payload_bytes,
            context_state_payload_bytes,
        )
    }
}
