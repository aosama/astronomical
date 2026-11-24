//! The ownership view of MLX active memory for both Qwen3.5 engines.
//!
//! The breakdown and the ceiling-utilization split are the same arithmetic in
//! both engines; only two figures differ. The resident engine holds every
//! expert while the streaming engine holds a retained subset, and the
//! streaming engine's model-core bytes exclude dense layers it reads from the
//! SSD on each forward. Each engine supplies those two figures through
//! [`Qwen3_5MemoryOwnership`]; the arithmetic itself is written once.
//!
//! The arithmetic reads the vision tower through [`Qwen3_5ServingModel`] and
//! the ceiling through a narrow method rather than through a
//! `Qwen3_5ModelBase`, so a text model with other weights, the folded Prism
//! Hadamard pack, composes the same breakdown without owning a base.

use crate::PerformanceAttribution;
use crate::PerformanceCounter;
use crate::memory::MemoryCeilingUtilization;
use crate::memory::telemetry::MlxActiveMemoryBreakdown;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::vision::Qwen3_5VisionModel;

use super::Qwen3_5ServingModel;

/// What one engine model holds in MLX memory, and the breakdown built on it.
pub trait Qwen3_5MemoryOwnership: Qwen3_5ServingModel {
    /// The MLX active-memory ceiling the model is serving under, which the
    /// split sets the reserves against before the memory governor opens.
    fn mlx_active_memory_ceiling_bytes(&self) -> u64;

    /// Expert weight bytes this engine currently holds in MLX memory.
    fn held_expert_payload_bytes(&self) -> u64;

    /// Text weight bytes, other than experts, this engine keeps in memory.
    fn held_model_core_payload_bytes(&self) -> u64;

    /// The memory governor's accounts in the shape the ceiling-utilization
    /// split is composed against, so the split shows what is really set
    /// aside (issue #2221). `None` until the engine's governor has opened.
    fn governed_memory_reserve_snapshot(&self) -> Option<crate::memory::MemoryReserveSnapshot>;

    /// The reserves the split is composed against. Before the governor
    /// opens nothing has been measured, so only the weights are set aside.
    fn memory_reserve_snapshot(&self) -> crate::memory::MemoryReserveSnapshot {
        match self.governed_memory_reserve_snapshot() {
            Some(governed_budget_snapshot) => governed_budget_snapshot,
            None => crate::memory::MemoryReserveSnapshot::with_only_weights_set_aside(
                self.mlx_active_memory_ceiling_bytes(),
                self.held_model_core_payload_bytes(),
            ),
        }
    }

    #[must_use]
    fn active_memory_breakdown(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        additional_context_state_payload_bytes: u64,
        mlx_active_memory_bytes: u64,
        additional_model_core_payload_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        let context_state_payload_bytes = request_decoder_state
            .payload_byte_count()
            .saturating_add(additional_context_state_payload_bytes);
        breakdown_with_context_state_payload_bytes(
            self,
            context_state_payload_bytes,
            mlx_active_memory_bytes,
            additional_model_core_payload_bytes,
        )
    }

    /// Composes the ownership view of a finalized MLX active-memory measurement.
    ///
    /// `live_context_state_payload_bytes` carries the payload of decoder state
    /// still live at the measured instant — after a request that is the
    /// retained live prefix (issue #1167). Attributing those bytes to the
    /// context owner keeps the published ceiling-utilization split honest
    /// (issue #1236): reporting zero context state charged the retained
    /// prefix against the activation-and-workspace reserve, a one-decode-step
    /// transient reserve far narrower than the retained state, and the clamp
    /// surfaced the excess as a phantom owner overrun. The streaming engine
    /// retains no live prefix between requests and passes zero.
    #[must_use]
    fn finalized_active_memory_breakdown(
        &self,
        mlx_active_memory_bytes: u64,
        live_context_state_payload_bytes: u64,
        additional_model_core_payload_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        breakdown_with_context_state_payload_bytes(
            self,
            live_context_state_payload_bytes,
            mlx_active_memory_bytes,
            additional_model_core_payload_bytes,
        )
    }

    /// Composes the ceiling-utilization split for one measured instant (issue #510).
    ///
    /// The reserves come from whatever decides them for this engine
    /// (`memory_reserve_snapshot`), so the published split and the
    /// engine's own decisions can never drift apart. Callers pass the
    /// breakdown reconciled from the same measurement, keeping every term
    /// pinned to one instant.
    fn memory_ceiling_utilization_for_breakdown(
        &self,
        mlx_active_memory_bytes: u64,
        active_breakdown: MlxActiveMemoryBreakdown,
    ) -> MemoryCeilingUtilization {
        MemoryCeilingUtilization::compose(
            self.memory_reserve_snapshot(),
            mlx_active_memory_bytes,
            active_breakdown,
        )
    }

    /// Composes the split for a sample taken between requests. The governor
    /// keeps a prefix retained for the next turn on its stored-state
    /// account, so the same accounts serve.
    fn between_request_memory_ceiling_utilization_for_breakdown(
        &self,
        mlx_active_memory_bytes: u64,
        finalized_active_breakdown: MlxActiveMemoryBreakdown,
    ) -> MemoryCeilingUtilization {
        MemoryCeilingUtilization::compose(
            self.memory_reserve_snapshot(),
            mlx_active_memory_bytes,
            finalized_active_breakdown,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Records why part of the MLX ceiling is unused at one decode step (issue #507).
    ///
    /// Only the step with the largest unused headroom is kept, and all terms are
    /// recorded together from that same step. Recording each term as its own
    /// maximum would mix terms from different steps and break the identity the
    /// acceptance journey asserts. `record_maximum_counter` alone cannot express
    /// that, so the peak is read back before re-recording the whole split.
    fn record_memory_ceiling_utilization(
        &self,
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
        let utilization = self
            .memory_ceiling_utilization_for_breakdown(mlx_active_memory_bytes, active_breakdown);
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
}

/// The breakdown for a known context-state payload. A free function rather
/// than a trait method so it stays private to this module: callers compose
/// the context term through the trait's public entry points.
fn breakdown_with_context_state_payload_bytes(
    memory_owner: &(impl Qwen3_5MemoryOwnership + ?Sized),
    context_state_payload_bytes: u64,
    mlx_active_memory_bytes: u64,
    additional_model_core_payload_bytes: u64,
) -> MlxActiveMemoryBreakdown {
    let expert_payload_bytes = memory_owner.held_expert_payload_bytes();
    let model_core_payload_bytes = memory_owner
        .held_model_core_payload_bytes()
        .saturating_add(
            memory_owner
                .vision_model()
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
