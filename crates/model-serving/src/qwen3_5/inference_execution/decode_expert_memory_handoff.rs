//! Prefill-to-decode expert-residency preparation for one user request.
//!
//! # What the user is waiting for
//!
//! After the model finishes reading the prompt, it starts writing tokens. Token
//! writing is much cheaper in activation memory than prompt reading. The RAM the
//! user already granted can therefore preserve more expert weights so generation
//! does not stream every routed weight from the solid-state drive.
//!
//! # Words used in this file
//!
//! - Stable complete layer: all experts for one decoder layer remain retained.
//! - Elastic routed page: exact experts already required by a decode route remain retained.
//! - Operation-local page: experts are released after the mandatory forward.
//! - Temporary request-pressure cap: a smaller retained-page ceiling installed
//!   so the remaining prompt can finish. It is not the user's normal RAM grant.
//!
//! # Why this barrier exists
//!
//! Prefill may demote the complete owner and freeze retained pages so the last
//! prompt chunks still fit. If that freeze is left in place, decode sees a tiny
//! leftover budget, rejects useful retained pages, and streams from disk even though
//! tens of gigabytes are free. This file is the one place that:
//!
//! 1. Releases the temporary cap after the last prefill cleanup barrier.
//! 2. Restores complete RAM ownership when the leftover ceiling admits it.
//! 3. Reconciles a pure decode topology target against already-owned pages.
//! 4. Seats complete layers `memory/` named that decode would never load itself.
//!
//! Reconciliation is best-effort. A failed plan must not fail the user's
//! request; decode can still stream missing routes.

use astronomical_ipc_protocol::RequestId;

use crate::qwen3_5_moe::Qwen3_5ExpertResidencyTransitionReason;
use crate::{InferenceEngineError, MemoryPhase, PerformanceOperation};

use super::{Qwen3_5EngineState, fatal_engine_error};

impl Qwen3_5EngineState {
    /// Reconciles retained ownership with the leftover ceiling after prefill.
    ///
    /// Call this exactly once, after the last prefill chunk has synchronized and
    /// cleaned allocator storage, and before the first decode forward. The
    /// request flag `generation_residency_preparation_attempted` is the one-shot guard.
    pub(super) fn prepare_decode_expert_residency_after_prefill(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<(), InferenceEngineError> {
        let Some(model) = self.model.as_mut() else {
            return Err(fatal_engine_error("Qwen3.5 engine lost its loaded model"));
        };
        // Prefill pressure protects the remaining prompt by installing a
        // temporary retained-page ceiling. That cap must die here. Decode uses
        // a smaller activation footprint, so the leftover composed budget is
        // the user's real grant again. Leaving the cap in place was the bug
        // that kept generation at about one gigabyte of experts.
        let resumed_after_prefill_memory_pressure =
            model.resume_expert_retention_after_request_memory_pressure();
        if resumed_after_prefill_memory_pressure {
            tracing::info!(
                request_id = request_id.value(),
                "released prefill request-pressure expert retention ceiling before decode"
            );
        }
        // Cache restore and prefill may demote a fitting model for a temporary
        // workspace. Decode no longer needs that workspace, so put every expert
        // back in RAM when the ceiling still admits it.
        if let Err(residency_restore_error) = model.try_promote_experts_to_resident(
            Qwen3_5ExpertResidencyTransitionReason::DecodeHandoff,
            &mut active_request.performance_attribution,
        ) {
            tracing::warn!(
                request_id = request_id.value(),
                error = %residency_restore_error,
                "complete expert residency restore before decode failed; paging remains"
            );
        }
        let context_token_count =
            u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX);
        let residency_preparation_result =
            active_request.performance_attribution.measure_operation(
                PerformanceOperation::GenerationPreparation,
                |performance_attribution| {
                    model.refresh_phase_aware_expert_residency_plan(
                        MemoryPhase::GenerationPreparation,
                        context_token_count,
                        performance_attribution,
                    )
                },
            );
        if let Err(residency_preparation_error) = residency_preparation_result {
            // Residency planning is an accelerator, not a correctness gate. A
            // missing plan makes every uncovered route operation-local, which is
            // slower but preserves exact model execution and the user's request.
            model.clear_phase_aware_expert_residency_plan();
            tracing::warn!(
                request_id = request_id.value(),
                error = %residency_preparation_error,
                "continued decode with operation-local expert streaming after optional residency planning failed"
            );
        }
        let complete_layer_indexes_to_seat =
            model.planned_complete_layer_indexes_to_seat_before_decode();
        if !complete_layer_indexes_to_seat.is_empty() {
            match model.seat_complete_layers_before_decode(
                &complete_layer_indexes_to_seat,
                &mut active_request.performance_attribution,
            ) {
                Ok(seated_payload_bytes) => {
                    tracing::info!(
                        request_id = request_id.value(),
                        planned_complete_layer_count = complete_layer_indexes_to_seat.len(),
                        seated_payload_bytes,
                        "seated planned complete layers before decode"
                    );
                }
                Err(seating_error) => {
                    tracing::warn!(
                        request_id = request_id.value(),
                        error = %seating_error,
                        "continued decode after planned complete-layer seating failed"
                    );
                }
            }
        }
        // Internal ownership log: this deliberately reports the retained cache's
        // seated bookkeeping, not the published measured claim — the seating pass
        // just enqueued these lazy pages and no snapshot is taken here.
        let expert_statistics = model.expert_weight_memory_cache_statistics();
        let total_layer_count = model
            .expert_pager
            .as_ref()
            .map_or(0, |expert_pager| expert_pager.layer_count());
        tracing::info!(
            request_id = request_id.value(),
            context_token_count,
            total_layer_count,
            resident_expert_count = expert_statistics.entry_count,
            resident_expert_payload_bytes = expert_statistics.resident_payload_byte_count,
            "generation preparation seated leftover complete layers before decode"
        );
        Ok(())
    }
}
