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
//! 2. Reconciles a pure decode topology target against already-owned pages.
//! 3. Leaves all new ownership to later execution-required prefill/decode reads.
//!
//! Reconciliation is best-effort. A failed plan must not fail the user's
//! request; decode can still stream missing routes.

use astronomical_ipc_protocol::RequestId;

use crate::{ExpertResidencyPhase, InferenceEngineError, PerformanceOperation};

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
        let context_token_count =
            u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX);
        let residency_preparation_result =
            active_request.performance_attribution.measure_operation(
                PerformanceOperation::GenerationPreparation,
                |performance_attribution| {
                    model.refresh_phase_aware_expert_residency_plan(
                        ExpertResidencyPhase::GenerationPreparation,
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
        let expert_residency = model.expert_residency_telemetry();
        tracing::info!(
            request_id = request_id.value(),
            context_token_count,
            total_layer_count = expert_residency.total_layer_count,
            resident_expert_count = expert_residency.resident_expert_count,
            resident_expert_payload_bytes = expert_residency.resident_expert_payload_bytes,
            "generation preparation preserved expert topology without eager source reads"
        );
        Ok(())
    }
}
