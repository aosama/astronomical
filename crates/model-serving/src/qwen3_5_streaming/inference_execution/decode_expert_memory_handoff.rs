//! The engine's side of preparing expert retention for a reply.
//!
//! The model decides and does the work (`model/reply_expert_residency`):
//! the memory governor is told the reply begins, the layers are planned and
//! seated, and the hot-expert tables are sized. This file calls it once per
//! request and writes what it did to the request's log.

use astronomical_ipc_protocol::RequestId;

use crate::InferenceEngineError;

use super::Qwen3_5EngineState;

impl Qwen3_5EngineState {
    /// Prepares the model's expert retention for the reply and logs it.
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
            return Err(super::fatal_engine_error(
                "Qwen3.5 engine lost its loaded model",
            ));
        };
        let context_token_count =
            u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX);
        let preparation =
            model.prepare_expert_residency_for_reply(&mut active_request.performance_attribution);
        if let Some(planning_error) = preparation.planning_error.as_ref() {
            tracing::warn!(
                request_id = request_id.value(),
                error = %planning_error,
                "continued decode with operation-local expert streaming after optional residency planning failed"
            );
        }
        match preparation.seating.as_ref() {
            Ok(seated_payload_bytes) => tracing::info!(
                request_id = request_id.value(),
                planned_complete_layer_count = preparation.planned_complete_layer_count,
                warm_floor_capped_complete_layer_count = preparation.seated_complete_layer_count,
                seated_payload_bytes,
                "seated planned complete layers before decode"
            ),
            Err(seating_error) => tracing::warn!(
                request_id = request_id.value(),
                error = %seating_error,
                "continued decode after planned complete-layer seating failed"
            ),
        }
        tracing::info!(
            request_id = request_id.value(),
            context_token_count,
            total_layer_count = preparation.total_layer_count,
            resident_expert_count = preparation.resident_expert_count,
            resident_expert_payload_bytes = preparation.resident_expert_payload_bytes,
            released_beyond_keep_count = preparation.released_beyond_keep_count,
            evicted_complete_layer_count = preparation.evicted_complete_layer_count,
            hot_expert_warm_slot_count = preparation.hot_expert_warm_slot_count,
            "expert retention prepared for the reply"
        );
        Ok(())
    }
}
