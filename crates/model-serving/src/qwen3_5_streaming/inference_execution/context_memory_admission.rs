use crate::qwen3_5_streaming::model::memory_admission::validate_context_memory_admission;
use crate::{InferenceEngineError, PerformanceAttribution, PerformanceOperation};

use super::Qwen3_5EngineState;

impl Qwen3_5EngineState {
    pub(super) fn validate_context_memory_admission(
        &mut self,
        context_token_count_requiring_reservation: usize,
        temporary_workspace_reservation_bytes: usize,
        additional_maximum_expert_page_reservation_bytes: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<u64, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let retained_expert_payload_bytes_before_admission = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let admission_result = performance_attribution.measure_operation(
            PerformanceOperation::MemoryAdmissionSnapshot,
            |_performance_attribution| {
                validate_context_memory_admission(
                    model,
                    self.memory_limits,
                    self.context_memory_reservation_bytes_per_token,
                    context_token_count_requiring_reservation,
                    temporary_workspace_reservation_bytes,
                    additional_maximum_expert_page_reservation_bytes,
                )
            },
        );
        let retained_expert_payload_bytes_after_admission = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        admission_result?;
        Ok(retained_expert_payload_bytes_before_admission
            .saturating_sub(retained_expert_payload_bytes_after_admission))
    }
}
