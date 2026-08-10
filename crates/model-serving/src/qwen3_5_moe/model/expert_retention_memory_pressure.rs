use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::InferenceEngineError;

use crate::qwen3_5::inference_execution::qwen3_5_runtime_error;
use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    pub(crate) fn freeze_expert_retention_growth_for_request_memory_pressure(&self) -> bool {
        let Some(expert_weight_memory_cache) = self.expert_weight_memory_cache.as_ref() else {
            return false;
        };
        if self.expert_pager.is_none() {
            return false;
        }
        expert_weight_memory_cache
            .borrow_mut()
            .freeze_retention_growth_for_request_memory_pressure()
    }

    pub(crate) fn limit_expert_retention_for_request_memory_pressure(
        &self,
        retained_expert_payload_reclamation_target_bytes: usize,
    ) -> bool {
        let Some(expert_weight_memory_cache) = self.expert_weight_memory_cache.as_ref() else {
            return false;
        };
        if self.expert_pager.is_none() {
            return false;
        }
        let expert_weight_memory_cache_statistics = self.expert_weight_memory_cache_statistics();
        let maximum_retained_payload_byte_count = expert_weight_memory_cache_statistics
            .resident_payload_byte_count
            .saturating_sub(
                u64::try_from(retained_expert_payload_reclamation_target_bytes).unwrap_or(u64::MAX),
            );
        expert_weight_memory_cache
            .borrow_mut()
            .limit_retention_for_request_memory_pressure(maximum_retained_payload_byte_count);
        true
    }

    pub(crate) fn resume_expert_retention_after_request_memory_pressure(&self) -> bool {
        let Some(expert_weight_memory_cache) = self.expert_weight_memory_cache.as_ref() else {
            return false;
        };
        expert_weight_memory_cache
            .borrow_mut()
            .resume_retention_after_request_memory_pressure()
    }
}

pub(crate) fn reclaim_retained_experts_for_request_memory_pressure(
    model: &Qwen3_5Model,
    retained_expert_payload_reclamation_target_bytes: usize,
) -> Result<Option<MlxMemorySnapshot>, InferenceEngineError> {
    if !model.limit_expert_retention_for_request_memory_pressure(
        retained_expert_payload_reclamation_target_bytes,
    ) {
        return Ok(None);
    }
    if let Err(allocator_reclamation_error) = model
        .runtime()
        .synchronize_gpu_stream_and_clear_allocator_cache()
    {
        model.resume_expert_retention_after_request_memory_pressure();
        return Err(qwen3_5_runtime_error(allocator_reclamation_error));
    }
    let memory_snapshot_after_reclamation = match model.runtime().memory_snapshot() {
        Ok(memory_snapshot_after_reclamation) => memory_snapshot_after_reclamation,
        Err(memory_snapshot_error) => {
            model.resume_expert_retention_after_request_memory_pressure();
            return Err(qwen3_5_runtime_error(memory_snapshot_error));
        }
    };
    Ok(Some(memory_snapshot_after_reclamation))
}
