//! Request-pressure control for Rust-owned expert layer residency.
//!
//! Expert retention may yield memory to decoder context. Reclamation includes a
//! stream barrier, allocator cleanup, and a fresh sample before retries continue.

use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::InferenceEngineError;

use crate::qwen3_5::inference_execution::qwen3_5_runtime_error;
use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    pub(crate) fn limit_expert_retention_for_request_memory_pressure(
        &self,
        retained_expert_payload_reclamation_target_bytes: usize,
    ) -> Result<bool, crate::qwen3_5::model::Qwen3_5ExecutionError> {
        let Some(retained_expert_layers) = self.retained_expert_layers.as_ref() else {
            return Ok(false);
        };
        Ok(retained_expert_layers
            .borrow_mut()
            .limit_for_request_pressure(
                u64::try_from(retained_expert_payload_reclamation_target_bytes).unwrap_or(u64::MAX),
            ))
    }

    pub(crate) fn resume_expert_retention_after_request_memory_pressure(&self) -> bool {
        if self.resident_expert_weights.is_some() {
            return false;
        }
        self.retained_expert_layers
            .as_ref()
            .is_some_and(|retained_expert_layers| {
                retained_expert_layers
                    .borrow_mut()
                    .resume_after_request_pressure()
            })
    }
}

pub(crate) fn reclaim_retained_experts_for_request_memory_pressure(
    model: &Qwen3_5Model,
    retained_expert_payload_reclamation_target_bytes: usize,
) -> Result<Option<MlxMemorySnapshot>, InferenceEngineError> {
    if !model
        .limit_expert_retention_for_request_memory_pressure(
            retained_expert_payload_reclamation_target_bytes,
        )
        .map_err(InferenceEngineError::from)?
    {
        return Ok(None);
    }
    // The pager first retires streaming ownership and releases Rust-selected
    // complete persistent layers. Clear allocator buffers before measuring the
    // physical effect of that topology change.
    if let Err(allocator_reclamation_error) = model
        .runtime()
        .synchronize_gpu_stream_and_clear_allocator_cache()
    {
        // A failed cleanup must not strand the model at a request-scoped frozen
        // ceiling once this recovery attempt has already failed.
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
