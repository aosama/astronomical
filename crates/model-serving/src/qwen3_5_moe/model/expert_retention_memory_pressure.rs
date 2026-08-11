//! Request-pressure control for the elastic native expert cache.
//!
//! Expert retention may yield memory to decoder context, but immutable native
//! snapshots can still own evicted page arrays until their graphics-processor
//! work completes. Reclamation therefore includes a stream barrier, allocator
//! cleanup, and a fresh memory sample before admission retries continue.

use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::InferenceEngineError;

use crate::qwen3_5::inference_execution::qwen3_5_runtime_error;
use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    pub(crate) fn freeze_expert_retention_growth_for_request_memory_pressure(&self) -> bool {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return false;
        };
        expert_pager.freeze_native_expert_retention_growth()
    }

    pub(crate) fn limit_expert_retention_for_request_memory_pressure(
        &self,
        retained_expert_payload_reclamation_target_bytes: usize,
    ) -> Result<bool, crate::qwen3_5::model::Qwen3_5ExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(false);
        };
        Ok(expert_pager.reclaim_native_expert_payload_bytes(
            u64::try_from(retained_expert_payload_reclamation_target_bytes).unwrap_or(u64::MAX),
        )?)
    }

    pub(crate) fn resume_expert_retention_after_request_memory_pressure(&self) -> bool {
        // Complete residency has no page-growth ceiling to release. Promotion
        // already froze native retention, which must remain dormant in this mode.
        if self.resident_expert_weights.is_some() {
            return false;
        }
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return false;
        };
        expert_pager.resume_native_expert_retention_growth()
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
    // Removing native cache entries only releases policy ownership. Synchronize
    // first so completed snapshots can drop their final MLX references, then
    // clear only reclaimable allocator buffers before measuring the effect.
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
