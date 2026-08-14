//! Temporary expert-RAM freeze while one request is still reading its prompt.
//!
//! # Why a temporary cap exists
//!
//! A long prompt plus an image can need more activation RAM than the leftover
//! ceiling after complete experts are resident. The engine then demotes the
//! complete owner and asks this file to shrink retained pages. The shrink is
//! not "throw experts away forever". It is "freeze the retained-page ceiling
//! at a smaller number until the remaining prompt chunks finish".
//!
//! That freeze lives in `RetainedExpertLayerCache` as
//! `request_pressure_maximum_resident_payload_bytes`. The long-lived normal
//! maximum is stored beside it so this file can later remove the freeze
//! without guessing the machine budget again.
//!
//! # When the freeze must end
//!
//! The last prefill chunk has a synchronization and allocator-cleanup barrier.
//! After that barrier, decode activations are small. Leaving the freeze in
//! place makes decode-warm believe the leftover budget is about one gigabyte
//! and reject every demand-selected page. `resume_expert_retention_after...`
//! only lifts the cap. The caller then decides whether to promote the complete
//! owner or fill demand-selected pages.
//!
//! Callers that lift the cap today:
//! - decode handoff, after the last prefill barrier
//! - request finalization, after request-owned arrays are dropped
//! - failed reclaim cleanup, so a failed recovery cannot strand the model

use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::InferenceEngineError;

use crate::qwen3_5::inference_execution::qwen3_5_runtime_error;
use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    /// Installs the temporary retained-page ceiling and evicts pages that no
    /// longer fit. Returns whether any page was actually released.
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

    /// Removes the temporary request-pressure ceiling.
    ///
    /// This does not load pages and does not restore the complete owner. It
    /// only makes the long-lived normal budget visible again. Returns `true`
    /// when a freeze was actually present.
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

/// Shrinks retained pages, then proves the physical effect with a fresh sample.
///
/// Order matters: retire pages, synchronize the graphics-processor stream,
/// clear allocator cache, then snapshot. Sampling before cleanup would still
/// count released buffers that the allocator has not given back.
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
