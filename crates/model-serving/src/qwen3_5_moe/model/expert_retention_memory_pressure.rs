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
//! That freeze lives in `RetainedExpertPageCache` as
//! `request_pressure_maximum_resident_payload_bytes`. The long-lived normal
//! maximum is stored beside it so this file can later remove the freeze
//! without guessing the machine budget again.
//!
//! # When the freeze must end
//!
//! The last prefill chunk has a synchronization and allocator-cleanup barrier.
//! After that barrier, decode activations are small. Leaving the freeze in
//! place hides the generation-phase retained budget and rejects useful existing
//! ownership. `resume_expert_retention_after...` only lifts the cap. The caller
//! reconciles existing topology without reading storage; later mandatory
//! forwards may install newly read complete or routed pages.
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
    /// Caps retained expert ownership at an absolute maximum, reclaiming
    /// over-ceiling tables immediately. Decode warming (issue #372) uses this
    /// at the post-evaluation flush so the hot-expert cache can only grow into
    /// the adaptive growth guard's unclaimed headroom, never into the margin
    /// the next forward's admission must hold for transients and KV growth.
    /// Returns whether owned arrays were released.
    pub(crate) fn limit_retained_experts_to(&self, maximum_resident_payload_bytes: u64) -> bool {
        self.retained_experts
            .as_ref()
            .is_none_or(|retained_experts| {
                retained_experts
                    .borrow_mut()
                    .limit_for_request_pressure_to_maximum(maximum_resident_payload_bytes)
            })
    }

    /// Releases every elastic (hot-expert warmed) retained table without
    /// touching pinned complete layers. Request finalization uses this so one
    /// request's decode warming cannot crowd the next request's prefill
    /// admission under a tight memory ceiling. Returns released payload bytes.
    pub(crate) fn release_elastic_routed_tables(&self) -> u64 {
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return 0;
        };
        let mut cache = retained_experts.borrow_mut();
        let mut released_payload_bytes = 0_u64;
        // The real expert capacity decides the elastic/complete classification:
        // a table holding every expert is a pinned complete layer and must
        // survive; only strictly partial tables are warm-table elastic.
        let expert_capacity = self
            .expert_pager
            .as_ref()
            .and_then(|expert_pager| expert_pager.layer_plans().first())
            .map_or(0, |layer_plan| layer_plan.expert_capacity);
        for residency in cache.topology_snapshot(expert_capacity) {
            // Warm tables are elastic by class; complete layers are pinned by
            // the request-stable contract and must survive finalization.
            if residency.class == crate::RetainedExpertPageClass::ElasticRoutedExperts
                && cache.remove_layer(residency.layer_index)
            {
                released_payload_bytes =
                    released_payload_bytes.saturating_add(residency.payload_bytes);
            }
        }
        released_payload_bytes
    }
    /// Constrains retained experts to the strict-ceiling capacity left by one
    /// concrete forward admission. Returns whether owned arrays were released.
    pub(crate) fn limit_expert_retention_for_admitted_forward(
        &self,
        current_active_memory_bytes: u64,
        current_retained_expert_payload_bytes: u64,
        admitted_forward_reserve_bytes: u64,
    ) -> bool {
        let retained_expert_budget_bytes = self
            .mlx_ram_budget
            .borrow()
            .retained_expert_budget_for_admitted_forward(
                current_active_memory_bytes,
                current_retained_expert_payload_bytes,
                admitted_forward_reserve_bytes,
            );
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return false;
        };
        let released_retained_experts = retained_experts
            .borrow_mut()
            .limit_for_request_pressure_to_maximum(retained_expert_budget_bytes);
        if released_retained_experts {
            tracing::info!(
                current_active_memory_bytes,
                current_retained_expert_payload_bytes,
                admitted_forward_reserve_bytes,
                retained_expert_budget_bytes,
                "released retained experts to the exact admitted-forward budget"
            );
        } else {
            tracing::debug!(
                current_active_memory_bytes,
                current_retained_expert_payload_bytes,
                admitted_forward_reserve_bytes,
                retained_expert_budget_bytes,
                "admitted-forward expert budget still covers current retention"
            );
        }
        released_retained_experts
    }

    /// Installs the temporary retained-page ceiling and evicts pages that no
    /// longer fit. Returns whether any page was actually released.
    pub(crate) fn limit_expert_retention_for_request_memory_pressure(
        &self,
        retained_expert_payload_reclamation_target_bytes: usize,
    ) -> Result<bool, crate::qwen3_5::model::Qwen3_5ExecutionError> {
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return Ok(false);
        };
        Ok(retained_experts.borrow_mut().limit_for_request_pressure(
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
        self.retained_experts
            .as_ref()
            .is_some_and(|retained_experts| {
                retained_experts
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
