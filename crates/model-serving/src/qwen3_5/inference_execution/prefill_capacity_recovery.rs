//! Exact one-retry recovery for a fixed prompt-processing chunk.
//!
//! Runtime allocation pressure can occur after predictive admission because MLX
//! evaluates lazy graphs and owns allocator details unavailable to Rust planning.
//! Recovery must preserve model correctness and configuration semantics:
//!
//! 1. Restore the request checkpoint captured before the failed attempt.
//! 2. Release allocator cache using the synchronization rule appropriate to the
//!    failure type.
//! 3. Reconstruct the fixed workspace needed by the unchanged chunk.
//! 4. Demote a complete resident owner when present; paged reclamation cannot nibble it.
//! 5. Then reclaim elastic retained expert pages.
//! 6. Authorize one retry only when ownership proves the target was released.
//!
//! The two public helpers differ because typed active-limit rejection exposes the
//! failed allocation size and may still synchronize cleanly, whereas GPU OOM can
//! poison the synchronization event chain and exposes no allocation size.

use astronomical_ipc_protocol::RequestId;

use crate::qwen3_5_moe::{
    Qwen3_5ExpertResidencyTransitionReason, reclaim_retained_experts_for_request_memory_pressure,
};
use crate::{
    ForwardRecoveryPolicy, InferenceEngineError, PerformanceCounter, PerformanceOperation,
};

use super::engine_request::{Qwen3_5EngineRequest, Qwen3_5PrefillRequestCheckpoint};
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

impl Qwen3_5EngineState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn recover_fixed_prefill_chunck_after_active_memory_limit(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        attempted_prefill_chunck_token_count: usize,
        active_memory_bytes_at_failure: usize,
        attempted_allocation_bytes: usize,
        allowed_active_memory_bytes: usize,
        prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
        has_already_retried_after_reclamation: bool,
    ) -> Result<bool, InferenceEngineError> {
        record_rejection_and_restore_checkpoint(active_request, prefill_request_checkpoint)?;
        clear_allocator_cache_after_active_memory_limit(self, active_request)?;

        let memory_snapshot_before_reclamation = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        let retained_expert_payload_bytes = retained_expert_payload_bytes(
            self.model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?,
        );
        // The retry needs transient arrays already active at failure plus the
        // allocation that failed. They are additive, not alternative estimates.
        let fixed_forward_workspace_bytes = ForwardRecoveryPolicy::fixed_workspace_bytes(
            memory_snapshot_before_reclamation.active_memory_bytes(),
            active_memory_bytes_at_failure,
            attempted_allocation_bytes,
            self.adaptive_ram_growth_guard
                .admission_transient_high_water_bytes(),
        );
        let expert_reclamation_target_bytes = ForwardRecoveryPolicy::required_reclamation_bytes(
            memory_snapshot_before_reclamation.active_memory_bytes(),
            retained_expert_payload_bytes,
            allowed_active_memory_bytes,
            fixed_forward_workspace_bytes,
        );
        let retained_payload_before_reclamation =
            u64::try_from(retained_expert_payload_bytes).unwrap_or(u64::MAX);
        let did_demote_complete_resident_owner =
            demote_complete_resident_owner_for_prefill_recovery(self, active_request)?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let active_memory_bytes_after_reclamation = reclaim_and_sample_active_memory(
            model,
            expert_reclamation_target_bytes,
            memory_snapshot_before_reclamation.active_memory_bytes(),
        )?;
        let retained_payload_after_reclamation = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        model.shrink_request_expert_residency_after_reclamation(
            retained_payload_before_reclamation.saturating_sub(retained_payload_after_reclamation),
        );

        if active_request.should_use_speculative_prefill {
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
                retained_payload_before_reclamation
                    .saturating_sub(retained_payload_after_reclamation),
            );
        }
        let sparse_experts_are_paged = model.sparse_experts_are_paged();
        let should_retry_same_prefill_chunck = ForwardRecoveryPolicy::retry_is_authorized(
            has_already_retried_after_reclamation,
            retained_payload_before_reclamation,
            retained_payload_after_reclamation,
            expert_reclamation_target_bytes,
            sparse_experts_are_paged,
        );
        tracing::warn!(
            request_id = request_id.value(),
            attempted_prefill_chunck_token_count,
            active_memory_bytes_at_failure,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
            fixed_forward_workspace_bytes,
            expert_reclamation_target_bytes,
            retained_expert_payload_bytes,
            retained_payload_after_reclamation,
            active_memory_bytes_before_reclamation =
                memory_snapshot_before_reclamation.active_memory_bytes(),
            active_memory_bytes_after_reclamation,
            did_demote_complete_resident_owner,
            should_retry_same_prefill_chunck,
            "native MLX prefill allocation reached the active-memory ceiling; released resident or paged experts for the fixed chunk"
        );
        Ok(should_retry_same_prefill_chunck)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn recover_fixed_prefill_chunck_after_graphics_processor_exhaustion(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        attempted_prefill_chunck_token_count: usize,
        failure_reason: &str,
        prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
        has_already_retried_after_reclamation: bool,
    ) -> Result<bool, InferenceEngineError> {
        record_rejection_and_restore_checkpoint(active_request, prefill_request_checkpoint)?;
        // The failed synchronization already waited for command-buffer completion.
        // Synchronizing its poisoned event chain again would reproduce the error.
        clear_allocator_cache_without_stream_sync(self, active_request)?;

        let memory_snapshot_before_reclamation = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        let retained_expert_payload_bytes = retained_expert_payload_bytes(
            self.model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?,
        );
        let retained_payload_before_reclamation =
            u64::try_from(retained_expert_payload_bytes).unwrap_or(u64::MAX);
        // GPU OOM does not expose one failed allocation size. Use learned
        // transient high-water as the model-local fixed workspace estimate, with
        // one byte ensuring the formula represents actual required capacity.
        let fixed_forward_workspace_bytes = self
            .adaptive_ram_growth_guard
            .admission_transient_high_water_bytes()
            .max(1);
        let active_memory_limit_bytes = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .runtime()
            .memory_limits()
            .active_memory_limit_bytes();
        let expert_reclamation_target_bytes = ForwardRecoveryPolicy::required_reclamation_bytes(
            memory_snapshot_before_reclamation.active_memory_bytes(),
            retained_expert_payload_bytes,
            active_memory_limit_bytes,
            fixed_forward_workspace_bytes,
        );
        let did_demote_complete_resident_owner =
            demote_complete_resident_owner_for_prefill_recovery(self, active_request)?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let active_memory_bytes_after_reclamation = reclaim_and_sample_active_memory(
            model,
            expert_reclamation_target_bytes,
            memory_snapshot_before_reclamation.active_memory_bytes(),
        )?;
        let retained_payload_after_reclamation = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        model.shrink_request_expert_residency_after_reclamation(
            retained_payload_before_reclamation.saturating_sub(retained_payload_after_reclamation),
        );
        let sparse_experts_are_paged = model.sparse_experts_are_paged();
        let should_retry_same_prefill_chunck = ForwardRecoveryPolicy::retry_is_authorized(
            has_already_retried_after_reclamation,
            retained_payload_before_reclamation,
            retained_payload_after_reclamation,
            expert_reclamation_target_bytes,
            sparse_experts_are_paged,
        );
        tracing::warn!(
            request_id = request_id.value(),
            attempted_prefill_chunck_token_count,
            reason = failure_reason,
            fixed_forward_workspace_bytes,
            expert_reclamation_target_bytes,
            retained_expert_payload_bytes,
            retained_payload_after_reclamation,
            active_memory_bytes_before_reclamation =
                memory_snapshot_before_reclamation.active_memory_bytes(),
            active_memory_bytes_after_reclamation,
            did_demote_complete_resident_owner,
            should_retry_same_prefill_chunck,
            "graphics-processor memory exhaustion; released resident or paged experts for the fixed chunk"
        );
        Ok(should_retry_same_prefill_chunck)
    }
}

fn clear_allocator_cache_after_active_memory_limit(
    engine_state: &Qwen3_5EngineState,
    active_request: &mut Qwen3_5EngineRequest,
) -> Result<(), InferenceEngineError> {
    let model = engine_state
        .model
        .as_ref()
        .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
    // Synchronize before clearing reusable allocator storage. If synchronization
    // itself reports recoverable OOM, command completion has established the
    // failure boundary and cache cleanup remains the required next action.
    active_request
        .performance_attribution
        .measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| match model.runtime().synchronize_gpu_stream() {
                Ok(()) => model.runtime().clear_allocator_cache(),
                Err(mlx_runtime_error)
                    if mlx_runtime_error.is_recoverable_graphics_processor_out_of_memory() =>
                {
                    model.runtime().clear_allocator_cache()
                }
                Err(mlx_runtime_error) => Err(mlx_runtime_error),
            },
        )
        .map_err(qwen3_5_runtime_error)
}

fn clear_allocator_cache_without_stream_sync(
    engine_state: &Qwen3_5EngineState,
    active_request: &mut Qwen3_5EngineRequest,
) -> Result<(), InferenceEngineError> {
    let model = engine_state
        .model
        .as_ref()
        .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
    active_request
        .performance_attribution
        .measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| model.runtime().clear_allocator_cache(),
        )
        .map_err(qwen3_5_runtime_error)
}

fn demote_complete_resident_owner_for_prefill_recovery(
    engine_state: &mut Qwen3_5EngineState,
    active_request: &mut Qwen3_5EngineRequest,
) -> Result<bool, InferenceEngineError> {
    let complete_experts_are_resident = engine_state
        .model
        .as_ref()
        .is_some_and(|model| model.resident_expert_weights.is_some());
    if !complete_experts_are_resident {
        return Ok(false);
    }
    engine_state
        .model
        .as_mut()
        .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
        .demote_resident_experts_to_paging(
            Qwen3_5ExpertResidencyTransitionReason::RequestPressure,
            &mut active_request.performance_attribution,
        )
        .map_err(InferenceEngineError::from)?;
    Ok(true)
}

fn record_rejection_and_restore_checkpoint(
    active_request: &mut Qwen3_5EngineRequest,
    prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
) -> Result<(), InferenceEngineError> {
    active_request
        .performance_attribution
        .record_counter(PerformanceCounter::PrefillCapacityRejectionCount, 1);
    // Restore before cleanup or eviction so retry begins from the same decoder,
    // MTP, cursor, position, and visual-consumption frontier.
    active_request
        .restore_prefill_request_checkpoint(prefill_request_checkpoint)
        .map_err(qwen3_5_runtime_error)
}

fn retained_expert_payload_bytes(model: &crate::qwen3_5::model::Qwen3_5Model) -> usize {
    usize::try_from(
        model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count,
    )
    .unwrap_or(usize::MAX)
}

fn reclaim_and_sample_active_memory(
    model: &crate::qwen3_5::model::Qwen3_5Model,
    expert_reclamation_target_bytes: usize,
    active_memory_bytes_before_reclamation: usize,
) -> Result<usize, InferenceEngineError> {
    let memory_snapshot_after_reclamation = if expert_reclamation_target_bytes == 0 {
        None
    } else {
        reclaim_retained_experts_for_request_memory_pressure(
            model,
            expert_reclamation_target_bytes,
        )?
    };
    // MLX active bytes can lag cache ownership until lazy page arrays drop. Retry
    // authorization therefore uses retained-payload accounting; this is telemetry.
    Ok(memory_snapshot_after_reclamation
        .as_ref()
        .map_or(active_memory_bytes_before_reclamation, |memory_snapshot| {
            memory_snapshot.active_memory_bytes()
        }))
}
