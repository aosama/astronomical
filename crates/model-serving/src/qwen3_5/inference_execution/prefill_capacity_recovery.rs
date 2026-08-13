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
//! 4. Reclaim only elastic retained expert bytes.
//! 5. Authorize one retry only when cache ownership proves the target was released.
//!
//! The two public helpers differ because typed active-limit rejection exposes the
//! failed allocation size and may still synchronize cleanly, whereas GPU OOM can
//! poison the synchronization event chain and exposes no allocation size.

use astronomical_ipc_protocol::RequestId;

use crate::qwen3_5_moe::reclaim_retained_experts_for_request_memory_pressure;
use crate::{
    ForwardRecoveryPolicy, InferenceEngineError, PerformanceCounter, PerformanceOperation,
};

use super::engine_request::{Qwen3_5EngineRequest, Qwen3_5PrefillRequestCheckpoint};
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

impl Qwen3_5EngineState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn recover_fixed_prefill_chunck_after_active_memory_limit(
        &self,
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
        let model = self
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
            .map_err(qwen3_5_runtime_error)?;

        let memory_snapshot_before_reclamation = model
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        let retained_expert_payload_bytes = retained_expert_payload_bytes(model);
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
        let active_memory_bytes_after_reclamation = reclaim_and_sample_active_memory(
            model,
            expert_reclamation_target_bytes,
            memory_snapshot_before_reclamation.active_memory_bytes(),
        )?;
        let retained_payload_after_reclamation = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;

        if active_request.should_use_speculative_prefill {
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
                retained_payload_before_reclamation
                    .saturating_sub(retained_payload_after_reclamation),
            );
        }
        let should_retry_same_prefill_chunck = ForwardRecoveryPolicy::retry_is_authorized(
            has_already_retried_after_reclamation,
            retained_payload_before_reclamation,
            retained_payload_after_reclamation,
            expert_reclamation_target_bytes,
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
            should_retry_same_prefill_chunck,
            "native MLX prefill allocation reached the active-memory ceiling; reclaimed elastic experts for the fixed chunk"
        );
        Ok(should_retry_same_prefill_chunck)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn recover_fixed_prefill_chunck_after_graphics_processor_exhaustion(
        &self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        attempted_prefill_chunck_token_count: usize,
        failure_reason: &str,
        prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
        has_already_retried_after_reclamation: bool,
    ) -> Result<bool, InferenceEngineError> {
        record_rejection_and_restore_checkpoint(active_request, prefill_request_checkpoint)?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;

        // The failed synchronization already waited for command-buffer completion.
        // Synchronizing its poisoned event chain again would reproduce the error.
        active_request
            .performance_attribution
            .measure_operation(
                PerformanceOperation::MlxAllocatorCacheCleanup,
                |_performance_attribution| model.runtime().clear_allocator_cache(),
            )
            .map_err(qwen3_5_runtime_error)?;

        let memory_snapshot_before_reclamation = model
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        let retained_expert_payload_bytes = retained_expert_payload_bytes(model);
        let retained_payload_before_reclamation =
            u64::try_from(retained_expert_payload_bytes).unwrap_or(u64::MAX);
        // GPU OOM does not expose one failed allocation size. Use learned
        // transient high-water as the model-local fixed workspace estimate, with
        // one byte ensuring the formula represents actual required capacity.
        let fixed_forward_workspace_bytes = self
            .adaptive_ram_growth_guard
            .admission_transient_high_water_bytes()
            .max(1);
        let expert_reclamation_target_bytes = ForwardRecoveryPolicy::required_reclamation_bytes(
            memory_snapshot_before_reclamation.active_memory_bytes(),
            retained_expert_payload_bytes,
            model.runtime().memory_limits().active_memory_limit_bytes(),
            fixed_forward_workspace_bytes,
        );
        let active_memory_bytes_after_reclamation = reclaim_and_sample_active_memory(
            model,
            expert_reclamation_target_bytes,
            memory_snapshot_before_reclamation.active_memory_bytes(),
        )?;
        let retained_payload_after_reclamation = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let should_retry_same_prefill_chunck = ForwardRecoveryPolicy::retry_is_authorized(
            has_already_retried_after_reclamation,
            retained_payload_before_reclamation,
            retained_payload_after_reclamation,
            expert_reclamation_target_bytes,
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
            should_retry_same_prefill_chunck,
            "graphics-processor memory exhaustion; reclaimed elastic experts for the fixed chunk"
        );
        Ok(should_retry_same_prefill_chunck)
    }
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
