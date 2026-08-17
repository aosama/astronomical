//! Checkpoint-first, one-retry recovery for a fixed Laguna prompt chunk.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::laguna::{
    LagunaDecoderState, LagunaDecoderStateAllocationCheckpoint, LagunaExecutionError, LagunaModel,
};
use crate::{
    AdaptiveRamGrowthGuard, ForwardRecoveryDecision, ForwardRecoveryPolicy,
    ForwardRecoveryRequirements, InferenceEngineError, PerformanceAttribution, PerformanceCounter,
    PerformanceOperation,
};

/// Test-only failure state that compiles to a zero-sized no-op in production builds.
#[derive(Default)]
pub(super) struct LagunaPrefillFailureInjection {
    #[cfg(debug_assertions)]
    remaining_failure_count: u8,
}

impl LagunaPrefillFailureInjection {
    #[cfg(debug_assertions)]
    pub(super) fn arm_two_failures(&mut self) {
        self.remaining_failure_count = 2;
    }

    /// Keeping the branch behind the feature prevents acceptance machinery from entering serving.
    pub(super) fn take_next_failure(&mut self) -> bool {
        #[cfg(debug_assertions)]
        {
            if self.remaining_failure_count > 0 {
                self.remaining_failure_count = self.remaining_failure_count.saturating_sub(1);
                return true;
            }
        }
        false
    }
}

/// Restores mutable request state before reclaiming any expert owner.
#[allow(clippy::too_many_arguments)]
pub(super) fn recover_laguna_prefill_capacity(
    runtime: &MlxRuntime,
    model: &mut LagunaModel,
    adaptive_ram_growth_guard: &AdaptiveRamGrowthGuard,
    decoder_state: &mut LagunaDecoderState,
    allocation_checkpoint: LagunaDecoderStateAllocationCheckpoint,
    capacity_error: LagunaExecutionError,
    has_already_retried_after_reclamation: bool,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<bool, InferenceEngineError> {
    decoder_state
        .restore_allocation_checkpoint(allocation_checkpoint)
        .map_err(|restore_error| InferenceEngineError::Fatal {
            reason: format!("Laguna prefill checkpoint restoration failed: {restore_error}"),
        })?;
    performance_attribution.record_counter(PerformanceCounter::PrefillCapacityRejectionCount, 1);
    let failure_evidence =
        capacity_failure_evidence(runtime, &capacity_error, performance_attribution)?;
    clear_allocator_after_capacity_failure(
        runtime,
        failure_evidence.graphics_processor_memory_exhausted,
        performance_attribution,
    )?;
    let stable_snapshot = performance_attribution
        .measure_operation(PerformanceOperation::MemoryAdmissionSnapshot, |_| {
            runtime.memory_snapshot()
        })
        .map_err(|memory_error| InferenceEngineError::Fatal {
            reason: format!("Laguna recovery could not sample stable memory: {memory_error}"),
        })?;
    let retained_payload_before_reclamation = model
        .expert_weight_memory_cache_statistics()
        .resident_payload_byte_count;
    let fixed_forward_workspace_bytes = ForwardRecoveryPolicy::fixed_workspace_bytes(
        stable_snapshot.active_memory_bytes(),
        failure_evidence.active_memory_bytes_at_failure,
        failure_evidence.attempted_allocation_bytes,
        adaptive_ram_growth_guard.admission_transient_high_water_bytes(),
    );
    let required_reclamation_bytes = ForwardRecoveryPolicy::required_reclamation_bytes(
        stable_snapshot.active_memory_bytes(),
        usize::try_from(retained_payload_before_reclamation).unwrap_or(usize::MAX),
        failure_evidence.allowed_active_memory_bytes,
        fixed_forward_workspace_bytes,
    );
    if model.native_routed_experts_are_resident() {
        model
            .demote_native_routed_experts(runtime, performance_attribution)
            .map_err(|demotion_error| InferenceEngineError::Fatal {
                reason: format!("Laguna recovery expert demotion failed: {demotion_error}"),
            })?;
    }
    if required_reclamation_bytes > 0 {
        model.reclaim_retained_experts_for_request_pressure(
            u64::try_from(required_reclamation_bytes).unwrap_or(u64::MAX),
        );
    }
    runtime
        .clear_allocator_cache()
        .map_err(|cleanup_error| InferenceEngineError::Fatal {
            reason: format!("Laguna recovery allocator cleanup failed: {cleanup_error}"),
        })?;
    let retained_payload_after_reclamation = model
        .expert_weight_memory_cache_statistics()
        .resident_payload_byte_count;
    let recovery_decision = ForwardRecoveryRequirements {
        stable_active_memory_bytes: stable_snapshot.active_memory_bytes(),
        active_memory_bytes_at_failure: failure_evidence.active_memory_bytes_at_failure,
        attempted_allocation_bytes: failure_evidence.attempted_allocation_bytes,
        observed_transient_high_water_bytes: adaptive_ram_growth_guard
            .admission_transient_high_water_bytes(),
        retained_expert_payload_bytes_before_reclamation: usize::try_from(
            retained_payload_before_reclamation,
        )
        .unwrap_or(usize::MAX),
        retained_expert_payload_bytes_after_reclamation: usize::try_from(
            retained_payload_after_reclamation,
        )
        .unwrap_or(usize::MAX),
        active_memory_ceiling_bytes: failure_evidence.allowed_active_memory_bytes,
        has_already_retried_after_reclamation,
    }
    .decide();
    let should_retry = matches!(recovery_decision, ForwardRecoveryDecision::Retry { .. });
    tracing::warn!(
        fixed_forward_workspace_bytes,
        required_reclamation_bytes,
        retained_payload_before_reclamation,
        retained_payload_after_reclamation,
        ?recovery_decision,
        "Laguna applied centralized fixed-prefill recovery"
    );
    if should_retry {
        performance_attribution.record_counter(PerformanceCounter::PrefillCapacityRetryCount, 1);
    }
    Ok(should_retry)
}

struct CapacityFailureEvidence {
    active_memory_bytes_at_failure: usize,
    attempted_allocation_bytes: usize,
    allowed_active_memory_bytes: usize,
    graphics_processor_memory_exhausted: bool,
}

fn capacity_failure_evidence(
    runtime: &MlxRuntime,
    capacity_error: &LagunaExecutionError,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<CapacityFailureEvidence, InferenceEngineError> {
    match capacity_error {
        LagunaExecutionError::Runtime(MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
        }) => Ok(CapacityFailureEvidence {
            active_memory_bytes_at_failure: *active_memory_bytes,
            attempted_allocation_bytes: *attempted_allocation_bytes,
            allowed_active_memory_bytes: *allowed_active_memory_bytes,
            graphics_processor_memory_exhausted: false,
        }),
        LagunaExecutionError::Runtime(runtime_error)
            if runtime_error.is_recoverable_graphics_processor_out_of_memory() =>
        {
            let active_memory_bytes_at_failure = performance_attribution
                .measure_operation(PerformanceOperation::MemoryAdmissionSnapshot, |_| {
                    runtime.memory_snapshot()
                })
                .map_err(|memory_error| InferenceEngineError::Fatal {
                    reason: format!(
                        "Laguna GPU-memory recovery could not sample memory: {memory_error}"
                    ),
                })?
                .active_memory_bytes();
            Ok(CapacityFailureEvidence {
                active_memory_bytes_at_failure,
                attempted_allocation_bytes: 1,
                allowed_active_memory_bytes: runtime.memory_limits().active_memory_limit_bytes(),
                graphics_processor_memory_exhausted: true,
            })
        }
        LagunaExecutionError::ExpertAllocationRejected {
            pending_allocation_bytes,
            ..
        } => {
            let active_memory_bytes_at_failure = performance_attribution
                .measure_operation(PerformanceOperation::MemoryAdmissionSnapshot, |_| {
                    runtime.memory_snapshot()
                })
                .map_err(|memory_error| InferenceEngineError::Fatal {
                    reason: format!(
                        "Laguna page-admission recovery could not sample memory: {memory_error}"
                    ),
                })?
                .active_memory_bytes();
            Ok(CapacityFailureEvidence {
                active_memory_bytes_at_failure,
                attempted_allocation_bytes: usize::try_from(*pending_allocation_bytes)
                    .unwrap_or(usize::MAX),
                allowed_active_memory_bytes: runtime.memory_limits().active_memory_limit_bytes(),
                graphics_processor_memory_exhausted: false,
            })
        }
        _ => Err(InferenceEngineError::Fatal {
            reason: "Laguna attempted capacity recovery for a non-memory failure".to_owned(),
        }),
    }
}

fn clear_allocator_after_capacity_failure(
    runtime: &MlxRuntime,
    graphics_processor_memory_exhausted: bool,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    performance_attribution
        .measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| {
                if graphics_processor_memory_exhausted {
                    return runtime.clear_allocator_cache();
                }
                match runtime.synchronize_gpu_stream() {
                    Ok(()) => runtime.clear_allocator_cache(),
                    Err(runtime_error)
                        if runtime_error.is_recoverable_graphics_processor_out_of_memory() =>
                    {
                        runtime.clear_allocator_cache()
                    }
                    Err(runtime_error) => Err(runtime_error),
                }
            },
        )
        .map_err(|cleanup_error| InferenceEngineError::Fatal {
            reason: format!("Laguna capacity recovery cleanup failed: {cleanup_error}"),
        })
}
