//! Whole-model transitions between contiguous residency and native paging.
//!
//! Promotion follows prepare -> build candidate -> publish. Native retention is
//! frozen before its pages are reclaimed, and the model remains observably paged
//! until a complete candidate exists. Demotion follows synchronize -> unpublish
//! -> drop -> clear allocator -> resume paging. These orders prevent lazy MLX
//! work from retaining released arrays and make every failure state usable.

use astronomical_runtime_integration::MlxRuntimeError;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::Qwen3_5ResidentExpertWeights;
use crate::{PerformanceAttribution, PerformanceOperation};

/// A safe owner-thread boundary that requested an expert residency transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen3_5ExpertResidencyTransitionReason {
    Startup,
    RequestAdmission,
    RequestPressure,
    RequestFinalization,
    CeilingRaise,
    CeilingLower,
    SpeculativePrefillDraftLoading,
}

/// Nonfatal outcomes from an optional complete-model promotion attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen3_5ExpertResidencyPromotionOutcome {
    AlreadyResident,
    Promoted,
    DoesNotFit,
    RecoverableCapacityRejection,
}

impl Qwen3_5Model {
    pub(crate) fn try_promote_experts_to_resident(
        &mut self,
        transition_reason: Qwen3_5ExpertResidencyTransitionReason,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5ExpertResidencyPromotionOutcome, Qwen3_5ExecutionError> {
        performance_attribution.measure_operation(
            PerformanceOperation::ResidentWeightMaterializationSynchronizationWait,
            |performance_attribution| {
                self.try_promote_experts_to_resident_without_attribution(
                    transition_reason,
                    performance_attribution.positional_file_read_metrics(),
                )
            },
        )
    }

    fn try_promote_experts_to_resident_without_attribution(
        &mut self,
        transition_reason: Qwen3_5ExpertResidencyTransitionReason,
        positional_file_read_metrics: Option<
            std::sync::Arc<astronomical_runtime_integration::PositionalFileReadMetrics>,
        >,
    ) -> Result<Qwen3_5ExpertResidencyPromotionOutcome, Qwen3_5ExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(Qwen3_5ExpertResidencyPromotionOutcome::AlreadyResident);
        };
        if self.resident_expert_weights.is_some() {
            return Ok(Qwen3_5ExpertResidencyPromotionOutcome::AlreadyResident);
        }
        let complete_expert_payload_bytes = expert_pager.complete_expert_payload_byte_count()?;
        tracing::info!(
            ?transition_reason,
            complete_expert_payload_bytes,
            "started complete-model expert residency admission"
        );

        // No paged snapshot may still be executing when its cache is emptied.
        self.runtime.synchronize_gpu_stream()?;
        expert_pager.freeze_native_expert_retention_growth();
        // Everything inside this closure is speculative. `self` is not changed
        // until the complete candidate reaches the publication point below.
        let candidate_resident_expert_weights_result = (|| {
            let retained_native_expert_payload_bytes = expert_pager
                .native_expert_cache_statistics()
                .resident_payload_byte_count();
            if retained_native_expert_payload_bytes > 0 {
                expert_pager
                    .reclaim_native_expert_payload_bytes(retained_native_expert_payload_bytes)?;
            }
            self.runtime
                .synchronize_gpu_stream_and_clear_allocator_cache()?;
            if expert_pager
                .native_expert_cache_statistics()
                .resident_payload_byte_count()
                != 0
            {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "native expert pages remained retained before resident promotion",
                });
            }

            // Admission uses fresh active bytes after native pages and reusable
            // allocator buffers are gone. The complete payload is exact artifact
            // geometry, so no laptop-specific headroom constant is required.
            let idle_memory_snapshot = self.runtime.memory_snapshot()?;
            let idle_active_memory_bytes =
                u64::try_from(idle_memory_snapshot.active_memory_bytes()).map_err(|_| {
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "idle MLX active memory exceeds the u64 range",
                    }
                })?;
            let stable_memory_ceiling_bytes = u64::try_from(
                self.runtime.memory_limits().active_memory_limit_bytes(),
            )
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "MLX active memory ceiling exceeds the u64 range",
            })?;
            let projected_resident_active_memory_bytes = idle_active_memory_bytes
                .checked_add(complete_expert_payload_bytes)
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "projected resident MLX active memory overflowed",
                })?;
            if projected_resident_active_memory_bytes > stable_memory_ceiling_bytes {
                tracing::info!(
                    ?transition_reason,
                    idle_active_memory_bytes,
                    complete_expert_payload_bytes,
                    projected_resident_active_memory_bytes,
                    stable_memory_ceiling_bytes,
                    outcome = "does_not_fit",
                    "completed complete-model expert residency admission"
                );
                return Ok(None);
            }

            Qwen3_5ResidentExpertWeights::load(self, positional_file_read_metrics).map(Some)
        })();
        let candidate_resident_expert_weights = match candidate_resident_expert_weights_result {
            Ok(Some(candidate_resident_expert_weights)) => candidate_resident_expert_weights,
            Ok(None) => {
                expert_pager.resume_native_expert_retention_growth();
                return Ok(Qwen3_5ExpertResidencyPromotionOutcome::DoesNotFit);
            }
            Err(resident_loading_error) => {
                let is_recoverable_capacity_rejection =
                    resident_loading_error_is_recoverable_capacity(&resident_loading_error);
                let cleanup_result = self
                    .runtime
                    .synchronize_gpu_stream_and_clear_allocator_cache();
                expert_pager.resume_native_expert_retention_growth();
                if let Err(cleanup_error) = cleanup_result {
                    return Err(cleanup_error.into());
                }
                if is_recoverable_capacity_rejection {
                    tracing::info!(
                        ?transition_reason,
                        complete_expert_payload_bytes,
                        outcome = "recoverable_capacity_rejection",
                        error = %resident_loading_error,
                        "completed complete-model expert residency admission"
                    );
                    return Ok(
                        Qwen3_5ExpertResidencyPromotionOutcome::RecoverableCapacityRejection,
                    );
                }
                return Err(resident_loading_error);
            }
        };
        // This assignment is the only Paged -> Resident publication point.
        self.resident_expert_weights = Some(candidate_resident_expert_weights);
        tracing::info!(
            ?transition_reason,
            complete_expert_payload_bytes,
            resident_layer_count = self
                .resident_expert_weights
                .as_ref()
                .map_or(0, Qwen3_5ResidentExpertWeights::layer_count),
            outcome = "promoted",
            "completed complete-model expert residency admission"
        );
        Ok(Qwen3_5ExpertResidencyPromotionOutcome::Promoted)
    }

    pub(crate) fn demote_resident_experts_to_paging(
        &mut self,
        transition_reason: Qwen3_5ExpertResidencyTransitionReason,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<bool, Qwen3_5ExecutionError> {
        if self.resident_expert_weights.is_none() {
            return Ok(false);
        }
        performance_attribution.measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| {
                // Synchronize before removing the owner because lazy arrays and
                // submitted kernels may still reference its backing allocations.
                self.runtime.synchronize_gpu_stream()?;
                let released_resident_expert_weights = self.resident_expert_weights.take();
                let released_resident_expert_payload_bytes = released_resident_expert_weights
                    .as_ref()
                    .map_or(0, Qwen3_5ResidentExpertWeights::payload_byte_count);
                drop(released_resident_expert_weights);
                // Dropped resident buffers become reusable allocator storage;
                // clearing it makes the newly paged mode's capacity observable.
                let allocator_cleanup_result = self.runtime.clear_allocator_cache();
                if let Some(expert_pager) = self.expert_pager.as_ref() {
                    expert_pager.resume_native_expert_retention_growth();
                }
                allocator_cleanup_result?;
                tracing::info!(
                    ?transition_reason,
                    released_resident_expert_payload_bytes,
                    "demoted complete resident experts to native demand paging"
                );
                Ok(true)
            },
        )
    }
}

fn resident_loading_error_is_recoverable_capacity(
    resident_loading_error: &Qwen3_5ExecutionError,
) -> bool {
    match resident_loading_error {
        Qwen3_5ExecutionError::Runtime(MlxRuntimeError::ActiveMemoryLimitExceeded { .. }) => true,
        Qwen3_5ExecutionError::Runtime(runtime_error) => {
            runtime_error.is_recoverable_graphics_processor_out_of_memory()
        }
        _ => false,
    }
}
