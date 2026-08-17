//! Laguna model seams that enact centralized admission, reclamation, and demotion decisions.

use astronomical_ipc_protocol::ExpertMemoryMode;
use astronomical_runtime_integration::MlxRuntime;

use crate::expert_paging::{ExpertResidencyPhase, RetainedExpertReclamation};
use crate::laguna::normalization::LagunaFeedForwardDescriptor;
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};
use crate::{AllocationAdmissionDecision, MlxAllocationBudget, MlxAllocationBudgetError};

use super::error::LagunaExecutionError;
use super::model::LagunaModel;

impl LagunaModel {
    #[must_use]
    pub fn retained_expert_ceiling_bytes(&self) -> u64 {
        self.residency.retained_expert_ceiling_bytes()
    }

    pub(in crate::laguna) fn reclaim_retained_experts_for_request_pressure(
        &self,
        required_reclamation_bytes: u64,
    ) -> RetainedExpertReclamation {
        self.residency
            .reclaim_for_request_pressure(required_reclamation_bytes)
    }

    pub(in crate::laguna) fn resume_expert_retention_after_request_pressure(&self) {
        self.residency.resume_after_request_pressure();
    }

    pub(in crate::laguna) fn prepare_generation_expert_residency(&self) {
        self.residency
            .refresh_explicit_phase_plan(ExpertResidencyPhase::GenerationPreparation);
    }

    #[must_use]
    pub(in crate::laguna) fn maximum_expert_page_bytes(&self) -> u64 {
        self.expert_allocation_budget
            .as_ref()
            .map_or(0, MlxAllocationBudget::maximum_expert_page_bytes)
    }

    pub(in crate::laguna) fn update_expert_allocation_ceiling(
        &mut self,
        active_memory_ceiling_bytes: u64,
    ) {
        if let Some(expert_allocation_budget) = self.expert_allocation_budget.as_mut() {
            expert_allocation_budget
                .update_active_memory_ceiling_bytes(active_memory_ceiling_bytes);
        }
    }

    pub(in crate::laguna) fn update_expert_allocation_transient_high_water(
        &self,
        observed_transient_high_water_bytes: u64,
    ) {
        if let Some(expert_allocation_budget) = self.expert_allocation_budget.as_ref() {
            expert_allocation_budget
                .update_observed_transient_high_water_bytes(observed_transient_high_water_bytes);
        }
    }

    /// Admits one known page payload before any bounded read creates MLX arrays.
    pub(super) fn admit_expert_page_allocation(
        &self,
        runtime: &MlxRuntime,
        pending_allocation_bytes: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), LagunaExecutionError> {
        let Some(expert_allocation_budget) = self.expert_allocation_budget.as_ref() else {
            return Ok(());
        };
        let initial_observation = performance_attribution
            .measure_operation(PerformanceOperation::MemoryAdmissionSnapshot, |_| {
                expert_allocation_budget.observe(runtime, pending_allocation_bytes)
            })
            .map_err(|allocation_error| {
                laguna_allocation_budget_error(allocation_error, pending_allocation_bytes)
            })?;
        match initial_observation.decide() {
            AllocationAdmissionDecision::Admit => Ok(()),
            AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit => {
                performance_attribution
                    .measure_operation(PerformanceOperation::MlxAllocatorCacheCleanup, |_| {
                        runtime.synchronize_gpu_stream_and_clear_allocator_cache()
                    })?;
                let post_cleanup_observation = performance_attribution
                    .measure_operation(PerformanceOperation::MemoryAdmissionSnapshot, |_| {
                        expert_allocation_budget.observe(runtime, pending_allocation_bytes)
                    })
                    .map_err(|allocation_error| {
                        laguna_allocation_budget_error(allocation_error, pending_allocation_bytes)
                    })?;
                match post_cleanup_observation.decide() {
                    AllocationAdmissionDecision::Admit => Ok(()),
                    AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit => {
                        Err(LagunaExecutionError::ExpertAllocationRejected {
                            pending_allocation_bytes,
                            shortfall_bytes: post_cleanup_observation
                                .active_memory_bytes
                                .saturating_add(pending_allocation_bytes)
                                .saturating_sub(
                                    expert_allocation_budget.active_memory_ceiling_bytes(),
                                ),
                        })
                    }
                    AllocationAdmissionDecision::Reject {
                        shortfall_bytes, ..
                    } => Err(LagunaExecutionError::ExpertAllocationRejected {
                        pending_allocation_bytes,
                        shortfall_bytes,
                    }),
                }
            }
            AllocationAdmissionDecision::Reject {
                shortfall_bytes, ..
            } => Err(LagunaExecutionError::ExpertAllocationRejected {
                pending_allocation_bytes,
                shortfall_bytes,
            }),
        }
    }

    /// Synchronization precedes owner removal so no pending graph observes released tensors.
    pub fn demote_native_routed_experts(
        &mut self,
        runtime: &MlxRuntime,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<bool, LagunaExecutionError> {
        if self.expert_memory_mode() != ExpertMemoryMode::Resident
            || self.residency.paging_plan().is_none()
        {
            return Ok(false);
        }
        performance_attribution.measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| {
                runtime.synchronize_gpu_stream()?;
                let released_payload_bytes = self.weights.release_routed_experts(&self.contract);
                runtime.clear_allocator_cache()?;
                tracing::info!(
                    released_payload_bytes,
                    "Laguna demoted native routed experts to bounded paging"
                );
                Ok(released_payload_bytes > 0)
            },
        )
    }

    pub(in crate::laguna) fn native_routed_experts_are_resident(&self) -> bool {
        let mut sparse_layer_count = 0_usize;
        let mut native_resident_layer_count = 0_usize;
        for layer_descriptor in self.contract().layers() {
            if !matches!(
                layer_descriptor.feed_forward(),
                LagunaFeedForwardDescriptor::Moe(_)
            ) {
                continue;
            }
            sparse_layer_count = sparse_layer_count.saturating_add(1);
            if self
                .weights
                .has_routed_experts(layer_descriptor.layer_index())
            {
                native_resident_layer_count = native_resident_layer_count.saturating_add(1);
            }
        }
        sparse_layer_count > 0 && native_resident_layer_count == sparse_layer_count
    }
}

fn laguna_allocation_budget_error(
    allocation_error: MlxAllocationBudgetError,
    pending_allocation_bytes: u64,
) -> LagunaExecutionError {
    match allocation_error {
        MlxAllocationBudgetError::MlxRuntime(runtime_error) => {
            LagunaExecutionError::Runtime(runtime_error)
        }
        MlxAllocationBudgetError::Rejected {
            shortfall_bytes, ..
        } => LagunaExecutionError::ExpertAllocationRejected {
            pending_allocation_bytes,
            shortfall_bytes,
        },
    }
}
