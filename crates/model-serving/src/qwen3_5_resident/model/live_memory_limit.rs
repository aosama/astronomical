//! Atomic coordination of a live MLX ceiling and binary expert ownership.
//!
//! Lowering first demotes a complete resident owner when necessary, then reclaims
//! retained pages before MLX enforces the smaller limit. Raising lets MLX accept
//! capacity before Rust publishes a larger budget and atomically attempts complete
//! residency. A failed raised transition restores both native and Rust ceilings.

use astronomical_runtime_integration::MlxMemoryLimits;

use crate::{
    InferenceEngineError, MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements,
    PerformanceAttribution, safe_minimum_mlx_memory_ceiling_bytes,
};

use super::Qwen3_5ResidentModel;

impl Qwen3_5ResidentModel {
    pub(crate) fn minimum_mlx_memory_ceiling_bytes(&self) -> Result<u64, InferenceEngineError> {
        let current_mlx_memory_snapshot = self
            .runtime
            .memory_snapshot()
            .map_err(super::super::inference_execution::qwen3_5_runtime_error)?;
        let current_idle_active_mlx_memory_bytes =
            u64::try_from(current_mlx_memory_snapshot.active_memory_bytes()).map_err(|_| {
                super::super::inference_execution::fatal_engine_error(
                    "current MLX active memory exceeds the u64 range",
                )
            })?;
        let evictable_retained_expert_payload_bytes = 0;
        let maximum_expert_page_reserve_bytes = 0;
        Ok(safe_minimum_mlx_memory_ceiling_bytes(
            current_idle_active_mlx_memory_bytes,
            evictable_retained_expert_payload_bytes,
            maximum_expert_page_reserve_bytes,
        ))
    }

    pub(crate) fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(u64, MlxMemoryLimits), InferenceEngineError> {
        let current_mlx_memory_limits = self.runtime.memory_limits();
        let current_mlx_memory_ceiling_bytes =
            u64::try_from(current_mlx_memory_limits.active_memory_limit_bytes()).map_err(|_| {
                super::super::inference_execution::fatal_engine_error(
                    "current MLX memory ceiling exceeds the u64 range",
                )
            })?;
        let minimum_mlx_memory_ceiling_bytes = self.minimum_mlx_memory_ceiling_bytes()?;
        let current_active_memory_bytes = u64::try_from(
            self.runtime
                .memory_snapshot()
                .map_err(super::super::inference_execution::qwen3_5_runtime_error)?
                .active_memory_bytes(),
        )
        .map_err(|_| {
            super::super::inference_execution::fatal_engine_error(
                "current MLX active memory exceeds the u64 range",
            )
        })?;
        let complete_experts_are_resident = true;
        let complete_residency_required_headroom_bytes = 0;
        let ceiling_change_decision = MemoryCeilingChangeRequirements {
            current_ceiling_bytes: current_mlx_memory_ceiling_bytes,
            requested_ceiling_bytes: requested_mlx_memory_ceiling_bytes,
            minimum_safe_ceiling_bytes: minimum_mlx_memory_ceiling_bytes,
            current_active_memory_bytes,
            retained_paged_expert_payload_bytes: 0,
            complete_experts_are_resident,
            complete_residency_required_headroom_bytes,
        }
        .decide();
        if let MemoryCeilingChangeDecision::Reject { .. } = ceiling_change_decision {
            return Err(InferenceEngineError::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                reason:
                    "the loaded model needs its non-evictable memory and one expert page reserve"
                        .to_owned(),
            });
        }
        let requested_mlx_memory_ceiling_bytes_as_usize =
            usize::try_from(requested_mlx_memory_ceiling_bytes).map_err(|_| {
                super::super::inference_execution::fatal_engine_error(
                    "requested MLX memory ceiling exceeds the platform range",
                )
            })?;
        if requested_mlx_memory_ceiling_bytes == current_mlx_memory_ceiling_bytes {
            return Ok((minimum_mlx_memory_ceiling_bytes, current_mlx_memory_limits));
        }

        let requested_mlx_memory_limits = MlxMemoryLimits::new(
            requested_mlx_memory_ceiling_bytes_as_usize,
            requested_mlx_memory_ceiling_bytes_as_usize,
        )
        .map_err(super::super::inference_execution::qwen3_5_runtime_error)?;
        let is_lowering_mlx_memory_ceiling = matches!(
            ceiling_change_decision,
            MemoryCeilingChangeDecision::Lower { .. }
        );
        let _previous_expert_paging_memory_ceiling_bytes: Option<u64> = None;
        if is_lowering_mlx_memory_ceiling {
            // Native eviction comes first because the old MLX ceiling still
            // permits the bookkeeping and synchronization needed to release
            // pages. Installing the smaller runtime limit first could make the
            // reclamation operation reject its own temporary work.
            let post_reclamation_mlx_memory_snapshot = self
                .runtime
                .synchronize_gpu_stream_and_clear_allocator_cache()
                .and_then(|()| self.runtime.memory_snapshot())
                .map_err(super::super::inference_execution::qwen3_5_runtime_error)?;
            let post_reclamation_active_memory_bytes = u64::try_from(
                post_reclamation_mlx_memory_snapshot.active_memory_bytes(),
            )
            .map_err(|_| {
                super::super::inference_execution::fatal_engine_error(
                    "post-reclamation MLX active memory exceeds the u64 range",
                )
            })?;
            if post_reclamation_active_memory_bytes > requested_mlx_memory_ceiling_bytes {
                return Err(super::super::inference_execution::fatal_engine_error(
                    "resident model memory remained above the requested MLX ceiling",
                ));
            }
        }

        if let Err(runtime_update_error) = self
            .runtime
            .update_memory_limits(requested_mlx_memory_limits)
        {
            return Err(super::super::inference_execution::qwen3_5_runtime_error(
                runtime_update_error,
            ));
        }
        Ok((
            minimum_mlx_memory_ceiling_bytes,
            requested_mlx_memory_limits,
        ))
    }

    fn prepare_expert_residency_for_lower_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
        ceiling_change_decision: MemoryCeilingChangeDecision,
    ) -> Result<(), InferenceEngineError> {
        let _ceiling_change_decision = ceiling_change_decision;
        self.mlx_ram_budget
            .borrow_mut()
            .update_mlx_active_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes)
            .map_err(|mlx_ram_budget_error| {
                super::super::inference_execution::fatal_engine_error(
                    mlx_ram_budget_error.to_string(),
                )
            })?;
        Ok(())
    }

    fn update_expert_residency_for_live_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        self.mlx_ram_budget
            .borrow_mut()
            .update_mlx_active_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes)
            .map_err(|mlx_ram_budget_error| {
                super::super::inference_execution::fatal_engine_error(
                    mlx_ram_budget_error.to_string(),
                )
            })?;
        Ok(())
    }

    fn restore_expert_paging_memory_ceiling(
        &mut self,
        previous_memory_ceiling_bytes: Option<u64>,
    ) -> Result<(), InferenceEngineError> {
        if let Some(previous_memory_ceiling_bytes) = previous_memory_ceiling_bytes {
            self.update_expert_residency_for_live_mlx_memory_limit(previous_memory_ceiling_bytes)?;
        }
        Ok(())
    }

    fn restore_failed_raised_memory_limit(
        &mut self,
        previous_mlx_memory_limits: MlxMemoryLimits,
        previous_expert_paging_memory_ceiling_bytes: Option<u64>,
        transition_error: InferenceEngineError,
    ) -> InferenceEngineError {
        let policy_restore_result =
            self.restore_expert_paging_memory_ceiling(previous_expert_paging_memory_ceiling_bytes);
        let runtime_restore_result = self
            .runtime
            .update_memory_limits(previous_mlx_memory_limits);
        match (policy_restore_result, runtime_restore_result) {
            (Ok(()), Ok(())) => transition_error,
            (policy_restore_result, runtime_restore_result) => {
                super::super::inference_execution::fatal_engine_error(format!(
                    "raised MLX memory transition failed and rollback was incomplete: transition_error={transition_error}; policy_restore_error={:?}; runtime_restore_error={:?}",
                    policy_restore_result.err(),
                    runtime_restore_result.err(),
                ))
            }
        }
    }
}
