//! Atomic coordination of a live MLX ceiling and binary expert ownership.
//!
//! Lowering first demotes a complete resident owner when necessary, then reduces
//! complete resident ownership before MLX enforces the smaller limit. Raising reverses
//! the order: MLX accepts capacity, Rust retention expands, and complete residency
//! is attempted. A failed transition restores the pager's prior ceiling.

use astronomical_runtime_integration::MlxMemoryLimits;

use crate::qwen3_5_moe::Qwen3_5ExpertResidencyTransitionReason;
use crate::{
    InferenceEngineError, MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements,
    PerformanceAttribution, safe_minimum_mlx_memory_ceiling_bytes,
};

use super::Qwen3_5Model;

impl Qwen3_5Model {
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
        let expert_weight_memory_cache_statistics = self.expert_weight_memory_cache_statistics();
        // Sparse expert payload is elastic in either mode: the complete owner can
        // demote, while paged mode can evict retained slots. Dense weights are not.
        let evictable_retained_expert_payload_bytes = if self.expert_pager.is_some() {
            expert_weight_memory_cache_statistics.resident_payload_byte_count
        } else {
            0
        };
        let maximum_expert_page_reserve_bytes = self
            .expert_pager
            .as_ref()
            .map_or(0, |expert_pager| expert_pager.maximum_expert_page_bytes());
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
        let expert_statistics = self.expert_weight_memory_cache_statistics();
        let ceiling_change_decision = MemoryCeilingChangeRequirements {
            current_ceiling_bytes: current_mlx_memory_ceiling_bytes,
            requested_ceiling_bytes: requested_mlx_memory_ceiling_bytes,
            minimum_safe_ceiling_bytes: minimum_mlx_memory_ceiling_bytes,
            current_active_memory_bytes,
            retained_paged_expert_payload_bytes: expert_statistics.resident_payload_byte_count,
            complete_experts_are_resident: self.resident_expert_weights.is_some(),
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
        let previous_expert_paging_memory_ceiling_bytes = self
            .expert_pager
            .as_ref()
            .map(|expert_pager| expert_pager.configured_mlx_memory_ceiling_bytes());
        if is_lowering_mlx_memory_ceiling {
            // Native eviction comes first because the old MLX ceiling still
            // permits the bookkeeping and synchronization needed to release
            // pages. Installing the smaller runtime limit first could make the
            // reclamation operation reject its own temporary work.
            self.prepare_expert_residency_for_lower_mlx_memory_limit(
                requested_mlx_memory_ceiling_bytes,
                ceiling_change_decision,
            )?;
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
                self.restore_expert_paging_memory_ceiling(
                    previous_expert_paging_memory_ceiling_bytes,
                )?;
                return Err(super::super::inference_execution::fatal_engine_error(
                    "retained model memory remained above the requested MLX ceiling after expert reclamation",
                ));
            }
        }

        if let Err(runtime_update_error) = self
            .runtime
            .update_memory_limits(requested_mlx_memory_limits)
        {
            self.restore_expert_paging_memory_ceiling(previous_expert_paging_memory_ceiling_bytes)?;
            return Err(super::super::inference_execution::qwen3_5_runtime_error(
                runtime_update_error,
            ));
        }
        if !is_lowering_mlx_memory_ceiling {
            // Once MLX accepts a larger ceiling, let Rust policy derive a
            // larger retention target from fresh counters. This never prewarms
            // pages; later router evidence repopulates capacity naturally.
            self.update_expert_residency_for_live_mlx_memory_limit(
                requested_mlx_memory_ceiling_bytes,
            )?;
            let mut disabled_performance_attribution = PerformanceAttribution::disabled();
            self.try_promote_experts_to_resident(
                Qwen3_5ExpertResidencyTransitionReason::CeilingRaise,
                &mut disabled_performance_attribution,
            )
            .map_err(InferenceEngineError::from)?;
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
        let (must_demote_complete_residency, retained_paged_expert_reclamation_bytes) =
            match ceiling_change_decision {
                MemoryCeilingChangeDecision::Lower {
                    must_demote_complete_residency,
                    retained_paged_expert_reclamation_bytes,
                } => (
                    must_demote_complete_residency,
                    retained_paged_expert_reclamation_bytes,
                ),
                _ => (false, 0),
            };
        if must_demote_complete_residency {
            let mut disabled_performance_attribution = PerformanceAttribution::disabled();
            self.demote_resident_experts_to_paging(
                Qwen3_5ExpertResidencyTransitionReason::CeilingLower,
                &mut disabled_performance_attribution,
            )
            .map_err(InferenceEngineError::from)?;
        }
        if self.resident_expert_weights.is_none()
            && let Some(retained_expert_layers) = self.retained_expert_layers.as_ref()
        {
            retained_expert_layers
                .borrow_mut()
                .limit_for_request_pressure(retained_paged_expert_reclamation_bytes);
        }
        self.mlx_ram_budget
            .borrow_mut()
            .update_mlx_active_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes)
            .map_err(|mlx_ram_budget_error| {
                super::super::inference_execution::fatal_engine_error(
                    mlx_ram_budget_error.to_string(),
                )
            })?;
        let Some(expert_pager) = self.expert_pager.as_mut() else {
            return Ok(());
        };
        expert_pager.update_configured_mlx_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes);
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
        let Some(expert_pager) = self.expert_pager.as_mut() else {
            return Ok(());
        };
        expert_pager.update_configured_mlx_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes);
        Ok(())
    }

    fn restore_expert_paging_memory_ceiling(
        &mut self,
        previous_memory_ceiling_bytes: Option<u64>,
    ) -> Result<(), InferenceEngineError> {
        if let Some(previous_memory_ceiling_bytes) = previous_memory_ceiling_bytes {
            self.update_expert_residency_for_live_mlx_memory_limit(previous_memory_ceiling_bytes)?;
            let mut disabled_performance_attribution = PerformanceAttribution::disabled();
            self.try_promote_experts_to_resident(
                Qwen3_5ExpertResidencyTransitionReason::CeilingRaise,
                &mut disabled_performance_attribution,
            )
            .map_err(InferenceEngineError::from)?;
        }
        Ok(())
    }
}
