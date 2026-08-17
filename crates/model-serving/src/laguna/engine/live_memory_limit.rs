//! Live Laguna MLX-ceiling validation and application.

use astronomical_ipc_protocol::ExpertMemoryMode;
use astronomical_runtime_integration::MlxMemoryLimits;

use crate::{
    InferenceEngineError, MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements,
    MlxMemoryLimitAdjustment, MlxRamBudgetPhase, PerformanceAttribution,
};

use super::execution::LagunaInferenceExecution;
use super::memory::laguna_ram_budget_snapshot;

impl LagunaInferenceExecution {
    pub(super) fn apply_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        let minimum_mlx_memory_ceiling_bytes = self.minimum_mlx_memory_ceiling_bytes;
        let runtime = self.runtime.as_ref().ok_or(InferenceEngineError::Fatal {
            reason: "the Laguna runtime is not loaded".to_owned(),
        })?;
        let current_memory_limits = runtime.memory_limits();
        let current_ceiling_bytes =
            u64::try_from(current_memory_limits.active_memory_limit_bytes()).unwrap_or(u64::MAX);
        let current_active_memory_bytes = u64::try_from(
            runtime
                .memory_snapshot()
                .map_err(|memory_error| InferenceEngineError::Fatal {
                    reason: format!(
                        "Laguna could not sample memory before a ceiling change: {memory_error}"
                    ),
                })?
                .active_memory_bytes(),
        )
        .unwrap_or(u64::MAX);
        let (expert_statistics, expert_memory_mode, complete_experts_are_resident) = {
            let model = self.model.as_ref().ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            })?;
            (
                model.expert_weight_memory_cache_statistics(),
                model.expert_memory_mode(),
                model.native_routed_experts_are_resident(),
            )
        };
        let ceiling_change_decision = MemoryCeilingChangeRequirements {
            current_ceiling_bytes,
            requested_ceiling_bytes: requested_mlx_memory_ceiling_bytes,
            minimum_safe_ceiling_bytes: minimum_mlx_memory_ceiling_bytes,
            current_active_memory_bytes,
            retained_paged_expert_payload_bytes: expert_statistics.resident_payload_byte_count,
            complete_experts_are_resident,
        }
        .decide();
        if matches!(
            ceiling_change_decision,
            MemoryCeilingChangeDecision::Reject { .. }
        ) {
            return Err(InferenceEngineError::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                reason: "the requested ceiling cannot preserve Laguna model core and mandatory runtime work"
                    .to_owned(),
            });
        }
        if ceiling_change_decision == MemoryCeilingChangeDecision::Unchanged {
            return Ok(MlxMemoryLimitAdjustment::new(
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                expert_memory_mode,
                self.collect_current_mlx_memory_telemetry(),
            ));
        }
        let previous_retained_expert_ceiling_bytes = self
            .model
            .as_ref()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            })?
            .retained_expert_ceiling_bytes();
        let mut updated_mlx_ram_budget = self
            .mlx_ram_budget
            .as_ref()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna RAM budget is not loaded".to_owned(),
            })?
            .clone();
        updated_mlx_ram_budget
            .update_mlx_active_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes)
            .map_err(|_| InferenceEngineError::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                reason: "Laguna could not apply the requested memory budget".to_owned(),
            })?;
        let mut updated_adaptive_ram_growth_guard = self
            .adaptive_ram_growth_guard
            .as_ref()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna adaptive RAM growth guard is not loaded".to_owned(),
            })?
            .clone();
        let active_memory_limit_bytes = usize::try_from(requested_mlx_memory_ceiling_bytes)
            .map_err(|_| InferenceEngineError::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                reason: "the requested Laguna memory ceiling exceeds this platform".to_owned(),
            })?;
        updated_adaptive_ram_growth_guard
            .update_active_memory_limit_bytes(active_memory_limit_bytes)
            .map_err(|guard_error| InferenceEngineError::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                reason: format!("Laguna adaptive RAM growth rejected the ceiling: {guard_error}"),
            })?;
        let context_token_count = self
            .active_request
            .as_ref()
            .map_or(0, |active_request| active_request.context_token_count);
        let updated_retained_expert_ceiling_bytes = laguna_ram_budget_snapshot(
            &updated_mlx_ram_budget,
            MlxRamBudgetPhase::Prefill,
            context_token_count,
        )
        .retained_expert_budget_bytes;
        if let MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency,
            retained_paged_expert_reclamation_bytes,
        } = ceiling_change_decision
        {
            let mut transition_attribution = PerformanceAttribution::disabled();
            if must_demote_complete_residency {
                self.model
                    .as_mut()
                    .ok_or(InferenceEngineError::Fatal {
                        reason: "the Laguna model is not loaded".to_owned(),
                    })?
                    .demote_native_routed_experts(runtime, &mut transition_attribution)
                    .map_err(|demotion_error| InferenceEngineError::Fatal {
                        reason: format!("Laguna expert demotion failed: {demotion_error}"),
                    })?;
            } else if retained_paged_expert_reclamation_bytes > 0 {
                let retained_payload_target_bytes = expert_statistics
                    .resident_payload_byte_count
                    .saturating_sub(retained_paged_expert_reclamation_bytes)
                    .min(updated_retained_expert_ceiling_bytes);
                self.model
                    .as_ref()
                    .ok_or(InferenceEngineError::Fatal {
                        reason: "the Laguna model is not loaded".to_owned(),
                    })?
                    .set_retained_expert_ceiling(retained_payload_target_bytes)
                    .map_err(|reclamation_error| InferenceEngineError::Fatal {
                        reason: format!("Laguna expert reclamation failed: {reclamation_error}"),
                    })?;
            }
            let post_reclamation_snapshot = match runtime
                .synchronize_gpu_stream_and_clear_allocator_cache()
                .and_then(|()| runtime.memory_snapshot())
            {
                Ok(memory_snapshot) => memory_snapshot,
                Err(memory_error) => {
                    self.model
                        .as_ref()
                        .ok_or(InferenceEngineError::Fatal {
                            reason: "the Laguna model is not loaded".to_owned(),
                        })?
                        .set_retained_expert_ceiling(previous_retained_expert_ceiling_bytes)
                        .map_err(|restore_error| InferenceEngineError::Fatal {
                            reason: format!(
                                "Laguna ceiling cleanup and expert-policy restoration failed: {memory_error}; {restore_error}"
                            ),
                        })?;
                    return Err(InferenceEngineError::Fatal {
                        reason: format!("Laguna ceiling-change cleanup failed: {memory_error}"),
                    });
                }
            };
            if u64::try_from(post_reclamation_snapshot.active_memory_bytes()).unwrap_or(u64::MAX)
                > requested_mlx_memory_ceiling_bytes
            {
                self.model
                    .as_ref()
                    .ok_or(InferenceEngineError::Fatal {
                        reason: "the Laguna model is not loaded".to_owned(),
                    })?
                    .set_retained_expert_ceiling(previous_retained_expert_ceiling_bytes)
                    .map_err(|restore_error| InferenceEngineError::Fatal {
                        reason: format!(
                            "Laguna could not restore expert policy after rejecting the ceiling: {restore_error}"
                        ),
                    })?;
                return Err(InferenceEngineError::MlxMemoryLimitRejected {
                    requested_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes,
                    reason: "Laguna active ownership remains above the requested ceiling after expert reclamation"
                        .to_owned(),
                });
            }
        }
        let updated_limits = MlxMemoryLimits::new(
            active_memory_limit_bytes,
            self.maximum_allocator_cache_memory_limit_bytes
                .min(active_memory_limit_bytes),
        )
        .map_err(|_| InferenceEngineError::MlxMemoryLimitRejected {
            requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            reason: "the requested Laguna memory limits are invalid".to_owned(),
        })?;
        self.model
            .as_ref()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            })?
            .set_retained_expert_ceiling(updated_retained_expert_ceiling_bytes)
            .map_err(|set_error| InferenceEngineError::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                reason: format!("Laguna could not apply the recomposed expert budget: {set_error}"),
            })?;
        if self
            .runtime
            .as_mut()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna runtime is not loaded".to_owned(),
            })?
            .update_memory_limits(updated_limits)
            .is_err()
        {
            self.model
                .as_ref()
                .ok_or(InferenceEngineError::Fatal {
                    reason: "the Laguna model is not loaded".to_owned(),
                })?
                .set_retained_expert_ceiling(previous_retained_expert_ceiling_bytes)
                .map_err(|restore_error| InferenceEngineError::Fatal {
                    reason: format!(
                        "Laguna runtime rejected the ceiling and expert-policy restoration failed: {restore_error}"
                    ),
                })?;
            return Err(InferenceEngineError::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                reason: "the Laguna runtime rejected the requested memory ceiling".to_owned(),
            });
        }
        self.mlx_ram_budget = Some(updated_mlx_ram_budget);
        self.adaptive_ram_growth_guard = Some(updated_adaptive_ram_growth_guard);
        self.model
            .as_mut()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            })?
            .update_expert_allocation_ceiling(requested_mlx_memory_ceiling_bytes);
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            self.model
                .as_ref()
                .map_or(ExpertMemoryMode::Paged, |model| model.expert_memory_mode()),
            self.collect_current_mlx_memory_telemetry(),
        ))
    }
}
