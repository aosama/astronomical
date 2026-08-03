use astronomical_runtime_integration::MlxMemoryLimits;

use crate::{InferenceEngineError, safe_minimum_mlx_memory_ceiling_bytes};

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
    ) -> Result<(u64, bool, MlxMemoryLimits), InferenceEngineError> {
        let current_mlx_memory_limits = self.runtime.memory_limits();
        let current_mlx_memory_ceiling_bytes =
            u64::try_from(current_mlx_memory_limits.active_memory_limit_bytes()).map_err(|_| {
                super::super::inference_execution::fatal_engine_error(
                    "current MLX memory ceiling exceeds the u64 range",
                )
            })?;
        let minimum_mlx_memory_ceiling_bytes = self.minimum_mlx_memory_ceiling_bytes()?;
        if requested_mlx_memory_ceiling_bytes < minimum_mlx_memory_ceiling_bytes {
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
            return Ok((
                minimum_mlx_memory_ceiling_bytes,
                false,
                current_mlx_memory_limits,
            ));
        }

        let requested_mlx_memory_limits = MlxMemoryLimits::new(
            requested_mlx_memory_ceiling_bytes_as_usize,
            requested_mlx_memory_ceiling_bytes_as_usize,
        )
        .map_err(super::super::inference_execution::qwen3_5_runtime_error)?;
        let is_lowering_mlx_memory_ceiling =
            requested_mlx_memory_ceiling_bytes < current_mlx_memory_ceiling_bytes;
        let previous_expert_paging_memory_ceiling_bytes = self
            .expert_pager
            .as_ref()
            .map(|expert_pager| expert_pager.configured_mlx_memory_ceiling_bytes());
        if is_lowering_mlx_memory_ceiling {
            self.prepare_expert_residency_for_lower_mlx_memory_limit(
                requested_mlx_memory_ceiling_bytes,
            )?;
            let post_reclamation_mlx_memory_snapshot = self
                .runtime
                .synchronize_gpu_stream_and_clear_allocator_cache()
                .and_then(|()| self.runtime.memory_snapshot())
                .map_err(super::super::inference_execution::qwen3_5_runtime_error)?;
            if u64::try_from(post_reclamation_mlx_memory_snapshot.active_memory_bytes())
                .unwrap_or(u64::MAX)
                > requested_mlx_memory_ceiling_bytes
            {
                self.restore_expert_paging_memory_ceiling(
                    previous_expert_paging_memory_ceiling_bytes,
                );
                return Err(super::super::inference_execution::fatal_engine_error(
                    "retained model memory remained above the requested MLX ceiling after expert reclamation",
                ));
            }
        }

        if let Err(runtime_update_error) = self
            .runtime
            .update_memory_limits(requested_mlx_memory_limits)
        {
            self.restore_expert_paging_memory_ceiling(previous_expert_paging_memory_ceiling_bytes);
            return Err(super::super::inference_execution::qwen3_5_runtime_error(
                runtime_update_error,
            ));
        }
        if !is_lowering_mlx_memory_ceiling {
            self.update_expert_residency_for_live_mlx_memory_limit(
                requested_mlx_memory_ceiling_bytes,
            )?;
        }
        Ok((
            minimum_mlx_memory_ceiling_bytes,
            true,
            requested_mlx_memory_limits,
        ))
    }

    fn prepare_expert_residency_for_lower_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        let Some(expert_pager) = self.expert_pager.as_mut() else {
            return Ok(());
        };
        let current_memory_budget_snapshot = expert_pager
            .memory_budget_snapshot_for_mlx_memory_limit_adjustment(&self.runtime)
            .map_err(super::super::inference_execution::qwen3_5_runtime_error)?;
        let candidate_memory_budget_snapshot = current_memory_budget_snapshot
            .with_configured_cap_bytes(requested_mlx_memory_ceiling_bytes);
        expert_pager.update_configured_mlx_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes);
        let expert_weight_memory_cache = self
            .expert_weight_memory_cache
            .as_ref()
            .ok_or_else(|| {
                super::super::inference_execution::fatal_engine_error(
                    "sparse Qwen3.5 model has an expert pager without an expert weight memory cache",
                )
            })?;
        let maximum_resident_payload_byte_count = expert_weight_memory_cache
            .borrow()
            .maximum_resident_payload_byte_count_for_memory_budget_snapshot(
                &candidate_memory_budget_snapshot,
            );
        expert_weight_memory_cache
            .borrow_mut()
            .update_maximum_resident_payload_byte_count(maximum_resident_payload_byte_count);
        Ok(())
    }

    fn update_expert_residency_for_live_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        let Some(expert_pager) = self.expert_pager.as_mut() else {
            return Ok(());
        };
        expert_pager.update_configured_mlx_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes);
        let current_memory_budget_snapshot = expert_pager
            .memory_budget_snapshot_for_mlx_memory_limit_adjustment(&self.runtime)
            .map_err(super::super::inference_execution::qwen3_5_runtime_error)?;
        let expert_weight_memory_cache = self
            .expert_weight_memory_cache
            .as_ref()
            .ok_or_else(|| {
                super::super::inference_execution::fatal_engine_error(
                    "sparse Qwen3.5 model has an expert pager without an expert weight memory cache",
                )
            })?;
        let maximum_resident_payload_byte_count = expert_weight_memory_cache
            .borrow()
            .maximum_resident_payload_byte_count_for_memory_budget_snapshot(
                &current_memory_budget_snapshot,
            );
        expert_weight_memory_cache
            .borrow_mut()
            .update_maximum_resident_payload_byte_count(maximum_resident_payload_byte_count);
        Ok(())
    }

    fn restore_expert_paging_memory_ceiling(&mut self, previous_memory_ceiling_bytes: Option<u64>) {
        if let Some(previous_memory_ceiling_bytes) = previous_memory_ceiling_bytes
            && let Some(expert_pager) = self.expert_pager.as_mut()
        {
            expert_pager.update_configured_mlx_memory_ceiling_bytes(previous_memory_ceiling_bytes);
        }
    }
}
