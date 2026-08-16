//! Live Laguna MLX-ceiling validation and application.

use astronomical_ipc_protocol::ExpertMemoryMode;
use astronomical_runtime_integration::MlxMemoryLimits;

use crate::{InferenceEngineError, MlxMemoryLimitAdjustment, MlxRamBudgetPhase};

use super::execution::LagunaInferenceExecution;
use super::memory::laguna_ram_budget_snapshot;

impl LagunaInferenceExecution {
    pub(super) fn apply_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        let minimum_mlx_memory_ceiling_bytes = self.minimum_mlx_memory_ceiling_bytes;
        if requested_mlx_memory_ceiling_bytes < minimum_mlx_memory_ceiling_bytes {
            return Err(InferenceEngineError::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                reason: "the requested ceiling cannot preserve Laguna model core and mandatory runtime work"
                    .to_owned(),
            });
        }
        if let Some(mlx_ram_budget) = self.mlx_ram_budget.as_mut() {
            mlx_ram_budget
                .update_mlx_active_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes)
                .map_err(|_| InferenceEngineError::MlxMemoryLimitRejected {
                    requested_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes,
                    reason: "Laguna could not apply the requested memory budget".to_owned(),
                })?;
        }
        if let Some(runtime) = self.runtime.as_mut() {
            let allocator_cache_memory_limit_bytes =
                runtime.memory_limits().allocator_cache_memory_limit_bytes();
            let active_memory_limit_bytes = usize::try_from(requested_mlx_memory_ceiling_bytes)
                .map_err(|_| InferenceEngineError::MlxMemoryLimitRejected {
                    requested_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes,
                    reason: "the requested Laguna memory ceiling exceeds this platform".to_owned(),
                })?;
            let updated_limits = MlxMemoryLimits::new(
                active_memory_limit_bytes,
                allocator_cache_memory_limit_bytes,
            )
            .map_err(|_| InferenceEngineError::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                reason: "the requested Laguna memory limits are invalid".to_owned(),
            })?;
            runtime.update_memory_limits(updated_limits).map_err(|_| {
                InferenceEngineError::MlxMemoryLimitRejected {
                    requested_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes,
                    reason: "the Laguna runtime rejected the requested memory ceiling".to_owned(),
                }
            })?;
        }
        if let (Some(model), Some(mlx_ram_budget)) =
            (self.model.as_ref(), self.mlx_ram_budget.as_ref())
        {
            let context_token_count = self
                .active_request
                .as_ref()
                .map_or(0, |active_request| active_request.context_token_count);
            let retained_expert_budget_bytes = laguna_ram_budget_snapshot(
                mlx_ram_budget,
                MlxRamBudgetPhase::Prefill,
                context_token_count,
            )
            .retained_expert_budget_bytes;
            model
                .set_retained_expert_ceiling(retained_expert_budget_bytes)
                .map_err(|_| InferenceEngineError::MlxMemoryLimitRejected {
                    requested_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes,
                    reason: "Laguna could not apply the recomposed expert budget".to_owned(),
                })?;
        }
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
