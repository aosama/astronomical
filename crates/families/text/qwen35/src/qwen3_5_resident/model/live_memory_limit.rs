//! Atomic coordination of a live MLX ceiling and binary expert ownership.
//!
//! Resident lowering reclaims allocator memory before installing a smaller MLX
//! ceiling; if complete weights still exceed it, the daemon replaces the
//! worker instead. Raising lets MLX accept capacity before Rust publishes a
//! larger budget and attempts complete residency. A failed raise restores both
//! native and Rust ceilings.

use runtime_integration::MlxMemoryLimits;

use crate::{
    InferenceEngineError, MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements,
    geometry_safe_minimum_mlx_memory_ceiling_bytes,
};

use crate::qwen3_5_core::model::live_memory_observation::{
    LiveMemoryObservation, observe_live_memory, post_reclamation_active_memory_bytes,
    requested_mlx_memory_limits_when_changed,
};

use super::Qwen3_5ResidentModel;

impl Qwen3_5ResidentModel {
    pub fn minimum_mlx_memory_ceiling_bytes(&self) -> Result<u64, InferenceEngineError> {
        Ok(geometry_safe_minimum_mlx_memory_ceiling_bytes(
            &self.memory_facts().model_geometry(),
        ))
    }

    pub fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(u64, MlxMemoryLimits), InferenceEngineError> {
        let LiveMemoryObservation {
            current_mlx_memory_limits,
            current_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            current_active_memory_bytes,
        } = observe_live_memory(&self.runtime, self.minimum_mlx_memory_ceiling_bytes()?)?;
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
            if requested_mlx_memory_ceiling_bytes < minimum_mlx_memory_ceiling_bytes {
                // Even a streaming replacement cannot serve below the artifact's
                // geometry floor, so the honest answer is the typed rejection.
                return Err(InferenceEngineError::MlxMemoryLimitRejected {
                    requested_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes,
                    reason: "the requested ceiling is below the model's streaming execution floor"
                        .to_owned(),
                });
            }
            return Err(InferenceEngineError::ResidentStreamingReplacementRequired {
                reason:
                    "the resident engine cannot shrink below its loaded memory; the requested ceiling requires streaming execution"
                        .to_owned(),
            });
        }
        let Some(requested_mlx_memory_limits) = requested_mlx_memory_limits_when_changed(
            requested_mlx_memory_ceiling_bytes,
            current_mlx_memory_limits,
        )?
        else {
            return Ok((minimum_mlx_memory_ceiling_bytes, current_mlx_memory_limits));
        };
        let is_lowering_mlx_memory_ceiling = matches!(
            ceiling_change_decision,
            MemoryCeilingChangeDecision::Lower { .. }
        );
        if is_lowering_mlx_memory_ceiling {
            // Reclamation runs under the old ceiling so cleanup can synchronize
            // and release allocator memory before native enforcement tightens.
            let post_reclamation_active_memory_bytes =
                post_reclamation_active_memory_bytes(&self.runtime)?;
            if post_reclamation_active_memory_bytes > requested_mlx_memory_ceiling_bytes {
                return Err(InferenceEngineError::ResidentStreamingReplacementRequired {
                    reason: "resident model memory remained above the requested MLX ceiling after reclamation; streaming execution is required".to_owned(),
                });
            }
        }

        if let Err(runtime_update_error) = self
            .runtime
            .update_memory_limits(requested_mlx_memory_limits)
        {
            return Err(text_generation::engine_errors::fatal_runtime_error(
                runtime_update_error,
            ));
        }
        Ok((
            minimum_mlx_memory_ceiling_bytes,
            requested_mlx_memory_limits,
        ))
    }
}
