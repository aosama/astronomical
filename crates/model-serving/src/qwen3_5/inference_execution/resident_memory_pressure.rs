//! Request-time escape from a resident model that no longer fits projected growth.
//!
//! The first projection is allowed to observe resident ownership. If it fails,
//! this owner demotes the complete sparse payload, samples the resulting paged
//! baseline, reserves one routed expert page, and repeats the same exact growth
//! projection. Callers continue into page-level reclamation only if needed.

use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::qwen3_5::model::adaptive_ram_growth_logging::log_adaptive_ram_growth_pressure;
use crate::qwen3_5::model::memory_admission::invalid_request_error;
use crate::qwen3_5_moe::Qwen3_5ExpertResidencyTransitionReason;
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthProjection, InferenceEngineError,
    PerformanceAttribution,
};

use super::memory_admission::AdaptiveRamGrowthMemoryAdmissionError;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

pub(super) struct Qwen3_5ResidentAdaptiveGrowthDemotion {
    pub(super) adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    pub(super) memory_snapshot: MlxMemorySnapshot,
    pub(super) routed_expert_page_reservation_bytes: usize,
    pub(super) projection: AdaptiveRamGrowthProjection,
}

impl Qwen3_5EngineState {
    pub(super) fn demote_resident_experts_for_adaptive_growth(
        &mut self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        exact_persistent_growth_bytes: usize,
        exact_temporary_workspace_bytes: usize,
    ) -> Result<Option<Qwen3_5ResidentAdaptiveGrowthDemotion>, AdaptiveRamGrowthMemoryAdmissionError>
    {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        if model.resident_expert_weights.is_none() {
            return Ok(None);
        }
        // Preserve before/after evidence because the resident owner disappears
        // as one allocation class rather than as individually evicted pages.
        let resident_expert_statistics_before_demotion =
            model.expert_weight_memory_cache_statistics();
        let allocator_cache_memory_bytes_before_demotion = model
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?
            .allocator_cache_memory_bytes();
        let mut disabled_transition_attribution = PerformanceAttribution::disabled();
        self.model
            .as_mut()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .demote_resident_experts_to_paging(
                Qwen3_5ExpertResidencyTransitionReason::RequestPressure,
                &mut disabled_transition_attribution,
            )
            .map_err(InferenceEngineError::from)?;

        let adaptive_ram_growth_context =
            adaptive_ram_growth_context.with_sparse_experts_are_paged(true);
        let model_after_demotion = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let memory_snapshot = model_after_demotion
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        let routed_expert_page_reservation_bytes = model_after_demotion
            .expert_pager
            .as_ref()
            .map_or(0, |expert_pager| expert_pager.maximum_expert_page_bytes())
            .try_into()
            .map_err(|_| {
                invalid_request_error("routed expert page reservation exceeds the platform range")
            })?;
        let projection = self
            .adaptive_ram_growth_guard
            .project_growth_for_context(
                adaptive_ram_growth_context,
                memory_snapshot.active_memory_bytes(),
                exact_persistent_growth_bytes,
                routed_expert_page_reservation_bytes,
                exact_temporary_workspace_bytes,
            )
            .map_err(|adaptive_ram_growth_projection_error| {
                invalid_request_error(format!(
                    "adaptive RAM growth rejected after resident expert demotion: {adaptive_ram_growth_projection_error}"
                ))
            })?;
        log_adaptive_ram_growth_pressure(
            &projection,
            resident_expert_statistics_before_demotion,
            model_after_demotion.expert_weight_memory_cache_statistics(),
            allocator_cache_memory_bytes_before_demotion,
            resident_expert_statistics_before_demotion
                .resident_payload_byte_count
                .try_into()
                .unwrap_or(usize::MAX),
            "demote_resident_experts",
        );
        Ok(Some(Qwen3_5ResidentAdaptiveGrowthDemotion {
            adaptive_ram_growth_context,
            memory_snapshot,
            routed_expert_page_reservation_bytes,
            projection,
        }))
    }
}
