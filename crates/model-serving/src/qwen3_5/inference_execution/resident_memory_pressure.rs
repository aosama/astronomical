//! Request-time escape from a resident model that no longer fits projected growth.
//!
//! Context and adaptive-growth admission first observe resident ownership. If
//! that projection fails, this owner demotes the complete sparse payload and
//! repeats the same exact projection from the resulting paged baseline. Callers
//! continue into page-level reclamation only if needed.

use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::qwen3_5::model::adaptive_ram_growth_logging::log_adaptive_ram_growth_pressure;
use crate::qwen3_5::model::memory_admission::{
    context_memory_admission_fits_without_expert_reclamation, invalid_request_error,
    validate_context_memory_admission,
};
use crate::qwen3_5_moe::Qwen3_5ExpertResidencyTransitionReason;
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthProjection, InferenceEngineError,
    PerformanceAttribution, PerformanceOperation,
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
    /// Applies the binary resident-to-paged transition before every context
    /// admission boundary, including cache restoration and injected feedback.
    pub(super) fn validate_context_memory_admission_with_resident_expert_demotion(
        &mut self,
        context_token_count_requiring_reservation: usize,
        temporary_workspace_reservation_bytes: usize,
        additional_maximum_expert_page_reservation_bytes: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<u64, InferenceEngineError> {
        let target_expert_payload_bytes_before_context_admission = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        // Demote only when this exact operation would otherwise fail. Smaller
        // operations retain the faster complete-model resident path.
        let resident_model_requires_demotion = {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            model.resident_expert_weights.is_some()
                && !context_memory_admission_fits_without_expert_reclamation(
                    model,
                    self.memory_limits,
                    self.context_memory_reservation_bytes_per_token,
                    context_token_count_requiring_reservation,
                    temporary_workspace_reservation_bytes,
                    additional_maximum_expert_page_reservation_bytes,
                )?
        };
        if resident_model_requires_demotion {
            self.model
                .as_mut()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
                .demote_resident_experts_to_paging(
                    Qwen3_5ExpertResidencyTransitionReason::RequestAdmission,
                    performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let context_admission_outcome = performance_attribution.measure_operation(
            PerformanceOperation::MemoryAdmissionSnapshot,
            |_performance_attribution| {
                validate_context_memory_admission(
                    model,
                    self.memory_limits,
                    self.context_memory_reservation_bytes_per_token,
                    context_token_count_requiring_reservation,
                    temporary_workspace_reservation_bytes,
                    additional_maximum_expert_page_reservation_bytes,
                )
            },
        );
        let target_expert_payload_bytes_after_context_admission = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let target_expert_payload_bytes_reclaimed_during_context_admission =
            target_expert_payload_bytes_before_context_admission
                .saturating_sub(target_expert_payload_bytes_after_context_admission);
        context_admission_outcome?;
        Ok(target_expert_payload_bytes_reclaimed_during_context_admission)
    }

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
            .expert_page_reservation_bytes_for_forward(
                adaptive_ram_growth_context.forward_token_count(),
            )
            .map_err(InferenceEngineError::from)?
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
