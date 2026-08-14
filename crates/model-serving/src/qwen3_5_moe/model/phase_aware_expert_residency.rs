//! Converts Qwen geometry and composed RAM budgets into the pure residency target.

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::expert_paging::expert_pager::ExpertPagingError;
use crate::{
    ExpertLayerGeometry, ExpertLayerResidencyTarget, ExpertResidencyPhase, MlxRamBudgetPhase,
    PerformanceAttribution, PerformanceCounter, PerformanceOperation, RetainedExpertPageClass,
    RetainedExpertReclamation, plan_phase_aware_expert_residency,
};

impl Qwen3_5Model {
    /// Applies the last-chunk or ordinary demand multiplier for later recordings.
    pub(crate) fn set_expert_demand_assignment_weight(&self, assignment_weight: u64) {
        if let Some(retained_expert_layers) = self.retained_expert_layers.as_ref() {
            retained_expert_layers
                .borrow_mut()
                .set_demand_assignment_weight(assignment_weight);
        }
    }

    /// Clears an optional optimization target so execution safely streams misses.
    pub(crate) fn clear_phase_aware_expert_residency_plan(&self) {
        *self.active_expert_residency_plan.borrow_mut() = None;
    }

    /// Refreshes the no-I/O target after memory policy has established a phase budget.
    pub(crate) fn refresh_phase_aware_expert_residency_plan(
        &self,
        phase: ExpertResidencyPhase,
        context_token_count: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            *self.active_expert_residency_plan.borrow_mut() = None;
            return Ok(());
        };
        if self.resident_expert_weights.is_some() {
            if phase == ExpertResidencyPhase::GenerationPreparation {
                let resident_statistics = self.expert_weight_memory_cache_statistics();
                performance_attribution.record_counter(
                    PerformanceCounter::ExpertResidencyPlanCompleteLayerCount,
                    u64::try_from(resident_statistics.complete_layer_count).unwrap_or(u64::MAX),
                );
                performance_attribution.record_counter(
                    PerformanceCounter::ExpertResidencyPreexistingCompletePayloadBytes,
                    resident_statistics.complete_layer_payload_byte_count,
                );
                performance_attribution.record_counter(
                    PerformanceCounter::ExpertResidencyPreservedCompletePayloadBytes,
                    resident_statistics.complete_layer_payload_byte_count,
                );
                performance_attribution.record_counter(
                    PerformanceCounter::ExpertTopologyPreservedPayloadBytes,
                    resident_statistics.complete_layer_payload_byte_count,
                );
            }
            *self.active_expert_residency_plan.borrow_mut() = None;
            return Ok(());
        }
        let budget_phase = match phase {
            ExpertResidencyPhase::Prefill => MlxRamBudgetPhase::Prefill,
            ExpertResidencyPhase::GenerationPreparation | ExpertResidencyPhase::Decode => {
                MlxRamBudgetPhase::Decode
            }
            ExpertResidencyPhase::Idle => MlxRamBudgetPhase::Idle,
        };
        let retained_expert_ceiling_bytes = self
            .mlx_ram_budget
            .borrow()
            .plan(budget_phase, context_token_count, 0)
            .retained_expert_budget_bytes;
        let layer_geometries = expert_pager
            .layer_plans()
            .iter()
            .enumerate()
            .map(|(layer_index, layer_plan)| {
                Ok(ExpertLayerGeometry {
                    layer_index,
                    complete_layer_payload_bytes: layer_plan
                        .complete_expert_payload_byte_count()?,
                    expert_payload_bytes: layer_plan.expert_payload_byte_count()?,
                    expert_capacity: layer_plan.expert_capacity,
                    experts_per_token: usize::try_from(self.config.experts_per_token())
                        .unwrap_or(usize::MAX),
                })
            })
            .collect::<Result<Vec<_>, crate::ExpertManifestError>>()
            .map_err(ExpertPagingError::from)?;
        let Some(retained_expert_layers) = self.retained_expert_layers.as_ref() else {
            return Err(ExpertPagingError::InvalidPagingPlan {
                description: "paged Qwen model lost retained expert ownership".to_owned(),
            }
            .into());
        };
        let budget_reclamation = retained_expert_layers
            .borrow_mut()
            .update_maximum_resident_payload_bytes(retained_expert_ceiling_bytes);
        record_expert_reclamation_attribution(performance_attribution, budget_reclamation);
        let current_residencies = retained_expert_layers.borrow().topology_snapshot();
        let residency_plan = performance_attribution
            .measure_operation(
                PerformanceOperation::PhaseAwareExpertResidencyPlanning,
                |_performance_attribution| {
                    plan_phase_aware_expert_residency(
                        phase,
                        retained_expert_ceiling_bytes,
                        &layer_geometries,
                        &current_residencies,
                    )
                },
            )
            .map_err(|plan_error| ExpertPagingError::InvalidPagingPlan {
                description: plan_error.to_string(),
            })?;
        if phase == ExpertResidencyPhase::GenerationPreparation {
            performance_attribution.record_counter(
                PerformanceCounter::ExpertResidencyPlanCompleteLayerCount,
                u64::try_from(residency_plan.complete_layer_targets.len()).unwrap_or(u64::MAX),
            );
            let planned_partial_layer_count = residency_plan
                .layer_targets
                .iter()
                .filter(|target| {
                    matches!(
                        target,
                        ExpertLayerResidencyTarget::PreservePartial
                            | ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
                    )
                })
                .count();
            let planned_streamed_layer_count = residency_plan
                .layer_targets
                .iter()
                .filter(|target| {
                    matches!(
                        target,
                        ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
                            | ExpertLayerResidencyTarget::StreamOperationLocal
                            | ExpertLayerResidencyTarget::ReleasePartial
                            | ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
                    )
                })
                .count();
            performance_attribution.record_counter(
                PerformanceCounter::ExpertResidencyPlanPartialLayerCount,
                u64::try_from(planned_partial_layer_count).unwrap_or(u64::MAX),
            );
            performance_attribution.record_counter(
                PerformanceCounter::ExpertResidencyPlanStreamedLayerCount,
                u64::try_from(planned_streamed_layer_count).unwrap_or(u64::MAX),
            );
            for current_residency in &current_residencies {
                let (preexisting_counter, preserved_counter) = match current_residency.class {
                    RetainedExpertPageClass::StableCompleteLayer => (
                        PerformanceCounter::ExpertResidencyPreexistingCompletePayloadBytes,
                        PerformanceCounter::ExpertResidencyPreservedCompletePayloadBytes,
                    ),
                    RetainedExpertPageClass::ElasticRoutedExperts => (
                        PerformanceCounter::ExpertResidencyPreexistingPartialPayloadBytes,
                        PerformanceCounter::ExpertResidencyPreservedPartialPayloadBytes,
                    ),
                };
                performance_attribution
                    .record_counter(preexisting_counter, current_residency.payload_bytes);
                if matches!(
                    residency_plan.layer_targets[current_residency.layer_index],
                    ExpertLayerResidencyTarget::PreserveComplete
                        | ExpertLayerResidencyTarget::PreservePartial
                        | ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
                ) {
                    performance_attribution
                        .record_counter(preserved_counter, current_residency.payload_bytes);
                    performance_attribution.record_counter(
                        PerformanceCounter::ExpertTopologyPreservedPayloadBytes,
                        current_residency.payload_bytes,
                    );
                }
            }
        }
        // Enact releases without loading replacements. This makes the planner's
        // routed reservations real before any later mandatory route page commits.
        for current_residency in &current_residencies {
            let target = residency_plan.layer_targets[current_residency.layer_index];
            if !matches!(
                target,
                ExpertLayerResidencyTarget::ReleasePartial
                    | ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
            ) {
                continue;
            }
            if retained_expert_layers
                .borrow_mut()
                .remove_layer(current_residency.layer_index)
            {
                let class_counter = match current_residency.class {
                    RetainedExpertPageClass::StableCompleteLayer => {
                        PerformanceCounter::ExpertResidencyRetiredCompletePayloadBytes
                    }
                    RetainedExpertPageClass::ElasticRoutedExperts => {
                        PerformanceCounter::ExpertResidencyRetiredPartialPayloadBytes
                    }
                };
                performance_attribution
                    .record_counter(class_counter, current_residency.payload_bytes);
                performance_attribution.record_counter(
                    PerformanceCounter::ExpertTopologyRetiredPayloadBytes,
                    current_residency.payload_bytes,
                );
            }
        }
        tracing::info!(
            ?phase,
            retained_expert_ceiling_bytes,
            complete_layer_target_count = residency_plan.complete_layer_targets.len(),
            reserved_routed_overlay_bytes = residency_plan.reserved_routed_overlay_bytes,
            expected_preserved_bytes = residency_plan.expected_preserved_bytes,
            maximum_new_retained_bytes = residency_plan.maximum_new_retained_bytes,
            low_budget_partial_mode = residency_plan.is_low_budget_partial_mode,
            "published phase-aware expert residency plan"
        );
        if phase == ExpertResidencyPhase::GenerationPreparation {
            retained_expert_layers.borrow_mut().clear_expert_demand();
        }
        *self.active_expert_residency_plan.borrow_mut() = Some(residency_plan);
        Ok(())
    }

    /// Returns the current execution-required read-through action for one layer.
    pub(super) fn expert_residency_target(
        &self,
        layer_index: usize,
    ) -> Option<ExpertLayerResidencyTarget> {
        self.active_expert_residency_plan
            .borrow()
            .as_ref()?
            .layer_targets
            .get(layer_index)
            .copied()
    }
}

/// Attributes ownership released while applying one composed retained-page ceiling.
pub(crate) fn record_expert_reclamation_attribution(
    performance_attribution: &mut PerformanceAttribution,
    reclamation: RetainedExpertReclamation,
) {
    if reclamation.released_partial_payload_bytes > 0 {
        performance_attribution.record_counter(
            PerformanceCounter::ExpertResidencyRetiredPartialPayloadBytes,
            reclamation.released_partial_payload_bytes,
        );
    }
    if reclamation.released_complete_payload_bytes > 0 {
        performance_attribution.record_counter(
            PerformanceCounter::ExpertResidencyRetiredCompletePayloadBytes,
            reclamation.released_complete_payload_bytes,
        );
    }
    let released_payload_bytes = reclamation.released_payload_bytes();
    if released_payload_bytes > 0 {
        performance_attribution.record_counter(
            PerformanceCounter::ExpertTopologyRetiredPayloadBytes,
            released_payload_bytes,
        );
    }
}
