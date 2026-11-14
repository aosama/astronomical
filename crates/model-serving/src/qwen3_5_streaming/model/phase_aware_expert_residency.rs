//! Converts Qwen geometry and composed RAM budgets into the pure residency target.

use crate::expert_paging::QuantizedExpertLayerPlan;
use crate::memory::residency::ExpertLayerGeometry;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::ExpertPagingError;
use crate::qwen3_5_streaming::expert_streaming::retained_expert_layer_page_payload_bytes;
use crate::{
    CurrentExpertLayerResidency, ExpertLayerResidencyTarget, ExpertResidencyPlan, MemoryPhase,
    PerformanceAttribution, PerformanceCounter, PerformanceOperation, RequestExpertResidency,
    RetainedExpertPageClass, RetainedExpertReclamation, plan_expert_residency,
    publish_request_stable_residency_plan, seated_complete_layer_count,
    should_enact_planned_expert_release,
};

use super::streaming_model::Qwen3_5StreamingModel;

impl Qwen3_5StreamingModel {
    /// Opens one request's prefill demand window: the previous request's
    /// prefill evidence is spent, while lifetime decode evidence persists
    /// (see `demand_phase`).
    pub(crate) fn begin_prefill_demand_window_for_residency(&self) {
        if let Some(retained_experts) = self.retained_experts.as_ref() {
            retained_experts.borrow_mut().begin_prefill_demand_window();
        }
    }

    /// Scales prefill demand recorded by subsequent forwards of this chunk.
    /// Decode evidence ignores the multiplier (see `demand_phase`).
    pub(crate) fn set_prefill_demand_assignment_weight_for_residency(
        &self,
        assignment_weight: u64,
    ) {
        if let Some(retained_experts) = self.retained_experts.as_ref() {
            retained_experts
                .borrow_mut()
                .set_prefill_demand_assignment_weight(assignment_weight);
        }
    }

    /// Refreshes the residency plan for direct-MLX performance probes.
    ///
    /// Production calls this at request open (Prefill) and at the decode
    /// handoff; a model-layer probe must run the same publication or the
    /// retained cache keeps a zero ceiling and nothing the probe observes
    /// matches serving behavior.
    #[doc(hidden)]
    pub fn refresh_phase_aware_expert_residency_plan_for_tests(
        &self,
        phase: crate::MemoryPhase,
        context_token_count: u64,
        operation_token_count: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.govern_memory_as_the_engine_would_for_tests(
            phase,
            context_token_count,
            operation_token_count,
        );
        self.refresh_phase_aware_expert_residency_plan(phase, performance_attribution)
    }

    /// What the engine tells the memory governor before it plans: a model
    /// loaded directly has no governor, no probe and so no capacity.
    fn govern_memory_as_the_engine_would_for_tests(
        &self,
        phase: crate::MemoryPhase,
        context_token_count: u64,
        operation_token_count: u64,
    ) {
        if self.governed_memory.governor.borrow().is_none() {
            self.probe_expert_layer_working_memory(super::LayerProbeShape {
                prefill_chunk_token_count: usize::try_from(operation_token_count)
                    .unwrap_or(usize::MAX),
                full_attention_kv_state_growth_tokens: 256,
            });
            self.begin_serving_session_observation();
            self.open_memory_governor();
        }
        match phase {
            MemoryPhase::Prefill => self.govern_request_start(
                context_token_count,
                u32::try_from(operation_token_count).unwrap_or(u32::MAX),
            ),
            MemoryPhase::GenerationPreparation | MemoryPhase::Decode => self.govern_reply_start(),
            MemoryPhase::Idle => {}
        }
    }

    /// Clears an optional optimization target so execution safely streams misses.
    pub(crate) fn clear_phase_aware_expert_residency_plan(&self) {
        *self.active_expert_residency_plan.borrow_mut() = None;
        *self.request_expert_residency.borrow_mut() = None;
    }

    /// Refreshes the no-I/O target against the capacity the memory governor
    /// has given the expert cache.
    pub(crate) fn refresh_phase_aware_expert_residency_plan(
        &self,
        phase: MemoryPhase,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            *self.active_expert_residency_plan.borrow_mut() = None;
            return Ok(());
        };
        let retained_expert_ceiling_bytes = self.kept_expert_capacity_bytes();
        let experts_per_token =
            usize::try_from(self.config.experts_per_token()).unwrap_or(usize::MAX);
        let layer_geometries =
            retained_expert_layer_geometries(expert_pager.layer_plans(), experts_per_token)?;
        let (retained_page_ceiling_bytes, budget_reclamation, current_residencies) =
            self.refresh_retained_expert_ceiling(retained_expert_ceiling_bytes, expert_pager)?;
        tracing::info!(
            current_residency_count = current_residencies.len(),
            current_residency_payload_bytes = summed_residency_payload_bytes(&current_residencies),
            leftover_expert_budget_bytes = retained_expert_ceiling_bytes,
            retained_page_ceiling_bytes,
            "topology snapshot before residency planning"
        );
        let residency_plan = performance_attribution
            .measure_operation(
                PerformanceOperation::ExpertResidencyPlanning,
                |_performance_attribution| {
                    plan_expert_residency(
                        phase,
                        retained_page_ceiling_bytes,
                        &layer_geometries,
                        &current_residencies,
                    )
                },
            )
            .map_err(|plan_error| ExpertPagingError::InvalidPagingPlan {
                description: plan_error.to_string(),
            })?;
        self.publish_phase_aware_residency_decisions(
            phase,
            retained_expert_ceiling_bytes,
            residency_plan,
            &layer_geometries,
            &budget_reclamation,
            &current_residencies,
            performance_attribution,
        );
        Ok(())
    }

    /// Publishes the request-stable plan, enacts its releases without loading
    /// replacements, reports the occupancy-truth telemetry, and installs the
    /// plan as the active residency target.
    #[allow(clippy::too_many_arguments)]
    fn publish_phase_aware_residency_decisions(
        &self,
        phase: MemoryPhase,
        retained_expert_ceiling_bytes: u64,
        residency_plan: ExpertResidencyPlan,
        layer_geometries: &[ExpertLayerGeometry],
        budget_reclamation: &RetainedExpertReclamation,
        current_residencies: &[CurrentExpertLayerResidency],
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let (next_request_residency, residency_plan) = publish_request_stable_residency_plan(
            phase,
            self.request_expert_residency.borrow().as_ref(),
            residency_plan,
            current_residencies,
            budget_reclamation.released_complete_payload_bytes,
            layer_geometries,
        );
        *self.request_expert_residency.borrow_mut() = next_request_residency;
        if phase == MemoryPhase::GenerationPreparation {
            record_generation_preparation_plan_counters(
                performance_attribution,
                &residency_plan,
                current_residencies,
            );
        }
        // Enact releases without loading replacements. This makes the planner's
        // routed reservations real before any later mandatory route page commits.
        self.enact_planned_expert_releases(
            phase,
            &residency_plan,
            current_residencies,
            performance_attribution,
        );
        // Occupancy truth first: the pinned count published here must equal
        // the layers actually resident as complete layers, because a contract
        // pin is intent, not ownership. The 2026-10-24 flight logged
        // `pinned_complete_layer_count=34` beside `current_residency_count=9`
        // because both the contract's pin roles and the occupancy were
        // conflated into one field; the contract's pin count is published
        // separately as `contracted_complete_layer_count`.
        let seated_complete_layer_count = seated_complete_layer_count(current_residencies);
        let contracted_complete_layer_count = self
            .request_expert_residency
            .borrow()
            .as_ref()
            .map(request_residency_pinned_layer_count)
            .unwrap_or(0);
        publish_phase_aware_residency_plan_info(
            phase,
            retained_expert_ceiling_bytes,
            &residency_plan,
            seated_complete_layer_count,
            contracted_complete_layer_count,
            budget_reclamation,
        );
        *self.active_expert_residency_plan.borrow_mut() = Some(residency_plan);
    }

    /// Snapshots what the cache keeps for the planner. The capacity is the
    /// governor's and is not changed here; nothing is given back.
    fn refresh_retained_expert_ceiling(
        &self,
        retained_expert_ceiling_bytes: u64,
        expert_pager: &crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5ExpertPager,
    ) -> Result<
        (
            u64,
            RetainedExpertReclamation,
            Vec<CurrentExpertLayerResidency>,
        ),
        Qwen3_5ExecutionError,
    > {
        let experts_per_token =
            usize::try_from(self.config.experts_per_token()).unwrap_or(usize::MAX);
        let layer_geometries =
            retained_expert_layer_geometries(expert_pager.layer_plans(), experts_per_token)?;
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return Err(ExpertPagingError::InvalidPagingPlan {
                description: "paged Qwen model lost retained expert ownership".to_owned(),
            }
            .into());
        };
        let expert_capacity = layer_geometries
            .first()
            .map(|geometry| geometry.expert_capacity)
            .unwrap_or(0);
        let current_residencies = retained_experts.borrow().topology_snapshot(expert_capacity);
        Ok((
            retained_expert_ceiling_bytes,
            RetainedExpertReclamation::default(),
            current_residencies,
        ))
    }

    /// Enacts planned releases without loading replacements. This makes the
    /// planner's routed reservations real before any later mandatory route
    /// page commits.
    fn enact_planned_expert_releases(
        &self,
        phase: MemoryPhase,
        residency_plan: &ExpertResidencyPlan,
        current_residencies: &[CurrentExpertLayerResidency],
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return;
        };
        for current_residency in current_residencies {
            let target = residency_plan.layer_targets[current_residency.layer_index];
            if !should_enact_planned_expert_release(phase, target) {
                continue;
            }
            if retained_experts
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
    }

    /// Reads the active residency plan's target for one layer.
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

/// Pure: total payload bytes across a topology snapshot, saturated — the
/// topology telemetry's `current_residency_payload_bytes` field.
fn summed_residency_payload_bytes(current_residencies: &[CurrentExpertLayerResidency]) -> u64 {
    let mut total_payload_bytes = 0_u64;
    for current_residency in current_residencies {
        total_payload_bytes = total_payload_bytes.saturating_add(current_residency.payload_bytes);
    }
    total_payload_bytes
}

/// Pure: how many complete layers a request-stable residency contract pins.
fn request_residency_pinned_layer_count(request_residency: &RequestExpertResidency) -> usize {
    request_residency.pinned_complete_layer_indexes().len()
}

/// Logs the published phase-aware expert residency plan with its occupancy
/// and reclamation facts.
fn publish_phase_aware_residency_plan_info(
    phase: MemoryPhase,
    retained_expert_ceiling_bytes: u64,
    residency_plan: &ExpertResidencyPlan,
    seated_complete_layer_count: usize,
    contracted_complete_layer_count: usize,
    budget_reclamation: &RetainedExpertReclamation,
) {
    let streamed_layer_count = count_planned_streamed_layers(&residency_plan.layer_targets);
    tracing::info!(
        ?phase,
        retained_expert_ceiling_bytes,
        complete_layer_target_count = residency_plan.complete_layer_targets.len(),
        pinned_complete_layer_count = seated_complete_layer_count,
        contracted_complete_layer_count,
        streamed_layer_count,
        released_complete_payload_bytes = budget_reclamation.released_complete_payload_bytes,
        released_partial_payload_bytes = budget_reclamation.released_partial_payload_bytes,
        reserved_routed_overlay_bytes = residency_plan.reserved_routed_overlay_bytes,
        expected_preserved_bytes = residency_plan.expected_preserved_bytes,
        maximum_new_retained_bytes = residency_plan.maximum_new_retained_bytes,
        low_budget_partial_mode = residency_plan.is_low_budget_partial_mode,
        "published phase-aware expert residency plan"
    );
}

/// Pure: counts the planned partial-layer targets for the
/// GenerationPreparation telemetry.
fn count_planned_partial_layers(layer_targets: &[ExpertLayerResidencyTarget]) -> usize {
    layer_targets
        .iter()
        .filter(|target| {
            matches!(
                target,
                ExpertLayerResidencyTarget::PreservePartial
                    | ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
            )
        })
        .count()
}

/// Pure: counts the planned streamed-layer targets for the
/// GenerationPreparation telemetry.
fn count_planned_streamed_layers(layer_targets: &[ExpertLayerResidencyTarget]) -> usize {
    layer_targets
        .iter()
        .filter(|target| {
            matches!(
                target,
                ExpertLayerResidencyTarget::StreamOperationLocal
                    | ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
                    | ExpertLayerResidencyTarget::ReleasePartial
                    | ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
            )
        })
        .count()
}

/// Records the GenerationPreparation plan telemetry: the complete, partial,
/// and streamed target counts plus the per-residency pre-existing and
/// preserved payload attribution.
fn record_generation_preparation_plan_counters(
    performance_attribution: &mut PerformanceAttribution,
    residency_plan: &ExpertResidencyPlan,
    current_residencies: &[CurrentExpertLayerResidency],
) {
    performance_attribution.record_counter(
        PerformanceCounter::ExpertResidencyPlanCompleteLayerCount,
        u64::try_from(residency_plan.complete_layer_targets.len()).unwrap_or(u64::MAX),
    );
    let planned_partial_layer_count = count_planned_partial_layers(&residency_plan.layer_targets);
    let planned_streamed_layer_count = count_planned_streamed_layers(&residency_plan.layer_targets);
    performance_attribution.record_counter(
        PerformanceCounter::ExpertResidencyPlanPartialLayerCount,
        u64::try_from(planned_partial_layer_count).unwrap_or(u64::MAX),
    );
    performance_attribution.record_counter(
        PerformanceCounter::ExpertResidencyPlanStreamedLayerCount,
        u64::try_from(planned_streamed_layer_count).unwrap_or(u64::MAX),
    );
    for current_residency in current_residencies {
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

pub(in crate::qwen3_5_streaming) fn retained_expert_layer_geometries(
    layer_plans: &[QuantizedExpertLayerPlan],
    experts_per_token: usize,
) -> Result<Vec<ExpertLayerGeometry>, ExpertPagingError> {
    layer_plans
        .iter()
        .enumerate()
        .map(|(layer_index, layer_plan)| {
            Ok(ExpertLayerGeometry {
                layer_index,
                complete_layer_payload_bytes: retained_expert_layer_page_payload_bytes(
                    layer_plan,
                    layer_plan.expert_capacity,
                )?,
                expert_payload_bytes: retained_expert_layer_page_payload_bytes(layer_plan, 1)?,
                expert_capacity: layer_plan.expert_capacity,
                experts_per_token,
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "phase_aware_expert_residency_tests.rs"]
mod phase_aware_expert_residency_tests;
