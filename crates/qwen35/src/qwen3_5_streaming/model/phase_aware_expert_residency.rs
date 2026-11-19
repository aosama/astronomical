//! Where each expert layer should end up while a reply is written, and how
//! the engine asks about it (issue #2221).
//!
//! While a prompt is read there is no plan: every layer may be kept whole
//! as it is read, and the `astro_expert_cache` capacity, which the memory governor sets,
//! is the only refusal. When the reply begins the capacity rises, and one
//! plan says which layers to keep whole, which keep their hot experts, and
//! which keep nothing. The plan never gives anything back; only the
//! capacity does.

use crate::expert_paging::QuantizedExpertLayerPlan;
use crate::memory::residency::ExpertLayerGeometry;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::ExpertPagingError;
use crate::qwen3_5_streaming::expert_streaming::retained_expert_layer_page_payload_bytes;
use crate::{
    CurrentExpertLayerResidency, ExpertLayerState, ExpertResidencyPlan, MemoryPhase,
    PerformanceAttribution, PerformanceCounter, PerformanceOperation, RetainedExpertPageClass,
    plan_expert_residency, seated_complete_layer_count,
};

use super::streaming_model::Qwen3_5StreamingModel;

impl Qwen3_5StreamingModel {
    /// Opens one request's prefill demand window: the previous request's
    /// prefill evidence is spent, while lifetime decode evidence persists
    /// (see `demand_phase`).
    pub fn begin_prefill_demand_window_for_residency(&self) {
        if let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() {
            astro_expert_cache
                .borrow_mut()
                .begin_prefill_demand_window();
        }
    }

    /// Scales prefill demand recorded by subsequent forwards of this chunk.
    /// Decode evidence ignores the multiplier (see `demand_phase`).
    pub fn set_prefill_demand_assignment_weight_for_residency(&self, assignment_weight: u64) {
        if let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() {
            astro_expert_cache
                .borrow_mut()
                .set_prefill_demand_assignment_weight(assignment_weight);
        }
    }

    /// Puts a model loaded directly into the memory state the engine would
    /// have it in for `phase`, for parity and speed measurements: the
    /// governor is opened and told a request or a reply starts, and for a
    /// reply the layer states are planned. Without it the astro expert cache has no
    /// capacity and nothing a probe observes matches serving.
    #[doc(hidden)]
    pub fn enter_memory_phase_for_tests(
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
        match phase {
            MemoryPhase::GenerationPreparation | MemoryPhase::Decode => {
                self.plan_expert_layer_states_for_reply(performance_attribution)
            }
            MemoryPhase::Prefill | MemoryPhase::Idle => {
                self.forget_expert_layer_states();
                Ok(())
            }
        }
    }

    /// What the engine tells the memory governor before a phase: a model
    /// loaded directly has no governor, no probe and so no capacity.
    fn govern_memory_as_the_engine_would_for_tests(
        &self,
        phase: crate::MemoryPhase,
        context_token_count: u64,
        operation_token_count: u64,
    ) {
        if self.governed_memory.governor.borrow().is_none() {
            self.probe_expert_layer_mlx_working_buffers(super::LayerProbeShape {
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

    /// Drops the reply's plan: a new request reads its prompt with every
    /// layer free to be kept whole.
    pub fn forget_expert_layer_states(&self) {
        *self.active_expert_residency_plan.borrow_mut() = None;
    }

    /// Plans where each layer should end up for the reply, against the
    /// capacity the memory governor has given the astro expert cache. No storage
    /// is read here and nothing is given back.
    pub fn plan_expert_layer_states_for_reply(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let (Some(expert_pager), Some(astro_expert_cache)) =
            (self.expert_pager.as_ref(), self.astro_expert_cache.as_ref())
        else {
            self.forget_expert_layer_states();
            return Ok(());
        };
        let kept_expert_capacity_bytes = self.kept_expert_capacity_bytes();
        let experts_per_token =
            usize::try_from(self.config.experts_per_token()).unwrap_or(usize::MAX);
        let layer_geometries =
            retained_expert_layer_geometries(expert_pager.layer_plans(), experts_per_token)?;
        let expert_capacity = layer_geometries
            .first()
            .map_or(0, |geometry| geometry.expert_capacity);
        let current_residencies = astro_expert_cache
            .borrow()
            .topology_snapshot(expert_capacity);
        let residency_plan = performance_attribution
            .measure_operation(
                PerformanceOperation::ExpertResidencyPlanning,
                |_performance_attribution| {
                    plan_expert_residency(
                        kept_expert_capacity_bytes,
                        &layer_geometries,
                        &current_residencies,
                    )
                },
            )
            .map_err(|plan_error| ExpertPagingError::InvalidPagingPlan {
                description: plan_error.to_string(),
            })?;
        record_reply_plan_counters(
            performance_attribution,
            &residency_plan,
            &current_residencies,
        );
        tracing::info!(
            kept_expert_capacity_bytes,
            kept_layer_count = residency_plan.complete_layer_targets.len(),
            held_whole_layer_count = seated_complete_layer_count(&current_residencies),
            hot_layer_count = count_layers_in_state(&residency_plan, ExpertLayerState::Hot),
            absent_layer_count = count_layers_in_state(&residency_plan, ExpertLayerState::Absent),
            "planned the expert layer states for the reply"
        );
        *self.active_expert_residency_plan.borrow_mut() = Some(residency_plan);
        Ok(())
    }

    /// Where `layer_index` should end up. While no plan is active, which is
    /// while a prompt is read, every layer may be kept whole and the `astro_expert_cache`
    /// capacity is the only refusal.
    pub(super) fn expert_layer_state(&self, layer_index: usize) -> ExpertLayerState {
        self.active_expert_residency_plan
            .borrow()
            .as_ref()
            .and_then(|residency_plan| residency_plan.layer_states.get(layer_index).copied())
            .unwrap_or(ExpertLayerState::Kept)
    }

    /// What the astro expert cache holds of each layer now, for decisions that compare a
    /// layer's state with it.
    pub fn current_expert_layer_residencies(&self) -> Vec<CurrentExpertLayerResidency> {
        let (Some(expert_pager), Some(astro_expert_cache)) =
            (self.expert_pager.as_ref(), self.astro_expert_cache.as_ref())
        else {
            return Vec::new();
        };
        let expert_capacity = expert_pager
            .layer_plans()
            .first()
            .map_or(0, |layer_plan| layer_plan.expert_capacity);
        astro_expert_cache
            .borrow()
            .topology_snapshot(expert_capacity)
    }
}

/// How many layers the plan puts in `layer_state`.
fn count_layers_in_state(
    residency_plan: &ExpertResidencyPlan,
    layer_state: ExpertLayerState,
) -> usize {
    let mut layer_count = 0;
    for planned_layer_state in &residency_plan.layer_states {
        if *planned_layer_state == layer_state {
            layer_count += 1;
        }
    }
    layer_count
}

/// Records the reply plan's telemetry: how many layers are kept whole, hot
/// and absent, and how much of what the astro expert cache held the plan leaves in place.
fn record_reply_plan_counters(
    performance_attribution: &mut PerformanceAttribution,
    residency_plan: &ExpertResidencyPlan,
    current_residencies: &[CurrentExpertLayerResidency],
) {
    performance_attribution.record_counter(
        PerformanceCounter::ExpertResidencyPlanCompleteLayerCount,
        u64::try_from(residency_plan.complete_layer_targets.len()).unwrap_or(u64::MAX),
    );
    performance_attribution.record_counter(
        PerformanceCounter::ExpertResidencyPlanPartialLayerCount,
        u64::try_from(count_layers_in_state(residency_plan, ExpertLayerState::Hot))
            .unwrap_or(u64::MAX),
    );
    performance_attribution.record_counter(
        PerformanceCounter::ExpertResidencyPlanStreamedLayerCount,
        u64::try_from(count_layers_in_state(
            residency_plan,
            ExpertLayerState::Absent,
        ))
        .unwrap_or(u64::MAX),
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
        let stays_in_place = residency_plan
            .layer_states
            .get(current_residency.layer_index)
            .is_some_and(|layer_state| *layer_state != ExpertLayerState::Absent);
        if stays_in_place {
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
