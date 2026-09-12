//! Decode-time capture of true expert routes into the observation history.
//!
//! The capture hook runs inside the sparse mixture-of-experts forward: for a
//! one-token decode it retains the router's selected-indices array lazily and
//! pays no synchronization at all. The engine finalizes one record per token
//! after the forward's logits were already evaluated, so the batched array
//! evaluation there observes arrays the graphics processor already computed.
//!
//! Capture is gated on attribution being enabled: a disabled report performs
//! no retain, no allocation, and no clock read. The chain of previous-token
//! routes lives on the request-owned attribution, so every request starts its
//! prediction history without cross-request state.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::expert_paging::route_observation::{
    ObservedExpertRoute, RouteObservationRecord, RouteObservationRing,
    sorted_unique_layer_routed_expert_ids,
};
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

/// Model-owned decode route capture: the lazy pending arrays for the token
/// being forwarded plus the bounded observation history every request feeds.
/// Layer zero starts a new token, so stale pending arrays from a forward whose
/// record was never finalized are discarded there rather than mixed into the
/// next token's label.
#[derive(Debug)]
pub(crate) struct RouteObservationCollector {
    pending_layer_route_arrays: Vec<Option<MlxArray>>,
    observation_ring: RouteObservationRing,
}

impl RouteObservationCollector {
    pub(crate) fn new() -> Self {
        Self {
            pending_layer_route_arrays: Vec::new(),
            observation_ring: RouteObservationRing::new(
                RouteObservationRing::DEFAULT_OBSERVATION_CAPACITY,
            ),
        }
    }

    pub(crate) fn retain_layer_route(&mut self, layer_index: usize, selected_indices: &MlxArray) {
        if layer_index == 0 {
            self.pending_layer_route_arrays.clear();
        }
        if self.pending_layer_route_arrays.len() <= layer_index {
            self.pending_layer_route_arrays
                .resize_with(layer_index + 1, || None);
        }
        // Retaining bumps the array reference without evaluating anything.
        // A failed retain leaves the layer unobserved instead of failing the
        // forward; capture must never affect the generation request.
        match selected_indices.retain() {
            Ok(retained_indices) => {
                self.pending_layer_route_arrays[layer_index] = Some(retained_indices)
            }
            Err(retain_error) => {
                tracing::warn!(
                    layer_index,
                    error = %retain_error,
                    "route observation could not retain the selected indices; layer stays unobserved"
                );
            }
        }
    }

    /// Takes every pending route array, leaving the collector empty for the
    /// next token.
    fn take_pending_layer_route_arrays(&mut self) -> Vec<Option<MlxArray>> {
        std::mem::take(&mut self.pending_layer_route_arrays)
    }

    pub(crate) fn discard_pending_layer_route_arrays(&mut self) {
        self.pending_layer_route_arrays.clear();
    }

    fn observation_ring(&mut self) -> &mut RouteObservationRing {
        &mut self.observation_ring
    }
}

impl Qwen3_5Model {
    /// Finalizes one route-observation record for a completed decode forward.
    ///
    /// Call this exactly once per decode token, after the token's logits were
    /// evaluated: the retained router arrays are ancestors of those logits, so
    /// the batched evaluation here observes already-computed arrays instead of
    /// scheduling new graphics-processor work.
    ///
    /// Fail-open by contract: every failure logs a bounded warning and drops
    /// the pending capture rather than affecting the generation request.
    pub(crate) fn finalize_route_observation_record(
        &self,
        input_token_id: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let pending_layer_route_arrays = self
            .route_observation
            .borrow_mut()
            .take_pending_layer_route_arrays();
        if pending_layer_route_arrays.is_empty() {
            return;
        }
        let finalized_route = performance_attribution.measure_operation(
            PerformanceOperation::RouteObservationFinalization,
            |_performance_attribution| {
                self.evaluate_pending_route_arrays(&pending_layer_route_arrays)
            },
        );
        let Ok(finalized_route) = finalized_route else {
            tracing::warn!(
                input_token_id,
                "route observation finalization failed; dropping this token's capture"
            );
            return;
        };
        let Some(finalized_route) = finalized_route else {
            return;
        };
        let captured_layer_count = finalized_route
            .iter()
            .filter(|layer| layer.is_some())
            .count();
        let previous_token_route =
            performance_attribution.advance_route_observation_chain(finalized_route.clone());
        let observation = RouteObservationRecord {
            input_token_id,
            previous_token_route,
            token_route: finalized_route,
        };
        let evicted_oldest = {
            let mut collector = self.route_observation.borrow_mut();
            collector
                .observation_ring()
                .record_observation(observation.clone())
        };
        self.offer_observation_to_predictor(&observation, performance_attribution);
        performance_attribution
            .record_counter(PerformanceCounter::RouteObservationStoredRecordCount, 1);
        if evicted_oldest {
            performance_attribution
                .record_counter(PerformanceCounter::RouteObservationEvictedRecordCount, 1);
        }
        performance_attribution.record_counter(
            PerformanceCounter::RouteObservationCapturedLayerCount,
            u64::try_from(captured_layer_count).unwrap_or(u64::MAX),
        );
    }

    /// Evaluates the pending route arrays in one batch and compacts each
    /// layer's identifiers. Returns `Ok(None)` when nothing was observable.
    fn evaluate_pending_route_arrays(
        &self,
        pending_layer_route_arrays: &[Option<MlxArray>],
    ) -> Result<Option<ObservedExpertRoute>, Qwen3_5ExecutionError> {
        let retained_arrays: Vec<&MlxArray> = pending_layer_route_arrays
            .iter()
            .filter_map(|pending_route_array| pending_route_array.as_ref())
            .collect();
        if retained_arrays.is_empty() {
            return Ok(None);
        }
        self.runtime.evaluate_arrays(&retained_arrays)?;
        let mut observed_route: ObservedExpertRoute =
            Vec::with_capacity(pending_layer_route_arrays.len());
        for pending_route_array in pending_layer_route_arrays {
            let layer_route = pending_route_array
                .as_ref()
                .map(|route_array| route_array.copy_evaluated_u32_values())
                .transpose()?
                .and_then(|raw_expert_ids| sorted_unique_layer_routed_expert_ids(&raw_expert_ids));
            observed_route.push(layer_route);
        }
        Ok(Some(observed_route))
    }

    fn offer_observation_to_predictor(
        &self,
        observation: &RouteObservationRecord,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        if !performance_attribution.is_enabled() || self.config.expert_count() == 0 {
            return;
        }
        let mut predictor_owner = self.expert_route_predictor.borrow_mut();
        if predictor_owner.is_none() {
            *predictor_owner =
                crate::qwen3_5_moe::expert_paging::predictor::ExpertRoutePredictorOwner::try_start(
                    usize::try_from(self.config.layer_count()).unwrap_or(0),
                    usize::try_from(self.config.expert_count()).unwrap_or(0),
                    self.config.vocabulary_size(),
                    usize::try_from(self.config.experts_per_token()).unwrap_or(1),
                );
        }
        let Some(predictor_owner) = predictor_owner.as_ref() else {
            return;
        };
        predictor_owner.try_submit(observation.clone());
        performance_attribution.record_snapshot_counter(
            PerformanceCounter::ExpertRoutePredictorTrainStepCount,
            predictor_owner.trained_record_count(),
        );
        performance_attribution.record_snapshot_counter(
            PerformanceCounter::ExpertRoutePredictorTopKHitCount,
            predictor_owner.top_k_hit_count(),
        );
        performance_attribution.record_snapshot_counter(
            PerformanceCounter::ExpertRoutePredictorEvaluatedExpertCount,
            predictor_owner.evaluated_expert_count(),
        );
        performance_attribution.record_snapshot_counter(
            PerformanceCounter::ExpertRoutePredictorTrainSliceNanoseconds,
            predictor_owner.last_slice_nanoseconds(),
        );
    }

    /// Ranks leftover experts the next forward is predicted to need and asks
    /// the warm table not to evict them. Fail-open: lock contention or a missing
    /// trainer leaves the existing eviction order unchanged.
    pub(crate) fn refresh_predictor_retention_hints(
        &self,
        next_token_id: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        if !performance_attribution.is_enabled() {
            return;
        }
        let predicted_experts_by_layer = {
            let predictor_owner_slot = self.expert_route_predictor.borrow();
            let Some(predictor_owner) = predictor_owner_slot.as_ref() else {
                return;
            };
            let previous_route = performance_attribution.previous_observed_expert_route();
            let top_k = usize::try_from(self.config.experts_per_token()).unwrap_or(1);
            predictor_owner.try_predict_top_experts_per_layer(
                next_token_id,
                previous_route.map(Vec::as_slice),
                top_k,
            )
        };
        let Some(predicted_experts_by_layer) = predicted_experts_by_layer else {
            return;
        };
        if let Some(retained_experts) = self.retained_experts.as_ref() {
            retained_experts
                .borrow_mut()
                .set_predicted_retention_experts(&predicted_experts_by_layer);
        }
    }
}
