//! Expert demand evidence for layers whose route never reaches the host.
//!
//! A seated complete layer multiplies its table on the graphics processor with
//! the route remapped on-device, so the forward never copies router indices to
//! the host for it. Demand learning still needs those routes: the residency
//! planner ranks retained layers by the demand they cover, and usage history
//! persists it. Copying each layer's route eagerly would reinstate the
//! per-layer synchronization this path exists to remove (40 waits per decode
//! token on a 40-layer model), so the route array is retained lazily here,
//! joins the forward's single completion evaluation as an extra root, and is
//! recorded afterwards as a plain host copy of already-computed identifiers.

use mlx_bridge::MlxArray;

use crate::PerformanceAttribution;
use crate::qwen3_5_streaming::model::Qwen3_5Model;

/// One seated layer's lazily retained route awaiting demand recording. When
/// `resident_layer_payload_byte_count` is set, the route was served entirely
/// from a seated complete layer and the honest read with no `astro_expert_cache` that it avoided is
/// still owed to both byte ledgers once the distinct routed union is known.
#[derive(Debug)]
struct PendingLayerRoute {
    layer_index: usize,
    expert_capacity: usize,
    selected_indices: MlxArray,
    resident_layer_payload_byte_count: Option<u64>,
    /// The prompt-demand weight in force when the route was built, which
    /// may have moved on by the time the route is recorded.
    prefill_demand_assignment_weight: u64,
}

/// Model-owned routes of on-device seated layers for the forward in flight.
#[derive(Debug, Default)]
pub struct DeferredExpertDemandCollector {
    pending_layer_routes: Vec<PendingLayerRoute>,
}

impl DeferredExpertDemandCollector {
    /// Retains one layer's route without evaluating it. A failed retain leaves
    /// that layer's demand unrecorded instead of failing the forward, because
    /// demand is planning evidence, never part of the generated output.
    pub fn retain_layer_route(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_indices: &MlxArray,
        prefill_demand_assignment_weight: u64,
    ) {
        self.retain_route_with_baseline(
            layer_index,
            expert_capacity,
            selected_indices,
            None,
            prefill_demand_assignment_weight,
        );
    }

    /// Retains one route that a seated complete layer served wholly from RAM,
    /// so its avoided bytes can be credited after the completion evaluation.
    pub fn retain_resident_layer_route(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_indices: &MlxArray,
        resident_layer_payload_byte_count: u64,
        prefill_demand_assignment_weight: u64,
    ) {
        self.retain_route_with_baseline(
            layer_index,
            expert_capacity,
            selected_indices,
            Some(resident_layer_payload_byte_count),
            prefill_demand_assignment_weight,
        );
    }

    fn retain_route_with_baseline(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_indices: &MlxArray,
        resident_layer_payload_byte_count: Option<u64>,
        prefill_demand_assignment_weight: u64,
    ) {
        match selected_indices.retain() {
            Ok(selected_indices) => self.pending_layer_routes.push(PendingLayerRoute {
                layer_index,
                expert_capacity,
                selected_indices,
                resident_layer_payload_byte_count,
                prefill_demand_assignment_weight,
            }),
            Err(retain_error) => {
                tracing::warn!(
                    layer_index,
                    error = %retain_error,
                    "deferred expert demand could not retain the selected indices; layer demand stays unrecorded"
                );
            }
        }
    }

    /// Borrows pending route arrays so they can join a completion evaluation.
    pub fn pending_route_array_refs(&self) -> impl Iterator<Item = &MlxArray> {
        self.pending_layer_routes
            .iter()
            .map(|pending_layer_route| &pending_layer_route.selected_indices)
    }

    #[cfg(test)]
    pub fn pending_layer_count(&self) -> usize {
        self.pending_layer_routes.len()
    }

    pub fn discard_pending_layer_routes(&mut self) {
        self.pending_layer_routes.clear();
    }

    /// Takes every pending route, leaving the collector empty.
    fn take_pending_layer_routes(&mut self) -> Vec<PendingLayerRoute> {
        std::mem::take(&mut self.pending_layer_routes)
    }
}

impl Qwen3_5Model {
    /// Queues one on-device seated layer's route for demand recording after
    /// the forward's completion evaluation.
    pub(super) fn defer_expert_demand(
        &self,
        layer_index: usize,
        expert_capacity: usize,
        selected_indices: &MlxArray,
    ) {
        let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() else {
            return;
        };
        let prefill_demand_assignment_weight = astro_expert_cache
            .borrow()
            .prefill_demand_assignment_weight();
        self.deferred_expert_demand.borrow_mut().retain_layer_route(
            layer_index,
            expert_capacity,
            selected_indices,
            prefill_demand_assignment_weight,
        );
    }

    /// Queues one route served wholly from a seated complete layer so its
    /// honest avoided bytes are credited after the completion evaluation,
    /// when the chunk's distinct routed union becomes a cheap host read.
    pub(super) fn defer_resident_layer_route(
        &self,
        layer_index: usize,
        expert_capacity: usize,
        selected_indices: &MlxArray,
        resident_layer_payload_byte_count: u64,
    ) {
        let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() else {
            return;
        };
        let prefill_demand_assignment_weight = astro_expert_cache
            .borrow()
            .prefill_demand_assignment_weight();
        self.deferred_expert_demand
            .borrow_mut()
            .retain_resident_layer_route(
                layer_index,
                expert_capacity,
                selected_indices,
                resident_layer_payload_byte_count,
                prefill_demand_assignment_weight,
            );
    }

    /// Records every queued route into the warm table's demand counts and
    /// credits any owed bytes avoided relative to a read with no `astro_expert_cache`.
    ///
    /// Call after the evaluation whose roots included the queued routes; the
    /// copies are then host reads of computed identifiers. Fail-open: a failed
    /// copy drops that layer's demand with a warning.
    pub fn record_deferred_expert_demand(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let pending_layer_routes = self
            .deferred_expert_demand
            .borrow_mut()
            .take_pending_layer_routes();
        let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() else {
            return;
        };
        for pending_layer_route in pending_layer_routes {
            match pending_layer_route.selected_indices.to_vec_u32() {
                Ok(raw_expert_ids) => {
                    let selected_expert_ids: Vec<usize> = raw_expert_ids
                        .into_iter()
                        .map(|expert_id| expert_id as usize)
                        .collect();
                    astro_expert_cache
                        .borrow_mut()
                        .record_expert_demand_at_prefill_weight(
                            pending_layer_route.layer_index,
                            pending_layer_route.expert_capacity,
                            &selected_expert_ids,
                            pending_layer_route.prefill_demand_assignment_weight,
                        );
                    self.record_owed_resident_route_avoided_bytes(
                        &pending_layer_route,
                        &selected_expert_ids,
                        performance_attribution,
                    );
                }
                Err(copy_error) => {
                    tracing::warn!(
                        layer_index = pending_layer_route.layer_index,
                        error = %copy_error,
                        "deferred expert demand could not copy the route; layer demand stays unrecorded"
                    );
                }
            }
        }
    }
    /// Credits the read with no `astro_expert_cache` that a wholly RAM-served resident route avoided,
    /// using the distinct routed experts of the copied route. Multi-token
    /// routes repeat experts across tokens; each expert is one storage read,
    /// so duplicates count once (issue #1715).
    fn record_owed_resident_route_avoided_bytes(
        &self,
        pending_layer_route: &PendingLayerRoute,
        selected_expert_ids: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let Some(resident_layer_payload_byte_count) =
            pending_layer_route.resident_layer_payload_byte_count
        else {
            return;
        };
        let mut distinct_routed_expert_ids = selected_expert_ids.to_vec();
        distinct_routed_expert_ids.sort_unstable();
        distinct_routed_expert_ids.dedup();
        self.record_avoided_resident_route(
            distinct_routed_expert_ids.len(),
            resident_layer_payload_byte_count,
            pending_layer_route.expert_capacity,
            performance_attribution,
        );
    }
}

#[cfg(test)]
#[path = "deferred_expert_demand_tests.rs"]
mod deferred_expert_demand_tests;
