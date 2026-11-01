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

use mlx_c_rust::MlxArray;

use crate::PerformanceAttribution;
use crate::qwen3_5_streaming::model::Qwen3_5Model;

/// One seated layer's lazily retained route awaiting demand recording. When
/// `resident_layer_payload_byte_count` is set, the route was served entirely
/// from a seated complete layer and the honest no-cache read it avoided is
/// still owed to both byte ledgers once the distinct routed union is known.
#[derive(Debug)]
struct PendingLayerRoute {
    layer_index: usize,
    expert_capacity: usize,
    selected_indices: MlxArray,
    resident_layer_payload_byte_count: Option<u64>,
}

/// Model-owned routes of on-device seated layers for the forward in flight.
#[derive(Debug, Default)]
pub(crate) struct DeferredExpertDemandCollector {
    pending_layer_routes: Vec<PendingLayerRoute>,
}

impl DeferredExpertDemandCollector {
    /// Retains one layer's route without evaluating it. A failed retain leaves
    /// that layer's demand unrecorded instead of failing the forward, because
    /// demand is planning evidence, never part of the generated output.
    pub(crate) fn retain_layer_route(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_indices: &MlxArray,
    ) {
        self.retain_route_with_baseline(layer_index, expert_capacity, selected_indices, None);
    }

    /// Retains one route that a seated complete layer served wholly from RAM,
    /// so its avoided bytes can be credited after the completion evaluation.
    pub(crate) fn retain_resident_layer_route(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_indices: &MlxArray,
        resident_layer_payload_byte_count: u64,
    ) {
        self.retain_route_with_baseline(
            layer_index,
            expert_capacity,
            selected_indices,
            Some(resident_layer_payload_byte_count),
        );
    }

    fn retain_route_with_baseline(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_indices: &MlxArray,
        resident_layer_payload_byte_count: Option<u64>,
    ) {
        match selected_indices.retain() {
            Ok(selected_indices) => self.pending_layer_routes.push(PendingLayerRoute {
                layer_index,
                expert_capacity,
                selected_indices,
                resident_layer_payload_byte_count,
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
    pub(crate) fn pending_route_array_refs(&self) -> impl Iterator<Item = &MlxArray> {
        self.pending_layer_routes
            .iter()
            .map(|pending_layer_route| &pending_layer_route.selected_indices)
    }

    #[cfg(test)]
    pub(crate) fn pending_layer_count(&self) -> usize {
        self.pending_layer_routes.len()
    }

    pub(crate) fn discard_pending_layer_routes(&mut self) {
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
        if self.retained_experts.is_none() {
            return;
        }
        self.deferred_expert_demand.borrow_mut().retain_layer_route(
            layer_index,
            expert_capacity,
            selected_indices,
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
        if self.retained_experts.is_none() {
            return;
        }
        self.deferred_expert_demand
            .borrow_mut()
            .retain_resident_layer_route(
                layer_index,
                expert_capacity,
                selected_indices,
                resident_layer_payload_byte_count,
            );
    }

    /// Records every queued route into the warm table's demand counts and
    /// credits any owed no-cache avoided bytes.
    ///
    /// Call after the evaluation whose roots included the queued routes; the
    /// copies are then host reads of computed identifiers. Fail-open: a failed
    /// copy drops that layer's demand with a warning.
    pub(crate) fn record_deferred_expert_demand(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let pending_layer_routes = self
            .deferred_expert_demand
            .borrow_mut()
            .take_pending_layer_routes();
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return;
        };
        for pending_layer_route in pending_layer_routes {
            match pending_layer_route.selected_indices.to_vec_u32() {
                Ok(raw_expert_ids) => {
                    let selected_expert_ids: Vec<usize> = raw_expert_ids
                        .into_iter()
                        .map(|expert_id| expert_id as usize)
                        .collect();
                    retained_experts.borrow_mut().record_expert_demand(
                        pending_layer_route.layer_index,
                        pending_layer_route.expert_capacity,
                        &selected_expert_ids,
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
    /// Credits the no-cache read a wholly RAM-served resident route avoided,
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
