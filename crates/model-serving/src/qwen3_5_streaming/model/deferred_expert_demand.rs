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

use crate::qwen3_5_streaming::model::Qwen3_5Model;

/// One seated layer's lazily retained route awaiting demand recording.
#[derive(Debug)]
struct PendingLayerRoute {
    layer_index: usize,
    expert_capacity: usize,
    selected_indices: MlxArray,
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
        match selected_indices.retain() {
            Ok(selected_indices) => self.pending_layer_routes.push(PendingLayerRoute {
                layer_index,
                expert_capacity,
                selected_indices,
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

    /// Records every queued route into the warm table's demand counts.
    ///
    /// Call after the evaluation whose roots included the queued routes; the
    /// copies are then host reads of computed identifiers. Fail-open: a failed
    /// copy drops that layer's demand with a warning.
    pub(crate) fn record_deferred_expert_demand(&self) {
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
}

#[cfg(test)]
#[path = "deferred_expert_demand_tests.rs"]
mod deferred_expert_demand_tests;
