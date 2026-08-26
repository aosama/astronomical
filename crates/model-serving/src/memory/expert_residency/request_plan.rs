//! One request-stable answer to pin, stream, and shrink.
//!
//! Leftover RAM arithmetic may be recomputed on every chunk. This owner freezes
//! the Prefill contract at request open: pinned complete layers stay pinned,
//! streamed layers stay streamed, and a capacity failure may only shrink the
//! pin set. Generation handoff discards the contract and replans from leftover.

use super::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPhase, PhaseAwareExpertResidencyPlan, RetainedExpertPageClass,
};

/// Role of one sparse layer for the rest of this request's Prefill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestExpertLayerRole {
    /// Keep or promote the complete layer until the request ends or shrinks.
    PinnedComplete,
    /// Read when a forward needs it. Never promote during this Prefill.
    Streamed,
}

/// Request-scoped expert contract. Family crates enact it; they do not replan it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestExpertResidency {
    layer_roles: Vec<RequestExpertLayerRole>,
}

impl RequestExpertResidency {
    /// Pins every complete-layer target from the opening leftover plan.
    #[must_use]
    pub fn open_prefill(candidate_plan: &PhaseAwareExpertResidencyPlan) -> Self {
        let mut layer_roles =
            vec![RequestExpertLayerRole::Streamed; candidate_plan.layer_targets.len()];
        for layer_index in &candidate_plan.complete_layer_targets {
            if let Some(layer_role) = layer_roles.get_mut(*layer_index) {
                *layer_role = RequestExpertLayerRole::PinnedComplete;
            }
        }
        Self { layer_roles }
    }

    #[must_use]
    pub fn layer_role(&self, layer_index: usize) -> Option<RequestExpertLayerRole> {
        self.layer_roles.get(layer_index).copied()
    }

    #[must_use]
    pub fn pinned_complete_layer_indexes(&self) -> Vec<usize> {
        self.layer_roles
            .iter()
            .enumerate()
            .filter_map(|(layer_index, layer_role)| {
                matches!(layer_role, RequestExpertLayerRole::PinnedComplete).then_some(layer_index)
            })
            .collect()
    }

    /// Drops pinned complete layers from the tail until `required_reclamation_bytes` is covered.
    ///
    /// Those layers become streamed for the rest of Prefill. A later leftover
    /// plan must not pin them again.
    #[must_use]
    pub fn shrink_after_capacity_failure(
        &self,
        required_reclamation_bytes: u64,
        layer_geometries: &[ExpertLayerGeometry],
    ) -> Self {
        if required_reclamation_bytes == 0 {
            return self.clone();
        }
        let mut layer_roles = self.layer_roles.clone();
        let mut remaining_reclamation_bytes = required_reclamation_bytes;
        for layer_index in (0..layer_roles.len()).rev() {
            if remaining_reclamation_bytes == 0 {
                break;
            }
            if layer_roles[layer_index] != RequestExpertLayerRole::PinnedComplete {
                continue;
            }
            let complete_layer_payload_bytes = layer_geometries
                .get(layer_index)
                .map(|geometry| geometry.complete_layer_payload_bytes)
                .unwrap_or(0);
            layer_roles[layer_index] = RequestExpertLayerRole::Streamed;
            remaining_reclamation_bytes =
                remaining_reclamation_bytes.saturating_sub(complete_layer_payload_bytes);
        }
        Self { layer_roles }
    }

    /// Rewrites a leftover Prefill plan so it cannot promote or release against this contract.
    #[must_use]
    pub fn stabilize_prefill_plan(
        &self,
        mut candidate_plan: PhaseAwareExpertResidencyPlan,
        current_residencies: &[CurrentExpertLayerResidency],
    ) -> PhaseAwareExpertResidencyPlan {
        let mut layer_is_currently_complete = vec![false; self.layer_roles.len()];
        for current_residency in current_residencies {
            if current_residency.layer_index < layer_is_currently_complete.len()
                && current_residency.class == RetainedExpertPageClass::StableCompleteLayer
            {
                layer_is_currently_complete[current_residency.layer_index] = true;
            }
        }
        if candidate_plan.layer_targets.len() != self.layer_roles.len() {
            candidate_plan.layer_targets.resize(
                self.layer_roles.len(),
                ExpertLayerResidencyTarget::StreamOperationLocal,
            );
        }
        let mut expected_preserved_bytes = 0_u64;
        let mut maximum_new_retained_bytes = 0_u64;
        for (layer_index, layer_role) in self.layer_roles.iter().enumerate() {
            let layer_target = match *layer_role {
                RequestExpertLayerRole::PinnedComplete
                    if layer_is_currently_complete[layer_index] =>
                {
                    if let Some(current_residency) = current_residencies
                        .iter()
                        .find(|residency| residency.layer_index == layer_index)
                    {
                        expected_preserved_bytes = expected_preserved_bytes
                            .saturating_add(current_residency.payload_bytes);
                    }
                    ExpertLayerResidencyTarget::PreserveComplete
                }
                RequestExpertLayerRole::PinnedComplete => {
                    maximum_new_retained_bytes = maximum_new_retained_bytes.saturating_add(
                        current_residencies
                            .iter()
                            .find(|residency| residency.layer_index == layer_index)
                            .map(|residency| residency.payload_bytes)
                            .unwrap_or(0),
                    );
                    ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
                }
                RequestExpertLayerRole::Streamed => {
                    ExpertLayerResidencyTarget::StreamOperationLocal
                }
            };
            candidate_plan.layer_targets[layer_index] = layer_target;
        }
        candidate_plan.complete_layer_targets = self.pinned_complete_layer_indexes();
        candidate_plan.reserved_routed_overlay_bytes = 0;
        candidate_plan.expected_preserved_bytes = expected_preserved_bytes;
        candidate_plan.maximum_new_retained_bytes = maximum_new_retained_bytes;
        candidate_plan.is_low_budget_partial_mode = false;
        candidate_plan
    }
}

/// Floors the Prefill retained-page ceiling at complete layers already in RAM.
///
/// Leftover arithmetic can tighten after a chunk because learned context reserve
/// grew. Evicting a seated complete layer to match that smaller number throws away
/// a page this request already paid to read, then every later chunk streams it
/// again. Real capacity failure still shrinks through `shrink_after_capacity_failure`.
#[must_use]
pub const fn retained_complete_layer_ceiling_after_prefill_budget_refresh(
    leftover_expert_budget_bytes: u64,
    current_complete_layer_payload_bytes: u64,
) -> u64 {
    if leftover_expert_budget_bytes > current_complete_layer_payload_bytes {
        leftover_expert_budget_bytes
    } else {
        current_complete_layer_payload_bytes
    }
}

/// Binds leftover packing to the request contract for the current phase.
///
/// Prefill opens once, then only shrinks. Generation, decode, and idle discard
/// the Prefill contract and keep the leftover candidate unchanged.
#[must_use]
pub fn publish_request_stable_residency_plan(
    phase: ExpertResidencyPhase,
    existing_request_residency: Option<&RequestExpertResidency>,
    candidate_plan: PhaseAwareExpertResidencyPlan,
    current_residencies: &[CurrentExpertLayerResidency],
    released_complete_payload_bytes: u64,
    layer_geometries: &[ExpertLayerGeometry],
) -> (
    Option<RequestExpertResidency>,
    PhaseAwareExpertResidencyPlan,
) {
    match phase {
        ExpertResidencyPhase::Prefill => {
            let mut request_residency = existing_request_residency
                .cloned()
                .unwrap_or_else(|| RequestExpertResidency::open_prefill(&candidate_plan));
            if existing_request_residency.is_some() && released_complete_payload_bytes > 0 {
                request_residency = request_residency.shrink_after_capacity_failure(
                    released_complete_payload_bytes,
                    layer_geometries,
                );
            }
            let stabilized_plan =
                request_residency.stabilize_prefill_plan(candidate_plan, current_residencies);
            (Some(request_residency), stabilized_plan)
        }
        ExpertResidencyPhase::GenerationPreparation
        | ExpertResidencyPhase::Decode
        | ExpertResidencyPhase::Idle => (None, candidate_plan),
    }
}
