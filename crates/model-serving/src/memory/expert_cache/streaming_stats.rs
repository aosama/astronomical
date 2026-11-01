//! Session-scoped expert-streaming counters, always-on by design.
//!
//! The diagnostics attribution (`performance_attribution`) is opt-in because
//! its timing work costs real serving latency, and that is the right trade
//! for deep investigation. The observatory expert page answers a different
//! question — "what did expert streaming do for me lately?" — and must never
//! show a hollow panel because diagnostics are off. These counters are
//! therefore plain saturating increments recorded at the same central sites
//! the attribution reads: no clocks, no allocation, no gates.
//!
//! Ownership: one accumulator per engine load, held by the streaming model,
//! so the values describe the whole serving session of that model instance.

use crate::inference_engine::contract::ExpertLayerTopology;

/// Saturating session counters for one engine load's expert streaming.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ExpertStreamingSessionCounters {
    pub(crate) retained_route_assignment_count: u64,
    pub(crate) storage_route_assignment_count: u64,
    pub(crate) hot_route_fully_covered_token_count: u64,
    pub(crate) hot_route_partially_covered_token_count: u64,
    pub(crate) hot_route_missed_token_count: u64,
    pub(crate) avoided_complete_layer_source_bytes: u64,
    pub(crate) mandatory_prefill_source_bytes: u64,
    pub(crate) mandatory_decode_source_bytes: u64,
}

impl ExpertStreamingSessionCounters {
    /// Records one decode token's warm-route coverage classification.
    pub(crate) fn record_route_coverage(
        &mut self,
        retained_assignment_count: u64,
        storage_assignment_count: u64,
        coverage_class: RouteCoverageClass,
    ) {
        self.retained_route_assignment_count = self
            .retained_route_assignment_count
            .saturating_add(retained_assignment_count);
        self.storage_route_assignment_count = self
            .storage_route_assignment_count
            .saturating_add(storage_assignment_count);
        match coverage_class {
            RouteCoverageClass::FullyCovered => {
                self.hot_route_fully_covered_token_count =
                    self.hot_route_fully_covered_token_count.saturating_add(1);
            }
            RouteCoverageClass::PartiallyCovered => {
                self.hot_route_partially_covered_token_count = self
                    .hot_route_partially_covered_token_count
                    .saturating_add(1);
            }
            RouteCoverageClass::Missed => {
                self.hot_route_missed_token_count =
                    self.hot_route_missed_token_count.saturating_add(1);
            }
        }
    }

    /// Records payload bytes an already-complete layer let serving skip
    /// reading from storage.
    pub(crate) fn record_avoided_complete_layer_source_bytes(&mut self, payload_bytes: u64) {
        self.avoided_complete_layer_source_bytes = self
            .avoided_complete_layer_source_bytes
            .saturating_add(payload_bytes);
    }

    /// Records payload bytes a phase genuinely had to stream from storage.
    /// The phase split mirrors the attribution catalog: prefill breadth and
    /// decode depth answer different questions (issue #1693).
    pub(crate) fn record_mandatory_source_bytes(
        &mut self,
        phase: SessionStreamingPhase,
        payload_bytes: u64,
    ) {
        match phase {
            SessionStreamingPhase::Prefill => {
                self.mandatory_prefill_source_bytes = self
                    .mandatory_prefill_source_bytes
                    .saturating_add(payload_bytes);
            }
            SessionStreamingPhase::Decode => {
                self.mandatory_decode_source_bytes = self
                    .mandatory_decode_source_bytes
                    .saturating_add(payload_bytes);
            }
        }
    }
}

/// One decode token's warm-route coverage outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouteCoverageClass {
    FullyCovered,
    PartiallyCovered,
    Missed,
}

/// Which serving phase a storage stream served.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SessionStreamingPhase {
    Prefill,
    Decode,
}

/// Pure mapping of one warm-cache layer residency onto the observatory DTO.
#[must_use]
pub(crate) fn expert_layer_topology_from_residency(
    layer_residency: &astronomical_memory_policy::residency::CurrentExpertLayerResidency,
) -> ExpertLayerTopology {
    ExpertLayerTopology {
        layer_index: u32::try_from(layer_residency.layer_index).unwrap_or(u32::MAX),
        retained_expert_ids: layer_residency
            .retained_expert_ids
            .iter()
            .map(|expert_id| u32::try_from(*expert_id).unwrap_or(u32::MAX))
            .collect(),
        retained_payload_bytes: layer_residency.payload_bytes,
        is_complete_layer: matches!(
            layer_residency.class,
            astronomical_memory_policy::residency::RetainedExpertPageClass::StableCompleteLayer
        ),
        covered_demand: layer_residency.covered_weighted_demand,
    }
}
