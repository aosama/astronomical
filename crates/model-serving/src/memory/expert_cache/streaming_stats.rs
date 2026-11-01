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
use crate::memory::serving_phase::SessionStreamingPhase;
use crate::performance_attribution::{PerformanceAttribution, StreamedPageSourcePlan};

/// Records one streamed page's source plan in both ledgers at once: the
/// opt-in per-request attribution counters and the always-on observatory
/// session counters. Splitting these recordings let the same bytes land in
/// different phase buckets (a session once disagreed with its attribution
/// reports by tens of gigabytes of prefill); one seam keeps them equal by
/// construction (issue #1715).
pub(crate) fn record_streamed_page_source_plan_in_both_ledgers(
    streamed_page_source_plan: &StreamedPageSourcePlan,
    performance_attribution: &mut PerformanceAttribution,
    session_counters: &mut ExpertStreamingSessionCounters,
) {
    performance_attribution.record_expert_streaming_source_plan(streamed_page_source_plan);
    session_counters.record_mandatory_source_bytes(
        streamed_page_source_plan.serving_phase,
        streamed_page_source_plan.payload_byte_count,
    );
}

/// Saturating session counters for one engine load's expert streaming.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ExpertStreamingSessionCounters {
    pub(crate) retained_route_assignment_count: u64,
    pub(crate) storage_route_assignment_count: u64,
    /// Decode coverage is classified once per paged or resident layer per
    /// decode token, so these count layer-tokens, not tokens (issue #1715).
    pub(crate) hot_route_fully_covered_layer_token_count: u64,
    pub(crate) hot_route_partially_covered_layer_token_count: u64,
    pub(crate) hot_route_missed_layer_token_count: u64,
    pub(crate) avoided_complete_layer_source_bytes: u64,
    pub(crate) mandatory_prefill_source_bytes: u64,
    pub(crate) mandatory_decode_source_bytes: u64,
    /// Positional-read elapsed time attributable to prefill page streams,
    /// kept separate from decode so the decode-only time-saved estimate
    /// never divides prefill seconds by decode miss counts (issue #1715).
    pub(crate) prefill_source_read_elapsed_nanoseconds: u64,
    pub(crate) decode_source_read_elapsed_nanoseconds: u64,
}

impl ExpertStreamingSessionCounters {
    /// Records one decode layer-token's warm-route coverage classification.
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
                self.hot_route_fully_covered_layer_token_count = self
                    .hot_route_fully_covered_layer_token_count
                    .saturating_add(1);
            }
            RouteCoverageClass::PartiallyCovered => {
                self.hot_route_partially_covered_layer_token_count = self
                    .hot_route_partially_covered_layer_token_count
                    .saturating_add(1);
            }
            RouteCoverageClass::Missed => {
                self.hot_route_missed_layer_token_count =
                    self.hot_route_missed_layer_token_count.saturating_add(1);
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

    /// Records positional-read elapsed time observed while one phase streamed
    /// pages, so each phase's read speed can be derived from its own seconds.
    pub(crate) fn record_phase_source_read_elapsed(
        &mut self,
        phase: SessionStreamingPhase,
        elapsed_nanoseconds: u64,
    ) {
        match phase {
            SessionStreamingPhase::Prefill => {
                self.prefill_source_read_elapsed_nanoseconds = self
                    .prefill_source_read_elapsed_nanoseconds
                    .saturating_add(elapsed_nanoseconds);
            }
            SessionStreamingPhase::Decode => {
                self.decode_source_read_elapsed_nanoseconds = self
                    .decode_source_read_elapsed_nanoseconds
                    .saturating_add(elapsed_nanoseconds);
            }
        }
    }
}

/// Pure: the storage bytes a server without any expert cache would have read
/// to serve one route from a seated complete layer.
///
/// The honest counterfactual is the payload of the *distinct* routed experts
/// — the decode token's routed set, or the chunk's routed union — never the
/// whole layer payload: a no-cache server streams exactly the routed union
/// (`stream_operation_local_routed_experts`), so a route touching every
/// expert of the layer equals exactly one layer payload and no route can
/// exceed it (issue #1715).
#[must_use]
pub(crate) fn no_cache_route_source_bytes(
    complete_layer_payload_bytes: u64,
    expert_capacity: usize,
    distinct_routed_expert_count: usize,
) -> u64 {
    if expert_capacity == 0 {
        return 0;
    }
    let expert_capacity_u64 = u64::try_from(expert_capacity).unwrap_or(u64::MAX);
    // Quantized pages price every expert of a layer identically (see
    // `source_manifests`: `selected_expert_ids.len() * bytes_per_expert`), so
    // the per-expert share is exact arithmetic, not an approximation.
    let bytes_per_expert = complete_layer_payload_bytes / expert_capacity_u64;
    let routed_expert_count = u64::try_from(distinct_routed_expert_count).unwrap_or(u64::MAX);
    bytes_per_expert
        .saturating_mul(routed_expert_count)
        .min(complete_layer_payload_bytes)
}

/// One decode token's warm-route coverage outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouteCoverageClass {
    FullyCovered,
    PartiallyCovered,
    Missed,
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
