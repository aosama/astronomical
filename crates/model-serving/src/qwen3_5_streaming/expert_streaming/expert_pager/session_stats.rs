//! Pager-owned observatory session statistics: recording and snapshots.
//!
//! All borrow discipline for `session_counters` and `session_read_metrics`
//! lives here so the streaming paths call one-line recorders and the
//! finalization path reads one snapshot. The counters are always-on; see
//! `memory::expert_cache::streaming_stats` for why they must not gate on
//! diagnostics attribution.

use std::sync::Arc;

use astronomical_runtime_integration::PositionalFileReadMetrics;

use crate::inference_engine::contract::ExpertLayerTopology;
use crate::memory::{
    ExpertStreamingSessionCounters, RouteCoverageClass, SessionStreamingPhase,
    expert_layer_topology_from_residency,
};

use super::Qwen3_5ExpertPager;

impl Qwen3_5ExpertPager {
    /// Records one decode token's warm-route coverage classification.
    pub(crate) fn record_session_route_coverage(
        &self,
        retained_assignment_count: u64,
        storage_assignment_count: u64,
        coverage_class: RouteCoverageClass,
    ) {
        self.session_counters.borrow_mut().record_route_coverage(
            retained_assignment_count,
            storage_assignment_count,
            coverage_class,
        );
    }

    /// Records payload bytes a complete retained layer let serving avoid
    /// reading from storage.
    pub(crate) fn record_session_avoided_source_bytes(&self, payload_bytes: u64) {
        self.session_counters
            .borrow_mut()
            .record_avoided_complete_layer_source_bytes(payload_bytes);
    }

    /// Records payload bytes one phase genuinely streamed from storage.
    pub(crate) fn record_session_mandatory_source_bytes(
        &self,
        phase: SessionStreamingPhase,
        payload_bytes: u64,
    ) {
        self.session_counters
            .borrow_mut()
            .record_mandatory_source_bytes(phase, payload_bytes);
    }

    /// Returns the always-on positional-read aggregate so expert-source
    /// readers stay instrumented even when diagnostics attribution is off.
    /// When attribution is enabled its private per-request aggregate keeps
    /// report deltas exact, and this session aggregate still accumulates.
    pub(crate) fn session_expert_read_metrics(&self) -> Arc<PositionalFileReadMetrics> {
        Arc::clone(&self.session_read_metrics)
    }

    /// Pure: session counters plus read totals, without topology.
    pub(crate) fn session_counter_snapshot(&self) -> ExpertStreamingSessionCounters {
        self.session_counters.borrow().clone()
    }

    /// Pure: the session read totals as plain numbers.
    pub(crate) fn session_read_totals(&self) -> (u64, u64, u64) {
        let read_metrics_snapshot = self.session_read_metrics.snapshot();
        (
            read_metrics_snapshot.read_call_count,
            read_metrics_snapshot.read_byte_count,
            read_metrics_snapshot.total_read_elapsed_nanoseconds,
        )
    }

    /// Maps warm-cache layer residencies onto observatory topology rows.
    pub(crate) fn session_layer_topology(
        &self,
        layer_residencies: &[astronomical_memory_policy::residency::CurrentExpertLayerResidency],
    ) -> Vec<ExpertLayerTopology> {
        layer_residencies
            .iter()
            .map(expert_layer_topology_from_residency)
            .collect()
    }
}
