//! Bounded request evidence for expert source plans issued by SSD streaming.
//!
//! Aggregate byte counters show total storage demand but cannot reveal whether
//! one nonresident layer was reread for every prompt chunk. One summary per phase
//! and layer preserves that evidence without allocating per token-layer pass.

use serde::Serialize;

use crate::memory::SessionStreamingPhase;

use super::{PerformanceAttribution, PerformanceCounter};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ExpertStreamingPhase {
    Prefill,
    Decode,
}

/// One streamed expert page's source plan: the shared record both accounting
/// ledgers consume, so neither ledger can re-derive (and mis-derive) a value
/// the other already holds (issue #1715). Defined beside the attribution
/// summaries it feeds, and consumed unchanged by the observatory session
/// counters' shared recorder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamedPageSourcePlan {
    pub(crate) layer_index: usize,
    pub(crate) route_token_count: i32,
    /// The serving phase the engine was actually in when the page streamed;
    /// never inferred from the route's token count.
    pub(crate) serving_phase: SessionStreamingPhase,
    pub(crate) routed_expert_count: usize,
    pub(crate) streamed_expert_count: usize,
    pub(crate) source_shard_count: usize,
    pub(crate) payload_byte_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ExpertStreamingSourceSummary {
    phase: ExpertStreamingPhase,
    layer_index: usize,
    source_plan_count: u64,
    total_route_token_count: u64,
    total_routed_expert_count: u64,
    total_streamed_expert_count: u64,
    total_source_shard_count: u64,
    payload_byte_count: u64,
}

impl ExpertStreamingSourceSummary {
    fn record_source_plan(
        &mut self,
        route_token_count: u64,
        routed_expert_count: u64,
        streamed_expert_count: u64,
        source_shard_count: u64,
        payload_byte_count: u64,
    ) {
        self.source_plan_count = self.source_plan_count.saturating_add(1);
        self.total_route_token_count = self
            .total_route_token_count
            .saturating_add(route_token_count);
        self.total_routed_expert_count = self
            .total_routed_expert_count
            .saturating_add(routed_expert_count);
        self.total_streamed_expert_count = self
            .total_streamed_expert_count
            .saturating_add(streamed_expert_count);
        self.total_source_shard_count = self
            .total_source_shard_count
            .saturating_add(source_shard_count);
        self.payload_byte_count = self.payload_byte_count.saturating_add(payload_byte_count);
    }
}

impl PerformanceAttribution {
    /// Records one successful lazy source plan without retaining routed expert IDs.
    ///
    /// The phase arrives from the engine's own serving-phase state: a
    /// multi-token route is usually prefill but can be a speculative or
    /// multi-token decode, and inferring the phase from the route length
    /// mis-bucketed those reads (issue #1715).
    pub fn record_expert_streaming_source_plan(
        &mut self,
        streamed_page_source_plan: &StreamedPageSourcePlan,
    ) {
        let Some(enabled_attribution) = self.enabled_attribution.as_mut() else {
            return;
        };
        let phase = expert_streaming_phase(streamed_page_source_plan.serving_phase);
        let mandatory_source_payload_counter = mandatory_source_payload_counter(phase);
        enabled_attribution.counter_values[mandatory_source_payload_counter as usize] =
            enabled_attribution.counter_values[mandatory_source_payload_counter as usize]
                .saturating_add(streamed_page_source_plan.payload_byte_count);
        let phase_slot = source_summary_phase_slot(phase);
        let Some(summary_index) = streamed_page_source_plan
            .layer_index
            .checked_mul(2)
            .and_then(|layer_summary_offset| layer_summary_offset.checked_add(phase_slot))
        else {
            return;
        };
        if enabled_attribution.expert_streaming_source_summaries.len() <= summary_index {
            enabled_attribution
                .expert_streaming_source_summaries
                .resize(summary_index + 1, None);
        }
        let route_token_count =
            u64::try_from(streamed_page_source_plan.route_token_count).unwrap_or(0);
        let summary = enabled_attribution.expert_streaming_source_summaries[summary_index]
            .get_or_insert(ExpertStreamingSourceSummary {
                phase,
                layer_index: streamed_page_source_plan.layer_index,
                source_plan_count: 0,
                total_route_token_count: 0,
                total_routed_expert_count: 0,
                total_streamed_expert_count: 0,
                total_source_shard_count: 0,
                payload_byte_count: 0,
            });
        summary.record_source_plan(
            route_token_count,
            usize_to_u64_saturating(streamed_page_source_plan.routed_expert_count),
            usize_to_u64_saturating(streamed_page_source_plan.streamed_expert_count),
            usize_to_u64_saturating(streamed_page_source_plan.source_shard_count),
            streamed_page_source_plan.payload_byte_count,
        );
    }
}

fn usize_to_u64_saturating(integer_count: usize) -> u64 {
    u64::try_from(integer_count).unwrap_or(u64::MAX)
}

/// Pure: the attribution phase for the engine's serving phase.
const fn expert_streaming_phase(serving_phase: SessionStreamingPhase) -> ExpertStreamingPhase {
    match serving_phase {
        SessionStreamingPhase::Prefill => ExpertStreamingPhase::Prefill,
        SessionStreamingPhase::Decode => ExpertStreamingPhase::Decode,
    }
}

/// Pure: the mandatory-source payload counter a phase's streamed bytes land in.
const fn mandatory_source_payload_counter(phase: ExpertStreamingPhase) -> PerformanceCounter {
    match phase {
        ExpertStreamingPhase::Prefill => {
            PerformanceCounter::MandatoryPrefillExpertSourcePayloadBytes
        }
        ExpertStreamingPhase::Decode => PerformanceCounter::MandatoryDecodeExpertSourcePayloadBytes,
    }
}

/// Pure: a phase's slot within one layer's pair of source summaries.
const fn source_summary_phase_slot(phase: ExpertStreamingPhase) -> usize {
    match phase {
        ExpertStreamingPhase::Prefill => 0,
        ExpertStreamingPhase::Decode => 1,
    }
}
