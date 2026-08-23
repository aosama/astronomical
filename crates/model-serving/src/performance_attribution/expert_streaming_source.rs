//! Bounded request evidence for expert source plans issued by SSD streaming.
//!
//! Aggregate byte counters show total storage demand but cannot reveal whether
//! one nonresident layer was reread for every prompt chunk. One summary per phase
//! and layer preserves that evidence without allocating per token-layer pass.

use serde::Serialize;

use super::{PerformanceAttribution, PerformanceCounter};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ExpertStreamingPhase {
    Prefill,
    Decode,
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
    pub fn record_expert_streaming_source_plan(
        &mut self,
        layer_index: usize,
        route_token_count: i32,
        routed_expert_count: usize,
        streamed_expert_count: usize,
        source_shard_count: usize,
        payload_byte_count: u64,
    ) {
        let Some(enabled_attribution) = self.enabled_attribution.as_mut() else {
            return;
        };
        let phase = if route_token_count > 1 {
            ExpertStreamingPhase::Prefill
        } else {
            ExpertStreamingPhase::Decode
        };
        let mandatory_source_payload_counter = match phase {
            ExpertStreamingPhase::Prefill => {
                PerformanceCounter::MandatoryPrefillExpertSourcePayloadBytes
            }
            ExpertStreamingPhase::Decode => {
                PerformanceCounter::MandatoryDecodeExpertSourcePayloadBytes
            }
        };
        enabled_attribution.counter_values[mandatory_source_payload_counter as usize] =
            enabled_attribution.counter_values[mandatory_source_payload_counter as usize]
                .saturating_add(payload_byte_count);
        let phase_slot = match phase {
            ExpertStreamingPhase::Prefill => 0,
            ExpertStreamingPhase::Decode => 1,
        };
        let Some(summary_index) = layer_index
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
        let route_token_count = u64::try_from(route_token_count).unwrap_or(0);
        let routed_expert_count = usize_to_u64_saturating(routed_expert_count);
        let streamed_expert_count = usize_to_u64_saturating(streamed_expert_count);
        let source_shard_count = usize_to_u64_saturating(source_shard_count);
        let summary = enabled_attribution.expert_streaming_source_summaries[summary_index]
            .get_or_insert(ExpertStreamingSourceSummary {
                phase,
                layer_index,
                source_plan_count: 0,
                total_route_token_count: 0,
                total_routed_expert_count: 0,
                total_streamed_expert_count: 0,
                total_source_shard_count: 0,
                payload_byte_count: 0,
            });
        summary.record_source_plan(
            route_token_count,
            routed_expert_count,
            streamed_expert_count,
            source_shard_count,
            payload_byte_count,
        );
    }
}

fn usize_to_u64_saturating(integer_count: usize) -> u64 {
    u64::try_from(integer_count).unwrap_or(u64::MAX)
}
