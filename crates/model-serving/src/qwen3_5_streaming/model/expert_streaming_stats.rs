//! Observatory-facing session statistics assembled from pager-owned counters.
//!
//! The pager owns always-on counters and the always-on positional-read
//! aggregate; this module composes them with the warm-cache topology and the
//! finalization residency claim into one `ExpertStreamingStats` snapshot. The
//! composition is read-only over model state and performs no MLX work, so it
//! stays safe to call at every generation finalization.

use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::inference_engine::contract::{
    ExpertResidencyTelemetry, ExpertStreamingStats, GenerationFinalization,
};
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5ExpertPager;

use super::Qwen3_5Model;

/// Library size and the uniform per-layer expert count for one pager.
///
/// A failed library measurement publishes zero so Observatory hides the SSD
/// remainder instead of subtracting the model file size. Capacity still comes
/// from the first layer plan, which every streaming layer shares.
pub(super) fn expert_library_facts(expert_pager: &Qwen3_5ExpertPager) -> (u64, u32) {
    let complete_expert_payload_byte_count = expert_pager
        .complete_expert_payload_byte_count()
        .unwrap_or_default();
    let expert_capacity = match expert_pager.layer_plans().first() {
        Some(first_layer_plan) => {
            u32::try_from(first_layer_plan.expert_capacity).unwrap_or(u32::MAX)
        }
        None => 0,
    };
    (complete_expert_payload_byte_count, expert_capacity)
}

impl Qwen3_5Model {
    /// Attaches the always-on expert-streaming session snapshot to a
    /// finalization when this model streams experts. Dense models (no pager)
    /// attach nothing and the daemon falls back to the residency claim alone.
    pub(crate) fn attach_expert_streaming_stats(
        &self,
        generation_finalization: GenerationFinalization,
        model_id: Option<&str>,
        expert_memory_mode: Option<ExpertMemoryMode>,
        expert_residency_telemetry: Option<ExpertResidencyTelemetry>,
    ) -> GenerationFinalization {
        match self.expert_streaming_session_stats(
            model_id,
            expert_memory_mode,
            expert_residency_telemetry,
        ) {
            Some(expert_streaming_stats) => generation_finalization
                .with_expert_streaming_stats(Some(Box::new(expert_streaming_stats))),
            None => generation_finalization,
        }
    }

    /// Composes the session snapshot; `None` when this model has no pager.
    fn expert_streaming_session_stats(
        &self,
        model_id: Option<&str>,
        expert_memory_mode: Option<ExpertMemoryMode>,
        expert_residency_telemetry: Option<ExpertResidencyTelemetry>,
    ) -> Option<ExpertStreamingStats> {
        let expert_pager = self.expert_pager.as_ref()?;
        let session_counters = expert_pager.session_counter_snapshot();
        let (read_call_count, read_byte_count, read_elapsed_nanoseconds) =
            expert_pager.session_read_totals();
        // The first layer plan carries the uniform expert capacity every
        // layer shares, the same source the decode warm handoff trusts.
        let (complete_expert_payload_byte_count, expert_capacity) =
            expert_library_facts(expert_pager);
        let layer_topology = match self.retained_experts.as_ref() {
            Some(retained_experts) => {
                // The observatory contract stores the uniform expert count as
                // u32. The warm-slot table still keys slots by usize, and
                // every supported target can hold that count.
                let topology_expert_capacity =
                    usize::try_from(expert_capacity).unwrap_or(usize::MAX);
                let topology_residencies = retained_experts
                    .borrow()
                    .topology_snapshot(topology_expert_capacity);
                expert_pager.session_layer_topology(&topology_residencies)
            }
            None => Vec::new(),
        };
        let expert_residency_telemetry = expert_residency_telemetry.unwrap_or_default();
        let observed_expert_memory_mode = match expert_memory_mode {
            Some(finalization_expert_memory_mode) => finalization_expert_memory_mode,
            None => self.observed_expert_memory_mode(),
        };
        Some(ExpertStreamingStats {
            model_id: model_id.map(str::to_owned),
            expert_memory_mode: observed_expert_memory_mode,
            total_layer_count: expert_residency_telemetry.total_layer_count,
            resident_expert_count: expert_residency_telemetry.resident_expert_count,
            resident_expert_payload_bytes: expert_residency_telemetry.resident_expert_payload_bytes,
            complete_expert_payload_byte_count,
            expert_capacity,
            layer_topology,
            retained_route_assignment_count: session_counters.retained_route_assignment_count,
            storage_route_assignment_count: session_counters.storage_route_assignment_count,
            hot_route_fully_covered_token_count: session_counters
                .hot_route_fully_covered_token_count,
            hot_route_partially_covered_token_count: session_counters
                .hot_route_partially_covered_token_count,
            hot_route_missed_token_count: session_counters.hot_route_missed_token_count,
            expert_source_read_call_count: read_call_count,
            expert_source_read_byte_count: read_byte_count,
            expert_source_read_elapsed_nanoseconds: read_elapsed_nanoseconds,
            avoided_complete_layer_source_bytes: session_counters
                .avoided_complete_layer_source_bytes,
            mandatory_prefill_source_bytes: session_counters.mandatory_prefill_source_bytes,
            mandatory_decode_source_bytes: session_counters.mandatory_decode_source_bytes,
            // Request-scoped token counts are enriched by the worker at the
            // emission boundary; the model itself does not track them.
            prompt_token_count: 0,
            generated_token_count: 0,
        })
    }
}
