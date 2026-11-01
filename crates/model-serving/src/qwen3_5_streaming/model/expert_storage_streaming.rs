//! The single storage boundary through which the streaming model reads
//! expert pages, so every read is tallied for the observed residency label
//! and filed under the engine's real serving phase.

use super::streaming_model::Qwen3_5StreamingModel;
use crate::PerformanceAttribution;
use crate::expert_paging::QuantizedExpertPageManifest;
use crate::memory::SessionStreamingPhase;
use crate::qwen3_5_streaming::expert_streaming::ExpertPagingError;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::{
    Qwen3_5ExpertPager, Qwen3_5ExpertStreamingRequestShape, Qwen3_5PagedExpertWeights,
};

impl Qwen3_5StreamingModel {
    /// Streams the selected expert page from storage and records its payload
    /// in the active request's storage-read tally.
    pub(super) fn stream_expert_page_from_storage(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        expert_ids: &[usize],
        request_shape: Qwen3_5ExpertStreamingRequestShape,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), ExpertPagingError> {
        let serving_phase = self.current_serving_phase(request_shape.route_token_count);
        let (streamed_expert_weights, streamed_page_manifest) = expert_pager
            .stream_selected_expert_page_from_ssd(
                &self.runtime,
                layer_index,
                expert_ids,
                request_shape,
                serving_phase,
                performance_attribution,
            )?;
        self.request_expert_storage_read_tally
            .record_streamed_page(streamed_page_manifest.payload_byte_count);
        Ok((streamed_expert_weights, streamed_page_manifest))
    }

    /// The engine's serving phase at this storage read.
    ///
    /// The warm cache's demand phase is the real phase state: each request
    /// opens in prefill and the first decode forward switches it. Phase is
    /// never inferred from the route's token count — speculative and
    /// multi-token decode routes would otherwise be filed as prefill in both
    /// the observatory counters and the attribution reports (issue #1715).
    /// A model without a warm cache cannot track phase state; its streaming
    /// reads fall back to the route-length approximation, documented here so
    /// the fallback is visible rather than implicit.
    fn current_serving_phase(&self, route_token_count: i32) -> SessionStreamingPhase {
        match self.retained_experts.as_ref() {
            Some(retained_experts) => retained_experts.borrow().current_serving_phase(),
            None => fallback_serving_phase_from_route_token_count(route_token_count),
        }
    }

    /// Starts the active request's storage-read observation window.
    pub(crate) fn begin_request_expert_storage_read_observation(&self) {
        self.request_expert_storage_read_tally.begin_request();
    }
}

/// Route-length approximation used only when no warm cache tracks phase.
fn fallback_serving_phase_from_route_token_count(route_token_count: i32) -> SessionStreamingPhase {
    if route_token_count > 1 {
        SessionStreamingPhase::Prefill
    } else {
        SessionStreamingPhase::Decode
    }
}
