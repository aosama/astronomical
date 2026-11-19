//! The single storage boundary through which the streaming model reads
//! expert pages, so every read is tallied for the observed residency label
//! and filed under the engine's real serving phase.

use super::streaming_model::Qwen3_5StreamingModel;
use crate::PerformanceAttribution;
use crate::expert_paging::QuantizedExpertPageManifest;
use crate::qwen3_5_streaming::expert_streaming::ExpertPagingError;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::{
    Qwen3_5ExpertPager, Qwen3_5ExpertStreamingRequestShape, Qwen3_5PagedExpertWeights,
};
use astronomical_gpu_building_blocks::serving_phase::SessionStreamingPhase;

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
        self.stream_expert_page_from_storage_in_phase(
            expert_pager,
            layer_index,
            expert_ids,
            request_shape,
            self.current_serving_phase(request_shape.route_token_count),
            performance_attribution,
        )
    }

    /// [`Self::stream_expert_page_from_storage`] for a caller that knows which
    /// phase the read serves better than the engine's phase state does.
    ///
    /// Decode seating streams complete layers after the last prefill chunk but
    /// before the first decode forward switches the phase state, so those
    /// reads were filed as prefill: on a 12,401-token prompt two seated layers
    /// showed six prefill reads across five prefill forwards, which hid the
    /// seating cost inside prefill statistics (issue #1717).
    pub(super) fn stream_expert_page_from_storage_in_phase(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        expert_ids: &[usize],
        request_shape: Qwen3_5ExpertStreamingRequestShape,
        serving_phase: SessionStreamingPhase,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), ExpertPagingError> {
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
    /// The astro expert cache's demand phase is the real phase state: each request
    /// opens in prefill and the first decode forward switches it. Phase is
    /// never inferred from the route's token count — speculative and
    /// multi-token decode routes would otherwise be filed as prefill in both
    /// the observatory counters and the attribution reports (issue #1715).
    /// A model without an astro expert cache cannot track phase state; its streaming
    /// reads fall back to the route-length approximation, documented here so
    /// the fallback is visible rather than implicit.
    pub(super) fn current_serving_phase(&self, route_token_count: i32) -> SessionStreamingPhase {
        match self.astro_expert_cache.as_ref() {
            Some(astro_expert_cache) => astro_expert_cache.borrow().current_serving_phase(),
            None => fallback_serving_phase_from_route_token_count(route_token_count),
        }
    }

    /// Starts every count of expert reads afresh once the model has loaded,
    /// so nothing read to measure the model at load is reported as serving.
    pub fn begin_serving_session_observation(&self) {
        if let Some(expert_pager) = self.expert_pager.as_ref() {
            expert_pager.begin_serving_session();
        }
        if let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() {
            let mut astro_expert_cache = astro_expert_cache.borrow_mut();
            astro_expert_cache.forget_disk_loads_made_while_loading();
            astro_expert_cache.begin_prefill_demand_window();
        }
        self.deferred_expert_demand
            .borrow_mut()
            .discard_pending_layer_routes();
        self.request_expert_storage_read_tally.begin_request();
    }

    /// Starts the active request's storage-read observation window.
    pub fn begin_request_expert_storage_read_observation(&self) {
        self.request_expert_storage_read_tally.begin_request();
    }
}

/// Route-length approximation used only when no astro expert cache tracks phase.
fn fallback_serving_phase_from_route_token_count(route_token_count: i32) -> SessionStreamingPhase {
    if route_token_count > 1 {
        SessionStreamingPhase::Prefill
    } else {
        SessionStreamingPhase::Decode
    }
}
