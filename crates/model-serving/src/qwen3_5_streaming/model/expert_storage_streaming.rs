//! The single storage boundary through which the streaming model reads
//! expert pages, so every read is tallied for the observed residency label.

use super::streaming_model::Qwen3_5StreamingModel;
use crate::PerformanceAttribution;
use crate::expert_paging::QuantizedExpertPageManifest;
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
        let (streamed_expert_weights, streamed_page_manifest) = expert_pager
            .stream_selected_expert_page_from_ssd(
                &self.runtime,
                layer_index,
                expert_ids,
                request_shape,
                performance_attribution,
            )?;
        self.request_expert_storage_read_tally
            .record_streamed_page(streamed_page_manifest.payload_byte_count);
        Ok((streamed_expert_weights, streamed_page_manifest))
    }

    /// Starts the active request's storage-read observation window.
    pub(crate) fn begin_request_expert_storage_read_observation(&self) {
        self.request_expert_storage_read_tally.begin_request();
    }
}
