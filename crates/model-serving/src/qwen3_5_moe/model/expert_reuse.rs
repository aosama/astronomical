//! Returns a packed complete expert layer on hit, or streams one complete layer on miss.

use crate::expert_paging::QuantizedExpertPageManifest;
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::expert_paging::expert_pager::{
    Qwen3_5ExpertPager, Qwen3_5ExpertStreamingRequestShape, Qwen3_5PagedExpertWeights,
};
use crate::{PerformanceAttribution, should_commit_mandatory_complete_layer};

/// How one forward should execute one mixture-of-experts layer.
pub(super) enum ExpertPageDisposition {
    /// A complete cached layer covers every possible route.
    FullHit,
    /// This layer is not seated. Prefill streams the complete layer.
    Miss,
}

impl Qwen3_5Model {
    /// Complete-layer cache hit when every expert of the decoder index is seated.
    pub(super) fn expert_page_disposition(
        &self,
        layer_index: usize,
        expert_capacity: usize,
    ) -> ExpertPageDisposition {
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return ExpertPageDisposition::Miss;
        };
        if retained_experts
            .borrow()
            .has_complete_layer(layer_index, expert_capacity)
        {
            ExpertPageDisposition::FullHit
        } else {
            ExpertPageDisposition::Miss
        }
    }

    /// Returns the cached complete packed page.
    pub(super) fn cached_packed_page(
        &self,
        layer_index: usize,
        complete_expert_ids: &[usize],
        expert_capacity: usize,
    ) -> Option<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest)> {
        let retained_experts = self.retained_experts.as_ref()?;
        let cache = retained_experts.borrow();
        cache.packed_page(layer_index, complete_expert_ids, expert_capacity)
    }

    /// Streams every expert in the layer. Seats that page only when the phase
    /// plan named this decoder index for mandatory complete-layer promotion.
    ///
    /// Leftover-RAM adopt during the first multi-token chunk seats more
    /// complete layers than activation headroom plus the stream slot can share
    /// with the MLX ceiling. The next complete-layer read then fails, and
    /// prefill halves down to one-token chunks.
    pub(super) fn stream_complete_expert_layer(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_token_count: i32,
        expert_capacity: usize,
        routed_expert_count: usize,
        production_default_paging: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), Qwen3_5ExecutionError>
    {
        let complete_expert_ids: Vec<usize> = (0..expert_capacity).collect();
        let (streamed_weights, streamed_manifest) = expert_pager.load_rust_streamed_experts(
            &self.runtime,
            layer_index,
            &complete_expert_ids,
            Qwen3_5ExpertStreamingRequestShape {
                route_token_count,
                routed_expert_count,
            },
            performance_attribution,
        )?;
        if let Some(retained_experts) = self.retained_experts.as_ref() {
            retained_experts.borrow_mut().record_disk_load(
                complete_expert_ids.len(),
                streamed_manifest.source_manifests.len(),
            );
            let residency_target = self.expert_residency_target(layer_index);
            if should_commit_mandatory_complete_layer(
                route_token_count,
                production_default_paging,
                residency_target,
            ) {
                // Adopt the lazy page; the layer interval eval materializes it.
                // A separate weight eval here double-synced every seated layer
                // on the first prefill chunk without changing gather inputs.
                let seated = retained_experts.borrow_mut().try_adopt_complete_layer(
                    &self.runtime,
                    layer_index,
                    &complete_expert_ids,
                    &streamed_weights,
                )?;
                tracing::debug!(
                    layer_index,
                    seated,
                    ?residency_target,
                    streamed_payload_bytes = streamed_manifest.payload_byte_count,
                    "complete-layer stream offered to retained ownership",
                );
            } else {
                tracing::debug!(
                    layer_index,
                    ?residency_target,
                    streamed_payload_bytes = streamed_manifest.payload_byte_count,
                    "kept complete-layer stream operation-local",
                );
            }
        }
        Ok((streamed_weights, streamed_manifest))
    }

    /// Decode-only: stream the unique routed experts for this token without
    /// displacing a seated complete layer.
    pub(super) fn stream_operation_local_routed_experts(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_token_count: i32,
        routed_expert_ids: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), Qwen3_5ExecutionError>
    {
        let (streamed_weights, streamed_manifest) = expert_pager.load_rust_streamed_experts(
            &self.runtime,
            layer_index,
            routed_expert_ids,
            Qwen3_5ExpertStreamingRequestShape {
                route_token_count,
                routed_expert_count: routed_expert_ids.len(),
            },
            performance_attribution,
        )?;
        if let Some(retained_experts) = self.retained_experts.as_ref() {
            retained_experts.borrow_mut().record_disk_load(
                routed_expert_ids.len(),
                streamed_manifest.source_manifests.len(),
            );
        }
        Ok((streamed_weights, streamed_manifest))
    }
}
