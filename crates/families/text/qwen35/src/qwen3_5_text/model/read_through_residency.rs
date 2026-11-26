//! Streams missing experts, stores each one, and runs this forward from the astro expert cache.

use mlx_bridge::MlxArray;

use crate::qwen3_5_text::expert_streaming::expert_pager::{
    ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5ExpertStreamingRequestShape,
};
use crate::qwen3_5_text::model::Qwen3_5Model;
use crate::{PerformanceAttribution, PerformanceOperation};
use text_generation::execution_error::TextExecutionError;

use super::Qwen3_5MoEPagedExecutionMode;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;

impl Qwen3_5Model {
    // Paging dependencies stay explicit instead of adding a request-context abstraction.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_with_expert_store_paging(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_token_count: i32,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
        known_sorted_unique_expert_ids: Option<&[usize]>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let copied_sorted_unique_expert_ids;
        let sorted_unique_expert_ids = match known_sorted_unique_expert_ids {
            Some(sorted_unique_expert_ids) => sorted_unique_expert_ids,
            None => {
                copied_sorted_unique_expert_ids =
                    self.copy_sorted_unique_expert_ids(selected_indices)?;
                &copied_sorted_unique_expert_ids
            }
        };
        if self.astro_expert_cache.is_some() {
            let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
            let (packed_weights, packed_manifest) = if route_token_count > 1 {
                self.stream_complete_expert_layer(
                    expert_pager,
                    layer_index,
                    route_token_count,
                    expert_capacity,
                    sorted_unique_expert_ids.len(),
                    paged_execution_mode == Qwen3_5MoEPagedExecutionMode::ProductionDefault,
                    self.current_serving_phase(route_token_count),
                    performance_attribution,
                )?
            } else {
                // astro expert cache hit: the warm table covers every routed
                // expert of this token, so the whole routed set is served from
                // retained RAM and no storage read happens for this layer
                // (issue #372).
                //
                self.record_decode_route_coverage(
                    layer_index,
                    sorted_unique_expert_ids,
                    performance_attribution,
                );
                match self.hot_expert_partial_page(
                    layer_index,
                    expert_capacity,
                    sorted_unique_expert_ids,
                    performance_attribution,
                ) {
                    Some(packed_page) => packed_page,
                    None => self.stream_operation_local_routed_experts(
                        expert_pager,
                        layer_index,
                        route_token_count,
                        sorted_unique_expert_ids,
                        paged_execution_mode == Qwen3_5MoEPagedExecutionMode::ProductionDefault,
                        performance_attribution,
                    )?,
                }
            };
            if let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() {
                astro_expert_cache.borrow_mut().record_expert_demand(
                    layer_index,
                    expert_pager.layer_plan(layer_index)?.expert_capacity,
                    sorted_unique_expert_ids,
                );
            }
            return self.forward_moe_paged_with_performance_attribution(
                hidden_states,
                mixture_of_experts_weights,
                &packed_weights,
                &packed_manifest,
                selected_indices,
                sorted_unique_expert_ids,
                selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            );
        }
        let (streamed_expert_weights, page_manifest) = self.stream_expert_page_from_storage(
            expert_pager,
            layer_index,
            sorted_unique_expert_ids,
            Qwen3_5ExpertStreamingRequestShape {
                route_token_count,
                routed_expert_count: sorted_unique_expert_ids.len(),
            },
            performance_attribution,
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ExpertPagingDiagnosticLogging,
            |_performance_attribution| {
                tracing::debug!(
                    layer_index,
                    streamed_expert_count = sorted_unique_expert_ids.len(),
                    streamed_payload_bytes = page_manifest.payload_byte_count,
                    "streamed experts without a retained astro_expert_cache"
                );
                Ok::<(), ExpertPagingError>(())
            },
        )?;
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &streamed_expert_weights,
            &page_manifest,
            selected_indices,
            sorted_unique_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }
}
