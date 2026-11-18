//! Serving a layer whose route has reached the host: from its complete
//! table, or, for a layer not held whole, through `miss_route_dispatch.rs`.
//!
//! `forward.rs` decides whether a route can stay on the graphics processor.
//! When it cannot, the routed experts are known here by id, so this file can
//! record the demand they show and pick the cheapest source for each.

use mlx_bridge::MlxArray;

use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{PerformanceAttribution, PerformanceCounter};

use super::super::expert_streaming::expert_pager::Qwen3_5ExpertPager;
use super::Qwen3_5MoEPagedExecutionMode;
use super::expert_dispatch_tally::ExpertDispatchPath;
use super::expert_reuse::ExpertPageDisposition;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;

impl Qwen3_5Model {
    /// Records the routed demand on the retained tables and dispatches on the
    /// layer's page disposition; both dispositions are terminal here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_routed_disposition(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        token_count: i32,
        decode_tokens_multiply_packed_pages: bool,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        selected_expert_ids: &[usize],
        routed_expert_ids: &[usize],
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        hidden_states: &MlxArray,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, Qwen3_5ExecutionError> {
        let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
        if let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() {
            astro_expert_cache.borrow_mut().record_expert_demand(
                layer_index,
                expert_capacity,
                selected_expert_ids,
            );
        }
        let page_disposition = self.expert_page_disposition(layer_index, expert_capacity);
        self.expert_dispatch_tally
            .record_path(host_routed_dispatch_path(&page_disposition));
        match page_disposition {
            ExpertPageDisposition::FullHit => Ok(Some(self.dispatch_full_hit_routed_page(
                layer_index,
                expert_capacity,
                token_count,
                decode_tokens_multiply_packed_pages,
                selected_indices,
                selected_scores,
                selected_expert_ids,
                routed_expert_ids,
                mixture_of_experts_weights,
                hidden_states,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            )?)),
            ExpertPageDisposition::Miss => Ok(Some(self.dispatch_miss_route(
                expert_pager,
                layer_index,
                expert_capacity,
                token_count,
                selected_indices,
                selected_scores,
                selected_expert_ids,
                routed_expert_ids,
                mixture_of_experts_weights,
                hidden_states,
                paged_execution_mode,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            )?)),
        }
    }

    /// Serves a full table hit from the seated layer: multi-token chunks
    /// decode exactly their routed union, decode tokens with the fused kernel
    /// decode the complete packed page.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_full_hit_routed_page(
        &self,
        layer_index: usize,
        expert_capacity: usize,
        token_count: i32,
        decode_tokens_multiply_packed_pages: bool,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        selected_expert_ids: &[usize],
        routed_expert_ids: &[usize],
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        hidden_states: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        performance_attribution.record_counter(
            PerformanceCounter::RetainedRouteAssignmentHitCount,
            u64::try_from(selected_expert_ids.len()).unwrap_or(u64::MAX),
        );
        if token_count > 1 || !decode_tokens_multiply_packed_pages {
            // A forward that decodes the page it multiplies gets
            // exactly its routed union of the seated layer.
            let (subset_weights, subset_manifest) = self
                .routed_subset_of_astro_expert_cache_page(
                    layer_index,
                    routed_expert_ids,
                    expert_capacity,
                )?
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "astro_expert_cache hit reported a page that vanished",
                })?;
            self.account_resident_subset_hit(
                token_count,
                routed_expert_ids.len(),
                subset_manifest.payload_byte_count,
                performance_attribution,
            );
            return self.forward_moe_paged_with_performance_attribution(
                hidden_states,
                mixture_of_experts_weights,
                &subset_weights,
                &subset_manifest,
                selected_indices,
                routed_expert_ids,
                selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            );
        }
        let complete_expert_ids: Vec<usize> = (0..expert_capacity).collect();
        let (packed_weights, packed_manifest) = self
            .astro_expert_cache_packed_page(layer_index, &complete_expert_ids, expert_capacity)
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "astro_expert_cache hit reported a page that vanished",
            })?;
        self.record_resident_layer_decode_hit(
            routed_expert_ids.len(),
            packed_manifest.payload_byte_count,
            expert_capacity,
            performance_attribution,
        );
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &packed_weights,
            &packed_manifest,
            selected_indices,
            &complete_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }
}

/// The tally's name for a layer served after its route reached the host.
fn host_routed_dispatch_path(page_disposition: &ExpertPageDisposition) -> ExpertDispatchPath {
    match page_disposition {
        ExpertPageDisposition::FullHit => ExpertDispatchPath::HostRouteTableHit,
        ExpertPageDisposition::Miss => ExpertDispatchPath::HostRouteTableMiss,
    }
}
