//! Serves a layer whose table does not hold the whole layer.
//!
//! A multi-token chunk streams the union of the experts it routes to. A
//! reply token classifies how much of its route the table holds and takes
//! the cheapest way: missing experts read straight into the table, a mixed
//! route, a table hit, or a stream.

use mlx_bridge::MlxArray;

use crate::qwen3_5_text::model::Qwen3_5Model;
use crate::{PerformanceAttribution, PerformanceCounter};
use text_generation::execution_error::TextExecutionError;

use super::super::expert_streaming::expert_pager::Qwen3_5ExpertPager;
use super::Qwen3_5MoEPagedExecutionMode;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;

impl Qwen3_5Model {
    /// Serves a table miss: a multi-token chunk streams its routed-expert
    /// union; a reply token is served by `dispatch_reply_token_miss`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_miss_route(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        expert_capacity: usize,
        token_count: i32,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        selected_expert_ids: &[usize],
        routed_expert_ids: &[usize],
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        hidden_states: &MlxArray,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        performance_attribution.record_counter(
            PerformanceCounter::RetainedRouteAssignmentMissCount,
            u64::try_from(selected_expert_ids.len()).unwrap_or(u64::MAX),
        );
        if self
            .paged_forward_missing_route_collector
            .serves_misses_as_reply_tokens(token_count)
        {
            return self.dispatch_reply_token_miss(
                expert_pager,
                layer_index,
                expert_capacity,
                selected_indices,
                selected_scores,
                selected_expert_ids,
                routed_expert_ids,
                mixture_of_experts_weights,
                hidden_states,
                paged_execution_mode,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            );
        }
        // Multi-token chunks stream the chunk's routed-expert union, not the
        // complete layer: the union is bounded by chunk tokens × experts per
        // token, which for every chunk smaller than expert_capacity ÷ experts
        // per token is a fraction of the complete layer's read and decode
        // work. The union page stays operation-local (warm tables fill from
        // decode tokens), and the forward admission reserves the same union
        // bound (issue #1422).
        let (packed_weights, packed_manifest) = self.stream_operation_local_routed_experts(
            expert_pager,
            layer_index,
            token_count,
            routed_expert_ids,
            paged_execution_mode == Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            performance_attribution,
        )?;
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &packed_weights,
            &packed_manifest,
            selected_indices,
            routed_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }

    /// Serves one reply token on a layer not held whole. Issue #373: warm-table
    /// coverage is classified before dispatch, so each cost stays measurable
    /// and picks the way: missing experts read straight into the table when
    /// it can take them (the route is then a table hit); otherwise a route
    /// partly in the table is served mixed, a route wholly in it from
    /// retained memory, and a route wholly outside it streams.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_reply_token_miss(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        expert_capacity: usize,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        selected_expert_ids: &[usize],
        routed_expert_ids: &[usize],
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        hidden_states: &MlxArray,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let production_default_paging =
            paged_execution_mode == Qwen3_5MoEPagedExecutionMode::ProductionDefault;
        let route_coverage = self.record_decode_route_coverage(
            layer_index,
            routed_expert_ids,
            performance_attribution,
        );
        let route_has_misses = !route_coverage.missing_expert_ids.is_empty();
        // With the misses read into the table there is no page loaded and
        // nothing to copy afterwards.
        let misses_were_fetched_into_table = route_has_misses
            && self.fetch_decode_misses_into_table(
                expert_pager,
                layer_index,
                &route_coverage,
                production_default_paging,
                performance_attribution,
            )?;
        if route_has_misses
            && !misses_were_fetched_into_table
            && serves_one_row_partly_from_its_table(
                selected_indices,
                &route_coverage.retained_expert_ids,
            )
        {
            // Mixed serving: serve the covered assignments from retained RAM
            // and stream only the missing experts.
            return self.forward_moe_mixed_decode_route_with_performance_attribution(
                hidden_states,
                mixture_of_experts_weights,
                expert_pager,
                layer_index,
                selected_indices,
                selected_scores,
                selected_expert_ids,
                &route_coverage,
                expert_capacity,
                paged_execution_mode,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            );
        }
        // The warm table covering every routed expert serves the route from
        // retained RAM with no storage read (issue #372).
        let table_page = self.hot_expert_partial_page(
            layer_index,
            expert_capacity,
            routed_expert_ids,
            performance_attribution,
        );
        let (packed_weights, packed_manifest) = match table_page {
            Some(table_page) => table_page,
            None => self.stream_operation_local_routed_experts(
                expert_pager,
                layer_index,
                // `[1, rows, experts per token]`: one row for a reply
                // token, several for a verified drafted block.
                selected_indices.shape()[1],
                routed_expert_ids,
                production_default_paging,
                performance_attribution,
            )?,
        };
        // The page covers exactly the routed union, so the gather remap uses
        // the routed expert ids.
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &packed_weights,
            &packed_manifest,
            selected_indices,
            routed_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }
}

/// Whether a route its table could not take is served mixed: the experts
/// the table holds from memory, the rest streamed. Mixed serving orders one
/// row's assignments, so the rows of a drafted forward (which reach here
/// only when a lookup refused and the forward is run again) stream their
/// routed union instead.
fn serves_one_row_partly_from_its_table(
    selected_indices: &MlxArray,
    retained_expert_ids: &[usize],
) -> bool {
    selected_indices.shape().get(1) == Some(&1) && !retained_expert_ids.is_empty()
}
