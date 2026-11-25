//! Serving a reply token on the assumption that every routed expert is held.
//!
//! See `paged_route_resolution` for why. This module holds the two halves of
//! optimistic serving: serving one layer from its table with a count of the
//! experts the table lacked, and the loop around a whole token that runs it
//! again the exact way when any layer's count was not zero.

use crate::qwen3_5_core::model::target_forward_graph::Qwen3_5TargetForwardGraph;
use std::cell::Cell;

use mlx_bridge::MlxArray;

use super::Qwen3_5Model;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::paged_route_resolution::OptimisticLayerServing;
use super::{RequestDecoderStateStack, TextExecutionError};
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5ExpertPager;
use crate::qwen3_5_streaming::{PagedRouteValidationOutcome, Qwen3_5MoEPagedExecutionMode};
use crate::{PerformanceAttribution, PerformanceCounter};

/// A way to serve a reply token in one evaluation that can turn out wrong,
/// in which case the token is served again the exact way.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpeculativeWay {
    /// Each partly held layer looks its route up in its table inside the
    /// graph, reading missing experts into the table
    /// (`in_graph_table_serving`). Wrong only when a lookup refuses.
    InGraphLookups,
    /// Every routed expert is assumed held. Wrong whenever one was not.
    Optimistic,
}

/// What recent reply tokens say about serving the next one optimistically.
#[derive(Debug, Default)]
pub struct OptimisticServingHistory {
    /// Whether the token before this one was served without reading any
    /// expert from storage. It starts false: the first token of a model has
    /// no token before it to vouch for the astro expert cache.
    previous_token_was_served_from_memory: Cell<bool>,
    /// Reply tokens served optimistically since the model loaded, and how
    /// many of them had to be run again.
    optimistic_token_count: Cell<u64>,
    rerun_token_count: Cell<u64>,
}

impl Qwen3_5Model {
    /// Serves one layer of a speculative reply token from its table, whatever
    /// part of the layer the table holds: optimistically here, or by a lookup
    /// inside the graph (`dispatch_in_graph_table_page`) when the forward is
    /// in-graph. `Ok(None)` when the forward is exact, or when the layer
    /// cannot be served the forward's way; an optimistic forward is then
    /// marked to be run again.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_speculative_table_page(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        token_count: i32,
        decode_tokens_multiply_packed_pages: bool,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        hidden_states: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        let missing_route_collector = &self.paged_forward_missing_route_collector;
        if !missing_route_collector.is_optimistic() {
            // The other speculative way, when this forward is taking it; an
            // exact forward gets `None` from there as it did from here.
            return self.dispatch_in_graph_table_page(
                layer_index,
                selected_indices,
                selected_scores,
                hidden_states,
                mixture_of_experts_weights,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            );
        }
        let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
        let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() else {
            missing_route_collector.abandon_optimistic_serving();
            return Ok(None);
        };
        let table_page = astro_expert_cache
            .borrow()
            .optimistic_table_page(layer_index, expert_capacity);
        let Some(table_page) = table_page else {
            missing_route_collector.abandon_optimistic_serving();
            return Ok(None);
        };
        if !table_page
            .weights
            .multiplies_complete_page_without_transient_decode(
                token_count,
                decode_tokens_multiply_packed_pages,
            )
        {
            missing_route_collector.abandon_optimistic_serving();
            return Ok(None);
        }
        let missing_assignment_count = match table_page.missing_expert_mask.as_deref() {
            Some(missing_expert_mask) => {
                Some(self.missing_assignment_count(missing_expert_mask, selected_indices)?)
            }
            None => None,
        };
        missing_route_collector.record_optimistic_layer(OptimisticLayerServing {
            missing_assignment_count,
            assignment_count: selected_indices.element_count(),
            table_payload_bytes: table_page.manifest.payload_byte_count,
            expert_capacity,
        });
        self.defer_expert_demand(layer_index, expert_capacity, selected_indices);
        let layer_output = self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &table_page.weights,
            &table_page.manifest,
            selected_indices,
            &table_page.manifest.expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )?;
        Ok(Some(layer_output))
    }

    /// Serves a layer from its table with the route remapped on-device: any
    /// table when the forward is optimistic, otherwise only a whole layer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_table_page_without_host_route(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        token_count: i32,
        decode_tokens_multiply_packed_pages: bool,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        hidden_states: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        if let Some(handled) = self.dispatch_speculative_table_page(
            expert_pager,
            layer_index,
            token_count,
            decode_tokens_multiply_packed_pages,
            selected_indices,
            selected_scores,
            mixture_of_experts_weights,
            hidden_states,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )? {
            return Ok(Some(handled));
        }
        self.dispatch_retained_full_hit_packed_page(
            expert_pager,
            layer_index,
            token_count,
            decode_tokens_multiply_packed_pages,
            selected_indices,
            selected_scores,
            mixture_of_experts_weights,
            hidden_states,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }

    /// How many of the routed experts the table lacks, as a lazy array: the
    /// mask is gathered by the route and summed on the graphics processor,
    /// so the route never has to reach the host to be checked.
    fn missing_assignment_count(
        &self,
        missing_expert_mask: &[u32],
        selected_indices: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        let expert_capacity = i32::try_from(missing_expert_mask.len()).map_err(|_| {
            TextExecutionError::InvalidInput {
                description: "expert capacity exceeds the MLX shape range",
            }
        })?;
        let missing_expert_mask = self
            .runtime
            .array_from_u32(missing_expert_mask, &[expert_capacity])?;
        let routed_missing_flags =
            self.runtime
                .take_axis(&missing_expert_mask, selected_indices, 0)?;
        let flat_routed_missing_flags = self.runtime.reshape(&routed_missing_flags, &[-1])?;
        Ok(self.runtime.sum_axis(&flat_routed_missing_flags, 0, true)?)
    }

    /// Runs one forward to completion: speculatively first when a way that
    /// keeps the token one evaluation is likely to be right, and again the
    /// exact way when it was not.
    pub(super) fn forward_resolving_expert_routes(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let storage_read_bytes_before = self
            .request_expert_storage_read_tally
            .expert_storage_read_byte_count();
        let mut speculative_attempt =
            self.speculative_attempt(token_count, request_decoder_state)?;
        loop {
            if let Some((speculative_way, _)) = speculative_attempt.as_ref() {
                self.begin_speculative_forward(*speculative_way, performance_attribution);
            }
            let attempt = self.build_and_evaluate_forward(
                token_indices,
                token_count,
                starting_position_tokens,
                request_decoder_state,
                performance_attribution,
            );
            self.end_speculative_forward();
            let (logits, validation_outcome) = attempt?;
            match validation_outcome {
                PagedRouteValidationOutcome::CompleteHit => {
                    self.remember_whether_the_token_read_from_storage(storage_read_bytes_before);
                    return Ok(logits);
                }
                PagedRouteValidationOutcome::MissedExperts => {
                    let Some((speculative_way, checkpoint)) = speculative_attempt.take() else {
                        return Err(TextExecutionError::InvalidInput {
                            description: "an exact forward reported experts missing from memory",
                        });
                    };
                    request_decoder_state.restore_checkpoint(checkpoint)?;
                    self.record_speculative_miss(speculative_way, performance_attribution);
                }
            }
        }
    }

    /// The speculative way to try first, with the state to return to should
    /// it turn out wrong; `None` for a token served exactly from the start.
    fn speculative_attempt(
        &self,
        token_count: i32,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<
        Option<(
            SpeculativeWay,
            text_generation::RequestDecoderStateStackCheckpoint,
        )>,
        TextExecutionError,
    > {
        let Some(speculative_way) = self.speculative_way(token_count) else {
            return Ok(None);
        };
        Ok(Some((speculative_way, request_decoder_state.checkpoint()?)))
    }

    /// The speculative way to try first, if any. Looking routes up inside
    /// the graph comes before optimism: it serves the token when experts are
    /// missing, where optimism only finds out that they were.
    fn speculative_way(&self, token_count: i32) -> Option<SpeculativeWay> {
        if self.should_serve_in_graph(token_count) {
            Some(SpeculativeWay::InGraphLookups)
        } else if self.should_serve_optimistically(token_count) {
            Some(SpeculativeWay::Optimistic)
        } else {
            None
        }
    }

    fn begin_speculative_forward(
        &self,
        speculative_way: SpeculativeWay,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        match speculative_way {
            SpeculativeWay::InGraphLookups => self.begin_in_graph_forward(1),
            SpeculativeWay::Optimistic => {
                self.paged_forward_missing_route_collector
                    .set_optimistic(true);
                performance_attribution
                    .record_counter(PerformanceCounter::OptimisticReplyTokenCount, 1);
                let history = &self.optimistic_serving_history;
                history
                    .optimistic_token_count
                    .set(history.optimistic_token_count.get() + 1);
            }
        }
    }

    /// Ends whichever speculative forward was in progress; harmless after an
    /// exact one.
    fn end_speculative_forward(&self) {
        self.paged_forward_missing_route_collector
            .set_optimistic(false);
        self.end_in_graph_forward();
    }

    fn record_speculative_miss(
        &self,
        speculative_way: SpeculativeWay,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let SpeculativeWay::Optimistic = speculative_way else {
            // In-graph serving counted its own refusal when it was found.
            return;
        };
        performance_attribution
            .record_counter(PerformanceCounter::OptimisticReplyTokenMissCount, 1);
        let history = &self.optimistic_serving_history;
        history
            .rerun_token_count
            .set(history.rerun_token_count.get() + 1);
        // A warming astro expert cache can miss on many tokens of one reply,
        // so this stays out of the default log.
        tracing::debug!(
            optimistic_token_count = history.optimistic_token_count.get(),
            rerun_token_count = history.rerun_token_count.get(),
            "an optimistic reply token routed to an expert not in memory; running it again"
        );
    }

    fn build_and_evaluate_forward(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, PagedRouteValidationOutcome), TextExecutionError> {
        let logits = self.build_forward_graph(
            token_indices,
            token_count,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            performance_attribution,
        )?;
        let validation_outcome = self.evaluate_forward_state_with_performance_attribution(
            &logits,
            request_decoder_state,
            performance_attribution,
        )?;
        Ok((logits, validation_outcome))
    }

    /// A reply token is served optimistically only after a token that read nothing from
    /// storage: misses come in runs while an astro expert cache warms, and serving optimistically
    /// through a run would pay for two forwards on every token of it.
    fn should_serve_optimistically(&self, token_count: i32) -> bool {
        if token_count != 1
            || !self
                .optimistic_serving_history
                .previous_token_was_served_from_memory
                .get()
        {
            return false;
        }
        self.has_a_partly_held_layer_and_no_empty_one()
    }

    /// Optimistic serving needs a table in every layer, and it only pays when some
    /// layer is partly held. When every layer is held whole the exact way
    /// already runs without a wait, and optimism would add the state
    /// checkpoint for nothing (measured: 57.3 against 58.9 tokens per second
    /// on a 40-layer model held whole).
    fn has_a_partly_held_layer_and_no_empty_one(&self) -> bool {
        let (Some(astro_expert_cache), Some(expert_pager)) =
            (self.astro_expert_cache.as_ref(), self.expert_pager.as_ref())
        else {
            return false;
        };
        let astro_expert_cache = astro_expert_cache.borrow();
        let mut has_partly_held_layer = false;
        for layer_index in 0..expert_pager.layer_count() {
            let Ok(layer_plan) = expert_pager.layer_plan(layer_index) else {
                return false;
            };
            let held_expert_count = astro_expert_cache.occupied_slot_count(layer_index);
            if held_expert_count == 0 {
                return false;
            }
            has_partly_held_layer |= held_expert_count < layer_plan.expert_capacity;
        }
        has_partly_held_layer
    }

    fn remember_whether_the_token_read_from_storage(&self, storage_read_bytes_before: u64) {
        let storage_read_bytes_after = self
            .request_expert_storage_read_tally
            .expert_storage_read_byte_count();
        self.optimistic_serving_history
            .previous_token_was_served_from_memory
            .set(storage_read_bytes_after == storage_read_bytes_before);
    }
}
