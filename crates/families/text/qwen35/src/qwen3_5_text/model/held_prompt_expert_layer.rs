//! One streamed expert layer, held for every span of a prompt that is read
//! one layer at a time.
//!
//! Chunk by chunk, an expert model reads a streamed layer once per chunk.
//! One layer at a time, the layer's complete page is read once and
//! multiplied by every span of the prompt before the next layer is read
//! (issue #2221). The page is the read-ahead of
//! `prefill_complete_layer_prefetch` when that read was this layer's, and a
//! fresh read otherwise.
//!
//! A complete page is the right read when the prompt is long enough to route
//! to every expert and the page multiplies as read
//! (`prompt_reads_complete_expert_layer`). Otherwise nothing is held and
//! each span reads the experts it routes to, as a chunk does.
//!
//! A layer the astro expert cache keeps only part of is held like any other: the whole
//! page is read once and its hot table is left alone unless the residency
//! plan lets the astro expert cache adopt the complete layer in its place.

use mlx_bridge::MlxArray;

use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::prefill_complete_layer_prefetch::PartlyKeptLayers;
use crate::qwen3_5_text::expert_streaming::expert_pager::{
    Qwen3_5ExpertPager, Qwen3_5ExpertStreamingRequestShape, Qwen3_5PagedExpertWeights,
};
use crate::qwen3_5_text::model::Qwen3_5Model;
use crate::{
    PerformanceAttribution, PerformanceCounter, should_commit_multi_token_union_as_complete_layer,
};
use engine_core::expert_paging::QuantizedExpertPageManifest;
use text_generation::execution_error::TextExecutionError;

/// The complete page of the layer a layer-at-a-time prompt is working on.
#[derive(Debug)]
pub struct HeldPromptExpertLayer {
    layer_index: usize,
    weights: Qwen3_5PagedExpertWeights,
    manifest: QuantizedExpertPageManifest,
}

/// The complete expert pages a prompt forward has in hand: the layer it is
/// working on, and the layer read ahead while that one computes.
#[derive(Debug, Default)]
pub struct PromptExpertPagesInHand {
    pub(super) held_layer: std::cell::RefCell<Option<HeldPromptExpertLayer>>,
    pub(super) read_ahead_layer:
        std::cell::RefCell<Option<super::prefill_complete_layer_prefetch::PrefetchedCompleteLayer>>,
}

/// Bytes of the complete expert pages in hand.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertPagesInHandBytes {
    /// The layer being worked on.
    pub held_bytes: u64,
    /// The layer read ahead, which outlives this layer's step.
    pub read_ahead_bytes: u64,
}

impl Qwen3_5Model {
    /// What the pages in hand occupy, for the ledger's slots.
    pub(super) fn expert_pages_in_hand_bytes(&self) -> ExpertPagesInHandBytes {
        ExpertPagesInHandBytes {
            held_bytes: self
                .prompt_expert_pages_in_hand
                .held_layer
                .borrow()
                .as_ref()
                .map_or(0, |held_layer| held_layer.manifest.payload_byte_count),
            read_ahead_bytes: self
                .prompt_expert_pages_in_hand
                .read_ahead_layer
                .borrow()
                .as_ref()
                .map_or(0, super::prefill_complete_layer_prefetch::PrefetchedCompleteLayer::payload_byte_count),
        }
    }

    /// Before a layer's spans: takes the layer's complete page in hand when
    /// the prompt reads it whole and the astro expert cache does not keep it.
    ///
    /// A read that fails is not fatal here. Nothing is held, and each span
    /// then reads its own experts and reports the failure if it persists.
    pub(super) fn hold_prompt_expert_layer(
        &self,
        layer_index: usize,
        spans_token_count: i32,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        self.release_held_prompt_expert_layer();
        self.discard_read_ahead_before(layer_index);
        let (Some(expert_pager), Some(astro_expert_cache)) =
            (self.expert_pager.as_ref(), self.astro_expert_cache.as_ref())
        else {
            return;
        };
        if !self.prompt_reads_complete_expert_layer(
            expert_pager,
            layer_index,
            spans_token_count,
            PartlyKeptLayers::Included,
        ) {
            return;
        }
        let Some((weights, manifest)) = self.complete_prompt_expert_layer_page(
            expert_pager,
            layer_index,
            spans_token_count,
            performance_attribution,
        ) else {
            return;
        };
        let expert_capacity = manifest.expert_ids.len();
        astro_expert_cache
            .borrow_mut()
            .record_disk_load(expert_capacity, manifest.source_manifests.len());
        if should_commit_multi_token_union_as_complete_layer(
            spans_token_count,
            true,
            self.expert_layer_state(layer_index),
            expert_capacity,
            expert_capacity,
        ) && self.adopt_held_layer_into_astro_expert_cache(
            astro_expert_cache,
            layer_index,
            &manifest,
            &weights,
        ) {
            performance_attribution.record_counter(
                PerformanceCounter::ExpertResidencyReadThroughSeatedCompletePayloadBytes,
                manifest.payload_byte_count,
            );
        }
        *self.prompt_expert_pages_in_hand.held_layer.borrow_mut() = Some(HeldPromptExpertLayer {
            layer_index,
            weights,
            manifest,
        });
    }

    /// Offers the page to the astro expert cache, which shares its arrays when the layer
    /// fits. A runtime failure leaves the layer streamed for this prompt.
    fn adopt_held_layer_into_astro_expert_cache(
        &self,
        astro_expert_cache: &std::cell::RefCell<
            crate::qwen3_5_text::expert_streaming::AstroExpertCache,
        >,
        layer_index: usize,
        manifest: &QuantizedExpertPageManifest,
        weights: &Qwen3_5PagedExpertWeights,
    ) -> bool {
        match astro_expert_cache.borrow_mut().try_adopt_complete_layer(
            &self.runtime,
            layer_index,
            &manifest.expert_ids,
            weights,
        ) {
            Ok(was_adopted) => was_adopted,
            Err(adoption_error) => {
                tracing::warn!(
                    layer_index,
                    error = %adoption_error,
                    "the astro expert cache could not adopt a prompt layer's page; the layer streams for this prompt"
                );
                false
            }
        }
    }

    /// The layer's complete page: the read-ahead when it was this layer's,
    /// and a read from storage otherwise.
    fn complete_prompt_expert_layer_page(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        spans_token_count: i32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Option<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest)> {
        if let Some(read_ahead_page) = self.take_prefetched_complete_layer(layer_index) {
            return Some(read_ahead_page);
        }
        let expert_capacity = expert_pager.layer_plan(layer_index).ok()?.expert_capacity;
        let complete_expert_ids: Vec<usize> = (0..expert_capacity).collect();
        match self.stream_expert_page_from_storage(
            expert_pager,
            layer_index,
            &complete_expert_ids,
            Qwen3_5ExpertStreamingRequestShape {
                route_token_count: spans_token_count,
                routed_expert_count: expert_capacity,
            },
            performance_attribution,
        ) {
            Ok(complete_page) => Some(complete_page),
            Err(read_error) => {
                tracing::debug!(
                    layer_index,
                    error = %read_error,
                    "could not read a prompt layer's complete page ahead of its spans"
                );
                None
            }
        }
    }

    /// After a layer's last span, and whenever a prompt stops early: lets
    /// the held page go. The spans already built still reference its arrays,
    /// so the bytes free when the last of them has been computed.
    pub fn release_held_prompt_expert_layer(&self) {
        self.prompt_expert_pages_in_hand
            .held_layer
            .borrow_mut()
            .take();
    }

    /// Serves a multi-token forward from a complete page already in hand
    /// for this layer: the page a prompt read one layer at a time holds for
    /// all of its spans, or the page a chunk forward read ahead for this one
    /// use. `Ok(None)` falls through to the retained and routed dispatch
    /// paths.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_complete_layer_page_in_hand(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        token_count: i32,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        if let Some(handled) = self.dispatch_held_prompt_expert_layer(
            layer_index,
            hidden_states,
            mixture_of_experts_weights,
            selected_indices,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )? {
            return Ok(Some(handled));
        }
        self.dispatch_prefetched_complete_layer(
            expert_pager,
            layer_index,
            token_count,
            hidden_states,
            mixture_of_experts_weights,
            selected_indices,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }

    /// Multiplies one span by the held page with its route remapped
    /// on-device, when the page in hand is this layer's. `Ok(None)` leaves
    /// the layer to the other dispatch paths.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_held_prompt_expert_layer(
        &self,
        layer_index: usize,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        let held_prompt_expert_layer = self.prompt_expert_pages_in_hand.held_layer.borrow();
        let Some(held_layer) = held_prompt_expert_layer
            .as_ref()
            .filter(|held_layer| held_layer.layer_index == layer_index)
        else {
            return Ok(None);
        };
        let expert_capacity = held_layer.manifest.expert_ids.len();
        self.defer_expert_demand(layer_index, expert_capacity, selected_indices);
        performance_attribution.record_counter(
            PerformanceCounter::RetainedRouteAssignmentMissCount,
            u64::try_from(selected_indices.element_count()).unwrap_or(u64::MAX),
        );
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &held_layer.weights,
            &held_layer.manifest,
            selected_indices,
            &held_layer.manifest.expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
        .map(Some)
    }
}

impl Qwen3_5Model {
    /// After an expert layer's last span: waits for the layer and records
    /// its spans' routes as prompt demand.
    ///
    /// A route is a slice of the table the router ranks every expert in, a
    /// kilobyte per token on a 256-expert layer, and holding the slice holds
    /// the table. Left for the wait that ends the prompt, every layer's
    /// tables stayed alive together: measured on a 10,240-token prompt,
    /// memory rose 10 MB with each layer, 0.4 GB by the last. Recorded
    /// here, they free as each layer finishes.
    ///
    /// The wait is the one the next layer's first span would make anyway. A
    /// dense model has no routes and keeps building the next layer's graph
    /// while this one computes.
    pub(super) fn record_finished_expert_layer_routes(
        &self,
        previous_layer_output: &mut Option<MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        if self.astro_expert_cache.is_none() {
            return Ok(());
        }
        self.wait_for_previous_layer(previous_layer_output, performance_attribution)?;
        self.record_deferred_expert_demand(performance_attribution);
        Ok(())
    }

    /// Reads the next streamed expert layer ahead once the first span this
    /// pass over the layer runs is submitted.
    ///
    /// That submission waited for the previous layer, so its page is gone
    /// and at most two are alive: this layer's and the one read here while
    /// this layer computes. Reading any earlier would hold three.
    pub(super) fn read_next_expert_layer_ahead_after_first_span(
        &self,
        submitted_span_index: usize,
        layer_index: usize,
        spans_token_count: i32,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        if submitted_span_index != 0 {
            return;
        }
        self.issue_prefill_complete_layer_prefetch(
            layer_index,
            spans_token_count,
            PartlyKeptLayers::Included,
            performance_attribution,
        );
    }
}
