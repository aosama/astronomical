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
//! A layer the cache keeps only part of is held like any other: the whole
//! page is read once and its hot table is left alone unless the residency
//! plan lets the cache adopt the complete layer in its place.

use mlx_bridge::MlxArray;

use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::prefill_complete_layer_prefetch::PartlyKeptLayers;
use crate::expert_paging::QuantizedExpertPageManifest;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::{
    Qwen3_5ExpertPager, Qwen3_5ExpertStreamingRequestShape, Qwen3_5PagedExpertWeights,
};
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{
    PerformanceAttribution, PerformanceCounter, should_commit_multi_token_union_as_complete_layer,
};

/// The complete page of the layer a layer-at-a-time prompt is working on.
#[derive(Debug)]
pub(crate) struct HeldPromptExpertLayer {
    layer_index: usize,
    weights: Qwen3_5PagedExpertWeights,
    manifest: QuantizedExpertPageManifest,
}

/// The complete expert pages a prompt forward has in hand: the layer it is
/// working on, and the layer read ahead while that one computes.
#[derive(Debug, Default)]
pub(crate) struct PromptExpertPagesInHand {
    pub(super) held_layer: std::cell::RefCell<Option<HeldPromptExpertLayer>>,
    pub(super) read_ahead_layer:
        std::cell::RefCell<Option<super::prefill_complete_layer_prefetch::PrefetchedCompleteLayer>>,
}

impl Qwen3_5Model {
    /// Before a layer's spans: takes the layer's complete page in hand when
    /// the prompt reads it whole and the cache does not keep it.
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
        let (Some(expert_pager), Some(retained_experts)) =
            (self.expert_pager.as_ref(), self.retained_experts.as_ref())
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
        retained_experts
            .borrow_mut()
            .record_disk_load(expert_capacity, manifest.source_manifests.len());
        if should_commit_multi_token_union_as_complete_layer(
            spans_token_count,
            true,
            self.expert_residency_target(layer_index),
            expert_capacity,
            expert_capacity,
        ) && self.adopt_held_layer_into_cache(retained_experts, layer_index, &manifest, &weights)
        {
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

    /// Offers the page to the cache, which shares its arrays when the layer
    /// fits. A runtime failure leaves the layer streamed for this prompt.
    fn adopt_held_layer_into_cache(
        &self,
        retained_experts: &std::cell::RefCell<
            crate::qwen3_5_streaming::expert_streaming::WarmExpertSlotTableCache,
        >,
        layer_index: usize,
        manifest: &QuantizedExpertPageManifest,
        weights: &Qwen3_5PagedExpertWeights,
    ) -> bool {
        match retained_experts.borrow_mut().try_adopt_complete_layer(
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
                    "the expert cache could not adopt a prompt layer's page; the layer streams for this prompt"
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
    pub(crate) fn release_held_prompt_expert_layer(&self) {
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
    ) -> Result<Option<MlxArray>, Qwen3_5ExecutionError> {
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
    ) -> Result<Option<MlxArray>, Qwen3_5ExecutionError> {
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
