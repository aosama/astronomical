//! Reads the next streamed sparse layer while the graphics processor works.
//!
//! Expert pages are lazy MLX loads on the central processing unit stream, so a
//! streamed prefill layer used to read only after its router indices reached
//! the host: one pipeline drain plus one storage read per streamed layer per
//! chunk, with the graphics processor idle for the read. A multi-token chunk
//! large enough to route every expert reads the complete layer anyway, and the
//! complete page needs no route, so it can be issued as soon as the previous
//! layer's graph is built. MLX then reads it on its own stream while earlier
//! layers compute, and the layer multiplies the complete page with its route
//! remapped on-device — no host route copy at all.
//!
//! At most one prefetched layer is in flight. Its bytes are charged to the
//! prefill plan as a second complete-layer stream slot, so the prefetch never
//! spends memory the expert budget handed to retained layers.
//!
//! Layers whose page would need a transient decode (jangtq2 codebook layers in
//! any multi-token chunk) are excluded from layer geometry before a page is
//! planned. Planning one and dropping it afterwards did no storage read, but it
//! still ran memory admission (often draining the pipeline to clear the
//! allocator cache) and booked a phantom full-layer source plan per streamed
//! layer per chunk, which made every codebook layer look streamed twice.

use mlx_c_rust::MlxArray;

use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use crate::expert_paging::{
    ExpertManifestError, QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::{
    Qwen3_5ExpertPager, Qwen3_5ExpertStreamingRequestShape, Qwen3_5PagedExpertWeights,
    codebook_page_multiplies_without_transient_decode,
};
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::qwen3_5_streaming::model::expert_reuse::ExpertPageDisposition;
use crate::{
    ExpertLayerResidencyTarget, PerformanceAttribution, PerformanceCounter,
    should_commit_multi_token_union_as_complete_layer,
};

/// Expected-coverage factor for "this chunk routes every expert".
///
/// A chunk of `T` tokens choosing `k` of `N` experts leaves an expert unrouted
/// with probability `(1 - k/N)^T ≈ e^(-T·k/N)`. At `T·k ≥ 8·N` the expected
/// count of unrouted experts is `N·e^-8`, under one expert for any layer of up
/// to 2,980 experts, so the routed union and the complete layer are the same
/// read and prefetching the complete layer wastes no storage bandwidth. That
/// bound assumes uniform routing; a skewed router leaves more experts unrouted,
/// so the read ahead may fetch a few experts the chunk never multiplies.
const COMPLETE_UNION_ASSIGNMENTS_PER_EXPERT: usize = 8;

/// Prefill RAM reserved for streaming complete expert layers.
///
/// One slot holds the layer being multiplied at its forward-time peak
/// (packed source plus any decoded output). The second holds the one layer
/// read ahead by the prefetch, which is only issued for pages multiplied
/// without a transient decode, so it never holds more than the packed source.
#[must_use]
pub(crate) const fn prefill_complete_layer_stream_slot_bytes(
    forward_time_expert_page_peak_bytes: u64,
    largest_complete_expert_source_payload_bytes: u64,
) -> u64 {
    forward_time_expert_page_peak_bytes.saturating_add(largest_complete_expert_source_payload_bytes)
}

/// Whether a `route_token_count`-token chunk is expected to route every expert.
#[must_use]
pub(crate) fn prefill_chunk_routes_complete_layer(
    route_token_count: i32,
    experts_per_token: usize,
    expert_capacity: usize,
) -> bool {
    let Ok(route_token_count) = usize::try_from(route_token_count) else {
        return false;
    };
    route_token_count > 1
        && expert_capacity > 0
        && route_token_count.saturating_mul(experts_per_token)
            >= expert_capacity.saturating_mul(COMPLETE_UNION_ASSIGNMENTS_PER_EXPERT)
}

/// Whether the residency plan lets a layer's complete page be read early.
///
/// Release targets are shrinking that layer's memory, so an early read would
/// fight the plan; every other target either streams or seats the layer.
#[must_use]
pub(crate) fn residency_target_allows_prefill_prefetch(
    residency_target: Option<ExpertLayerResidencyTarget>,
) -> bool {
    !matches!(
        residency_target,
        Some(
            ExpertLayerResidencyTarget::ReleasePartial
                | ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
        )
    )
}

/// Whether a layer's complete page could be multiplied as read, judged from
/// layer geometry alone so an unusable prefetch is never planned.
///
/// Mirrors `Qwen3_5PagedExpertWeights::multiplies_complete_page_without_transient_decode`:
/// affine and bfloat16 layers are always eligible for the read ahead, codebook
/// layers only under the shared codebook rule.
#[must_use]
pub(crate) fn layer_plan_allows_prefill_prefetch(
    layer_plan: &QuantizedExpertLayerPlan,
    route_token_count: i32,
    has_fused_codebook_decode_kernel: bool,
) -> bool {
    !layer_plan.stores_jangtq2_codebook_projection()
        || codebook_page_multiplies_without_transient_decode(
            route_token_count,
            has_fused_codebook_decode_kernel,
        )
}

/// Largest packed complete-layer source the prefill read ahead can hold.
///
/// Sizes the read-ahead stream slot from only the layers
/// [`layer_plan_allows_prefill_prefetch`] can ever pick for a multi-token
/// chunk, so a bundle whose layers are all codebook-encoded reserves nothing
/// for a prefetch that never runs instead of keeping a whole layer's bytes
/// out of the expert budget.
pub(crate) fn largest_prefill_prefetchable_complete_source_payload_bytes(
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<u64, ExpertManifestError> {
    // Any multi-token count gives the same answer: the codebook rule only
    // admits single decode tokens, and prefill chunks always route several.
    const MULTI_TOKEN_PREFILL_CHUNK_TOKEN_COUNT: i32 = 2;
    layer_plans
        .iter()
        .filter(|layer_plan| {
            layer_plan_allows_prefill_prefetch(
                layer_plan,
                MULTI_TOKEN_PREFILL_CHUNK_TOKEN_COUNT,
                true,
            )
        })
        .try_fold(0_u64, |largest_source_layer_bytes, layer_plan| {
            Ok(largest_source_layer_bytes.max(layer_plan.complete_expert_payload_byte_count()?))
        })
}

/// The nearest later layer the prefetch should read, if any.
#[must_use]
pub(crate) fn next_prefill_prefetch_layer_index(
    completed_layer_index: usize,
    layer_count: usize,
    mut is_prefetch_candidate: impl FnMut(usize) -> bool,
) -> Option<usize> {
    (completed_layer_index.saturating_add(1)..layer_count)
        .find(|&layer_index| is_prefetch_candidate(layer_index))
}

/// One complete layer whose lazy read MLX has already been asked to run.
#[derive(Debug)]
pub(crate) struct PrefetchedCompleteLayer {
    layer_index: usize,
    weights: Qwen3_5PagedExpertWeights,
    manifest: QuantizedExpertPageManifest,
}

impl Qwen3_5Model {
    /// Drops a prefetched layer the current forward will not consume.
    pub(crate) fn discard_prefill_complete_layer_prefetch(&self) {
        self.prefill_complete_layer_prefetch.borrow_mut().take();
    }

    /// Takes the prefetched complete page when it belongs to `layer_index`.
    pub(super) fn take_prefetched_complete_layer(
        &self,
        layer_index: usize,
    ) -> Option<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest)> {
        let mut prefetch_slot = self.prefill_complete_layer_prefetch.borrow_mut();
        if prefetch_slot
            .as_ref()
            .is_none_or(|prefetched| prefetched.layer_index != layer_index)
        {
            return None;
        }
        prefetch_slot
            .take()
            .map(|prefetched| (prefetched.weights, prefetched.manifest))
    }

    /// Issues the next streamed layer's complete read after `completed_layer_index`.
    ///
    /// Failures are never fatal: the layer's ordinary mandatory read still
    /// runs when the prefetch is skipped, refused by memory admission, or
    /// unusable without a transient decode.
    pub(super) fn issue_prefill_complete_layer_prefetch(
        &self,
        completed_layer_index: usize,
        route_token_count: i32,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        if self.prefill_complete_layer_prefetch.borrow().is_some() {
            return;
        }
        let (Some(expert_pager), Some(retained_experts)) =
            (self.expert_pager.as_ref(), self.retained_experts.as_ref())
        else {
            return;
        };
        let experts_per_token = usize::try_from(self.config.experts_per_token()).unwrap_or(0);
        let has_fused_codebook_decode_kernel =
            self.fused_jangtq2_gathered_projection_kernel.is_some();
        let layer_plans = expert_pager.layer_plans();
        let Some(prefetch_layer_index) = next_prefill_prefetch_layer_index(
            completed_layer_index,
            layer_plans.len(),
            |layer_index| {
                let expert_capacity = layer_plans[layer_index].expert_capacity;
                prefill_chunk_routes_complete_layer(
                    route_token_count,
                    experts_per_token,
                    expert_capacity,
                ) && layer_plan_allows_prefill_prefetch(
                    &layer_plans[layer_index],
                    route_token_count,
                    has_fused_codebook_decode_kernel,
                ) && matches!(
                    self.expert_page_disposition(layer_index, expert_capacity),
                    ExpertPageDisposition::Miss
                ) && retained_experts.borrow().occupied_slot_count(layer_index) == 0
                    && residency_target_allows_prefill_prefetch(
                        self.expert_residency_target(layer_index),
                    )
            },
        ) else {
            return;
        };
        let expert_capacity = layer_plans[prefetch_layer_index].expert_capacity;
        let complete_expert_ids: Vec<usize> = (0..expert_capacity).collect();
        let (weights, manifest) = match self.stream_expert_page_from_storage(
            expert_pager,
            prefetch_layer_index,
            &complete_expert_ids,
            Qwen3_5ExpertStreamingRequestShape {
                route_token_count,
                routed_expert_count: expert_capacity,
            },
            performance_attribution,
        ) {
            Ok(complete_page) => complete_page,
            Err(prefetch_error) => {
                tracing::debug!(
                    layer_index = prefetch_layer_index,
                    error = %prefetch_error,
                    "skipped prefill complete-layer prefetch"
                );
                return;
            }
        };
        // The geometry filter above already excluded these pages; this guard
        // only keeps a future encoding from being multiplied with a decode the
        // stream slot does not reserve. A lazy page dropped here reads nothing.
        if !weights.multiplies_complete_page_without_transient_decode(
            route_token_count,
            has_fused_codebook_decode_kernel,
        ) {
            tracing::warn!(
                layer_index = prefetch_layer_index,
                "prefill prefetch planned a page that needs a transient decode"
            );
            return;
        }
        let mut page_arrays = Vec::new();
        weights.append_array_references(&mut page_arrays);
        if let Err(submission_error) = self.runtime.async_eval_arrays(&page_arrays) {
            tracing::debug!(
                layer_index = prefetch_layer_index,
                error = %submission_error,
                "skipped prefill complete-layer prefetch submission"
            );
            return;
        }
        performance_attribution.record_counter(
            PerformanceCounter::PrefillCompleteLayerPrefetchPayloadBytes,
            manifest.payload_byte_count,
        );
        *self.prefill_complete_layer_prefetch.borrow_mut() = Some(PrefetchedCompleteLayer {
            layer_index: prefetch_layer_index,
            weights,
            manifest,
        });
    }

    /// Multiplies a prefetched complete page with the route remapped on-device.
    ///
    /// The complete read is the chunk's routed union, so the plan may seat it
    /// exactly as it would seat a full-coverage union; the route feeds demand
    /// learning after the forward's single completion evaluation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_with_prefetched_complete_layer(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_token_count: i32,
        prefetched_weights: Qwen3_5PagedExpertWeights,
        prefetched_manifest: QuantizedExpertPageManifest,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
        let complete_expert_ids: Vec<usize> = (0..expert_capacity).collect();
        if let Some(retained_experts) = self.retained_experts.as_ref() {
            retained_experts
                .borrow_mut()
                .record_disk_load(expert_capacity, prefetched_manifest.source_manifests.len());
            if should_commit_multi_token_union_as_complete_layer(
                route_token_count,
                true,
                self.expert_residency_target(layer_index),
                expert_capacity,
                expert_capacity,
            ) && retained_experts.borrow_mut().try_adopt_complete_layer(
                &self.runtime,
                layer_index,
                &complete_expert_ids,
                &prefetched_weights,
            )? {
                performance_attribution.record_counter(
                    PerformanceCounter::ExpertResidencyReadThroughSeatedCompletePayloadBytes,
                    prefetched_manifest.payload_byte_count,
                );
            }
        }
        self.defer_expert_demand(layer_index, expert_capacity, selected_indices);
        performance_attribution.record_counter(
            PerformanceCounter::RetainedRouteAssignmentMissCount,
            u64::try_from(selected_indices.element_count()).unwrap_or(u64::MAX),
        );
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &prefetched_weights,
            &prefetched_manifest,
            selected_indices,
            &complete_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }
}
