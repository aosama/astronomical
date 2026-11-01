//! Sparse routing selects retained experts or SSD streaming.
//!
//! # One token's journey through this dispatch
//!
//! The match below is the whole SSD-paging decision tree. Every outcome names
//! the serving function it lands in, so a reader can trace any layer's forward
//! from here without hunting across directories:
//!
//! 1. **Seated complete layer, page multiplies without a transient decode** —
//!    affine or bfloat16 pages for any chunk, codebook pages for a single
//!    decode token with the fused kernel. The whole layer is served from the
//!    warm slot tables with the route remapped on-device: zero host copies,
//!    zero SSD reads, and demand recorded after the forward's single
//!    completion wait (`deferred_expert_demand.rs`).
//! 2. **Seated complete codebook layer that would decode the table** — the
//!    chunk's routed union is gathered as a subset of the seated layer
//!    (`routed_subset_of_cached_page`, this file's sibling `expert_reuse.rs`).
//! 3. **Multi-token miss (prefill chunk)** — the chunk's routed-expert union is
//!    streamed from SSD as an operation-local page
//!    (`stream_operation_local_routed_experts` in `expert_reuse.rs`; loading in
//!    `expert_streaming/expert_pager/expert_page_streaming.rs`).
//! 4. **Single-token miss, mixed coverage** — some routed experts are warm,
//!    some are not: covered assignments are served from the slot tables and
//!    only the missing experts stream from SSD
//!    (`forward_moe_mixed_decode_route_with_performance_attribution` in
//!    `mixed_decode_execution.rs`, built on
//!    `cached_plus_streamed_page_route.rs`).
//! 5. **Single-token miss, fully warm** — the hot-expert cache hit serves the
//!    whole routed set from RAM (`hot_expert_partial_page` in
//!    `expert_reuse.rs`); a total miss falls through to outcome 3's streaming.
//!
//! The warm tables themselves live in
//! `expert_streaming/warm_expert_slot_table_cache.rs`; the byte plans and the
//! bounded SSD reader are family-neutral under `crate::expert_paging` and
//! `astronomical-runtime-integration`'s safetensors module.

use mlx_c_rust::MlxArray;

use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

use super::super::expert_streaming::expert_pager::Qwen3_5ExpertPager;
use super::Qwen3_5MoEPagedExecutionMode;
use super::expert_reuse::ExpertPageDisposition;
use super::feed_forward_weights::{Qwen3_5MoEFeedForwardWeights, Qwen3_5MoERouterGateWeights};
use super::routing;

impl Qwen3_5Model {
    /// Routes once, then selects contiguous resident arrays or native paging.
    // Paged execution inputs stay explicit rather than introducing a parameter facade.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_qwen3_5_moe(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        should_use_compiled_elementwise_graphs: bool,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let token_count = paged_token_count(&hidden_states.shape())?;
        let (selected_indices, selected_scores) = performance_attribution.measure_operation(
            PerformanceOperation::PagedRouterGraphConstruction,
            |_performance_attribution| {
                self.routed_expert_selection(hidden_states, mixture_of_experts_weights)
            },
        )?;

        let should_use_loaded_model_mode = matches!(
            paged_execution_mode,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault
        );
        self.retain_decode_route_if_observed(
            should_use_loaded_model_mode,
            token_count,
            layer_index,
            &selected_indices,
            performance_attribution.is_enabled(),
        );
        // A seated complete layer whose page multiplies without a transient
        // decode (affine or bfloat16 pages for any chunk, codebook pages for a
        // decode token with the fused kernel) is served straight from the
        // table with the route remapped on-device. No router indices reach
        // the host and no routed-subset copy is gathered, so the forward keeps
        // one graphics-processor wait instead of one per layer; at 40 layers
        // the per-layer route copy cost decode about 20 ms per token. The
        // route still feeds demand learning, deferred to the forward's single
        // completion evaluation. Codebook pages that would decode the whole
        // table fall through to the host-side route and decode only their
        // routed union.
        let decode_tokens_multiply_packed_pages =
            self.fused_jangtq2_gathered_projection_kernel.is_some();
        if should_use_loaded_model_mode
            && token_count > 1
            && let Some(handled) = self.dispatch_prefetched_complete_layer(
                expert_pager,
                layer_index,
                token_count,
                hidden_states,
                mixture_of_experts_weights,
                &selected_indices,
                &selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            )?
        {
            return Ok(handled);
        }
        if should_use_loaded_model_mode
            && self.retained_experts.is_some()
            && let Some(handled) = self.dispatch_retained_full_hit_packed_page(
                expert_pager,
                layer_index,
                token_count,
                decode_tokens_multiply_packed_pages,
                &selected_indices,
                &selected_scores,
                mixture_of_experts_weights,
                hidden_states,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            )?
        {
            return Ok(handled);
        }

        let selected_expert_ids = self.materialize_selected_route(
            should_use_loaded_model_mode,
            token_count,
            &selected_indices,
            performance_attribution,
        )?;
        let sorted_unique_expert_ids = selected_expert_ids.as_ref().map(|selected_expert_ids| {
            let mut routed_expert_ids = selected_expert_ids.clone();
            routed_expert_ids.sort_unstable();
            routed_expert_ids.dedup();
            routed_expert_ids
        });
        if should_use_loaded_model_mode
            && let Some(selected_expert_ids) = selected_expert_ids.as_ref()
            && let Some(routed_expert_ids) = sorted_unique_expert_ids.as_ref()
            && self.retained_experts.is_some()
            && let Some(handled) = self.dispatch_routed_disposition(
                expert_pager,
                layer_index,
                token_count,
                decode_tokens_multiply_packed_pages,
                &selected_indices,
                &selected_scores,
                selected_expert_ids,
                routed_expert_ids,
                mixture_of_experts_weights,
                hidden_states,
                paged_execution_mode,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            )?
        {
            return Ok(handled);
        }

        self.forward_unhandled_route(
            hidden_states,
            mixture_of_experts_weights,
            expert_pager,
            layer_index,
            token_count,
            &selected_indices,
            &selected_scores,
            should_use_compiled_elementwise_graphs,
            paged_execution_mode,
            sorted_unique_expert_ids.as_deref(),
            performance_attribution,
        )
    }

    /// Retains this decode token's route lazily for observation (issues
    /// #536/#542): the array stays lazy until the decode evaluation that
    /// materializes logits; finalization only copies host identifiers. The
    /// trunk-layer bound covers every decoder layer routed through this
    /// forward.
    fn retain_decode_route_if_observed(
        &self,
        should_use_loaded_model_mode: bool,
        token_count: i32,
        layer_index: usize,
        selected_indices: &MlxArray,
        attribution_enabled: bool,
    ) {
        if should_use_loaded_model_mode
            && token_count == 1
            && layer_index < self.config.layer_count() as usize
            && attribution_enabled
        {
            self.route_observation
                .borrow_mut()
                .retain_layer_route(layer_index, selected_indices);
        }
    }

    /// Serves a multi-token chunk from a prefetched complete layer when one
    /// is staged for this layer. `Ok(None)` falls through to the retained
    /// and routed dispatch paths.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_prefetched_complete_layer(
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
        if let Some((prefetched_weights, prefetched_manifest)) =
            self.take_prefetched_complete_layer(layer_index)
        {
            return Ok(Some(self.forward_moe_with_prefetched_complete_layer(
                hidden_states,
                mixture_of_experts_weights,
                expert_pager,
                layer_index,
                token_count,
                prefetched_weights,
                prefetched_manifest,
                selected_indices,
                selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            )?));
        }
        Ok(None)
    }

    /// The retained-table fast path: a seated complete layer whose packed page
    /// multiplies the whole table without a transient decode is served from
    /// the table with the route remapped on-device. `Ok(None)` falls through
    /// to the routed-dispatch paths.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_retained_full_hit_packed_page(
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
    ) -> Result<Option<MlxArray>, Qwen3_5ExecutionError> {
        let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
        if !matches!(
            self.expert_page_disposition(layer_index, expert_capacity),
            ExpertPageDisposition::FullHit,
        ) {
            return Ok(None);
        }
        let complete_expert_ids: Vec<usize> = (0..expert_capacity).collect();
        if let Some((packed_weights, packed_manifest)) =
            self.cached_packed_page(layer_index, &complete_expert_ids, expert_capacity)
            && packed_weights.multiplies_complete_page_without_transient_decode(
                token_count,
                decode_tokens_multiply_packed_pages,
            )
        {
            self.defer_expert_demand(layer_index, expert_capacity, selected_indices);
            performance_attribution.record_counter(
                PerformanceCounter::RetainedRouteAssignmentHitCount,
                u64::try_from(selected_indices.element_count()).unwrap_or(u64::MAX),
            );
            performance_attribution.record_counter(
                PerformanceCounter::AvoidedCompleteLayerExpertSourcePayloadBytes,
                packed_manifest.payload_byte_count,
            );
            self.record_avoided_source_bytes_for_paged_session(packed_manifest.payload_byte_count);
            return Ok(Some(self.forward_moe_paged_with_performance_attribution(
                hidden_states,
                mixture_of_experts_weights,
                &packed_weights,
                &packed_manifest,
                selected_indices,
                &complete_expert_ids,
                selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            )?));
        }
        Ok(None)
    }

    /// Session counters stay always-on, including when attribution diagnostics
    /// are off, so a retained full hit still records the source bytes it avoided.
    fn record_avoided_source_bytes_for_paged_session(&self, payload_byte_count: u64) {
        if let Some(expert_pager) = self.expert_pager.as_ref() {
            expert_pager.record_session_avoided_source_bytes(payload_byte_count);
        }
    }

    /// Materializes the routed expert ids on the host when the loaded-model
    /// mode needs them; decode tokens track their per-layer synchronization
    /// separately because materializing can evaluate pending MLX work.
    /// `Ok(None)` when the diagnostic modes run without host routes.
    fn materialize_selected_route(
        &self,
        should_use_loaded_model_mode: bool,
        token_count: i32,
        selected_indices: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<Vec<usize>>, Qwen3_5ExecutionError> {
        if !should_use_loaded_model_mode {
            return Ok(None);
        }
        let selected_expert_ids = if token_count == 1 {
            // Streamed decode must know the exact route before loading its
            // weights. Materializing that route can evaluate pending MLX
            // work, so track this per-layer synchronization separately.
            performance_attribution.measure_operation(
                PerformanceOperation::DecodeExpertRouteHostMaterialization,
                |performance_attribution| {
                    self.copy_selected_expert_ids(selected_indices, Some(performance_attribution))
                },
            )?
        } else {
            self.copy_selected_expert_ids(selected_indices, None)?
        };
        Ok(Some(selected_expert_ids))
    }

    /// Records the routed demand on the retained tables and dispatches on the
    /// layer's page disposition; both dispositions are terminal here.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_routed_disposition(
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
        if let Some(retained_experts) = self.retained_experts.as_ref() {
            retained_experts.borrow_mut().record_expert_demand(
                layer_index,
                expert_capacity,
                selected_expert_ids,
            );
        }
        match self.expert_page_disposition(layer_index, expert_capacity) {
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
                .routed_subset_of_cached_page(layer_index, routed_expert_ids, expert_capacity)?
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "cache hit reported a page that vanished",
                })?;
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
            .cached_packed_page(layer_index, &complete_expert_ids, expert_capacity)
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "cache hit reported a page that vanished",
            })?;
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

    /// Serves a table miss: multi-token chunks stream the chunk's routed-expert
    /// union; decode tokens classify warm-table coverage first (mixed, hot
    /// cache, or cold stream).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_miss_route(
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
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        performance_attribution.record_counter(
            PerformanceCounter::RetainedRouteAssignmentMissCount,
            u64::try_from(selected_expert_ids.len()).unwrap_or(u64::MAX),
        );
        let (packed_weights, packed_manifest) = if token_count > 1 {
            // Multi-token chunks stream the chunk's routed-expert
            // union, not the complete layer: the union is bounded
            // by chunk tokens × experts per token, which for every
            // chunk smaller than expert_capacity ÷ experts per
            // token is a fraction of the complete layer's read and
            // decode work. The union page stays operation-local
            // (warm tables fill from decode tokens), and the
            // forward admission reserves the same union bound
            // (issue #1422).
            self.stream_operation_local_routed_experts(
                expert_pager,
                layer_index,
                token_count,
                routed_expert_ids,
                paged_execution_mode == Qwen3_5MoEPagedExecutionMode::ProductionDefault,
                performance_attribution,
            )?
        } else {
            // Issue #373: classify warm-table coverage before
            // dispatch. A fully warm route serves from retained
            // RAM; a mixed route splits between the retained
            // rows and a storage read of only the missing
            // experts; a fully cold route streams the whole
            // routed set. Classifying first keeps each of those
            // costs measurable and picks the dispatch.
            let route_coverage = self.record_decode_route_coverage(
                layer_index,
                routed_expert_ids,
                performance_attribution,
            );
            if !route_coverage.retained_expert_ids.is_empty()
                && !route_coverage.missing_expert_ids.is_empty()
            {
                // Mixed serving: serve the covered assignments from
                // retained RAM and stream only the missing experts.
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
            if let Some((cached_weights, cached_manifest)) = self.hot_expert_partial_page(
                layer_index,
                expert_capacity,
                routed_expert_ids,
                performance_attribution,
            ) {
                // Hot-expert cache hit: the warm table covers every
                // routed expert of this token, so the routed set is
                // served from retained RAM with no storage read
                // (issue #372).
                (cached_weights, cached_manifest)
            } else {
                self.stream_operation_local_routed_experts(
                    expert_pager,
                    layer_index,
                    token_count,
                    routed_expert_ids,
                    paged_execution_mode == Qwen3_5MoEPagedExecutionMode::ProductionDefault,
                    performance_attribution,
                )?
            }
        };
        // The streamed page covers exactly the routed union, so
        // the gather remap uses the routed expert ids in both the
        // multi-token and decode paths. Complete-table full hits
        // returned earlier with their own complete id set.
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

    /// Serves every route the retained fast paths declined: multi-token
    /// chunks pick their configured paging mode, decode tokens fall through
    /// to expert-store paging.
    #[allow(clippy::too_many_arguments)]
    fn forward_unhandled_route(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        token_count: i32,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
        sorted_unique_expert_ids: Option<&[usize]>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        if token_count > 1 {
            return match paged_execution_mode {
                Qwen3_5MoEPagedExecutionMode::ProductionDefault => self
                    .forward_moe_with_expert_store_paging(
                        hidden_states,
                        mixture_of_experts_weights,
                        expert_pager,
                        layer_index,
                        token_count,
                        selected_indices,
                        selected_scores,
                        should_use_compiled_elementwise_graphs,
                        paged_execution_mode,
                        sorted_unique_expert_ids,
                        performance_attribution,
                    ),
                Qwen3_5MoEPagedExecutionMode::CompactPromptDiagnostic => self
                    .forward_moe_with_direct_prefill_paging(
                        hidden_states,
                        mixture_of_experts_weights,
                        expert_pager,
                        layer_index,
                        token_count,
                        selected_indices,
                        selected_scores,
                        should_use_compiled_elementwise_graphs,
                        performance_attribution,
                    ),
                Qwen3_5MoEPagedExecutionMode::TokenLocalDiagnostic => self
                    .forward_moe_with_per_token_paging(
                        hidden_states,
                        mixture_of_experts_weights,
                        expert_pager,
                        layer_index,
                        selected_indices,
                        selected_scores,
                        should_use_compiled_elementwise_graphs,
                        performance_attribution,
                    ),
            };
        }
        self.forward_moe_with_expert_store_paging(
            hidden_states,
            mixture_of_experts_weights,
            expert_pager,
            layer_index,
            token_count,
            selected_indices,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            paged_execution_mode,
            sorted_unique_expert_ids,
            performance_attribution,
        )
    }

    /// Runs the router gate and top-k expert routing for one MoE forward,
    /// returning the selected expert indices and their scores.
    fn routed_expert_selection(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
    ) -> Result<(MlxArray, MlxArray), Qwen3_5ExecutionError> {
        let router_logits = match &mixture_of_experts_weights.router_projection {
            Qwen3_5MoERouterGateWeights::Affine(quantized_weights) => {
                self.quantized_linear(hidden_states, quantized_weights)?
            }
            Qwen3_5MoERouterGateWeights::Unquantized(unquantized_weight) => {
                let transposed_gate_weight =
                    self.runtime.transpose_axes(unquantized_weight, &[1, 0])?;
                self.runtime
                    .matmul(hidden_states, &transposed_gate_weight)?
            }
        };
        routing::qwen3_5_moe_route_experts(
            &self.runtime,
            &router_logits,
            self.config.experts_per_token() as i32,
            self.config.normalizes_top_k_probabilities(),
        )
        .map_err(Qwen3_5ExecutionError::from)
    }
}

/// Pure: the paged token count is the second-to-last shape dimension with the
/// hidden dimension trailing it; a single-dimension shape falls back to index
/// zero and an empty shape is invalid input.
fn paged_token_count(hidden_state_shape: &[i32]) -> Result<i32, Qwen3_5ExecutionError> {
    hidden_state_shape
        .get(hidden_state_shape.len().saturating_sub(2))
        .copied()
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "paged MoE hidden states must include token and hidden dimensions",
        })
}

#[cfg(test)]
#[path = "forward_tests.rs"]
mod forward_tests;
