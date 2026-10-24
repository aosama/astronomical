//! Selects retained or streamed expert pages for mixture-of-experts forwards.

use crate::expert_paging::{ExpertWeightPage, QuantizedExpertPageManifest};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::expert_streaming::RoutedExpertCoverage;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::{
    Qwen3_5ExpertPager, Qwen3_5ExpertStreamingRequestShape, Qwen3_5PagedExpertWeights,
};
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{
    PerformanceAttribution, PerformanceCounter, should_commit_mandatory_complete_layer,
    should_commit_mandatory_routed_page, should_commit_multi_token_union_as_complete_layer,
    should_complete_partial_layer_on_prefill_read, should_seed_partial_layer_on_prefill_read,
};

/// How one forward should execute one mixture-of-experts layer.
pub(crate) enum ExpertPageDisposition {
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

    /// Classifies one decode token's warm-table route coverage and records it.
    ///
    /// Issue #373: the all-or-nothing rule serves a token from retained RAM only
    /// when every routed expert is warm, so a partially covered token still pays
    /// a full storage read. Classifying before dispatch makes that cost
    /// measurable and is the prerequisite for deciding whether mixed serving is
    /// worth its added forward path. Both decode dispatch sites call this so
    /// there is one classification answer.
    pub(super) fn record_decode_route_coverage(
        &self,
        layer_index: usize,
        routed_expert_ids: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) -> RoutedExpertCoverage {
        let route_coverage = self.retained_experts.as_ref().map_or_else(
            || RoutedExpertCoverage {
                retained_expert_ids: Vec::new(),
                missing_expert_ids: routed_expert_ids.to_vec(),
            },
            |retained_experts| {
                let route_coverage = retained_experts
                    .borrow()
                    .routed_expert_coverage(layer_index, routed_expert_ids);
                let (prefetch_hit_count, prefetch_miss_count) = retained_experts
                    .borrow_mut()
                    .consume_prefetch_coverage(layer_index, &route_coverage);
                if prefetch_hit_count > 0 {
                    performance_attribution.record_counter(
                        PerformanceCounter::PreviousTokenPrefetchHitCount,
                        prefetch_hit_count,
                    );
                }
                if prefetch_miss_count > 0 {
                    performance_attribution.record_counter(
                        PerformanceCounter::PreviousTokenPrefetchMissCount,
                        prefetch_miss_count,
                    );
                }
                route_coverage
            },
        );
        if route_coverage.missing_expert_ids.is_empty() {
            performance_attribution
                .record_counter(PerformanceCounter::HotExpertRouteFullyCoveredCount, 1);
        } else if route_coverage.retained_expert_ids.is_empty() {
            performance_attribution
                .record_counter(PerformanceCounter::HotExpertRouteFullyMissedCount, 1);
        } else {
            performance_attribution
                .record_counter(PerformanceCounter::HotExpertRoutePartiallyCoveredCount, 1);
        }
        performance_attribution.record_counter(
            PerformanceCounter::HotExpertRouteRetainedAssignmentCount,
            u64::try_from(route_coverage.retained_expert_ids.len()).unwrap_or(u64::MAX),
        );
        performance_attribution.record_counter(
            PerformanceCounter::HotExpertRouteMissingAssignmentCount,
            u64::try_from(route_coverage.missing_expert_ids.len()).unwrap_or(u64::MAX),
        );
        route_coverage
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
        cache.full_coverage_page(layer_index, complete_expert_ids, expert_capacity)
    }

    /// Returns the routed experts of a seated complete layer as their own
    /// page, for a forward that decodes the page it multiplies.
    ///
    /// A multi-token chunk (or a decode token on a worker without the fused
    /// kernel) decodes the packed page it multiplies, so handing it the
    /// complete table would decode every expert of the layer (about 4.8 GB
    /// on the 122B bundle) whatever the route size — which defeats prefill's
    /// chunk halving, whose premise is that a smaller chunk needs less
    /// memory. The subset page holds exactly the routed union and its
    /// manifest maps those global ids to slots 0.., so the caller gathers
    /// with `routed_expert_ids` exactly as it does for a streamed union.
    pub(super) fn routed_subset_of_cached_page(
        &self,
        layer_index: usize,
        routed_expert_ids: &[usize],
        expert_capacity: usize,
    ) -> Result<
        Option<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest)>,
        Qwen3_5ExecutionError,
    > {
        let Some((complete_page, complete_manifest)) =
            self.cached_packed_page(layer_index, routed_expert_ids, expert_capacity)
        else {
            return Ok(None);
        };
        let page_slots = routed_expert_ids
            .iter()
            .map(|&expert_id| {
                complete_manifest
                    .page_slot_by_global_expert_id
                    .get(expert_id)
                    .copied()
                    .filter(|slot| *slot != u32::MAX)
                    .map(|slot| slot as usize)
                    .ok_or(Qwen3_5ExecutionError::InvalidInput {
                        description: "a seated complete layer lost a routed expert's slot",
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let subset_page = complete_page
            .select_experts(&self.runtime, &page_slots)
            .map_err(|error| {
                tracing::warn!(
                    layer_index,
                    %error,
                    "the routed subset of a seated layer could not be built"
                );
                Qwen3_5ExecutionError::InvalidInput {
                    description: "the routed subset of a seated layer could not be built",
                }
            })?;
        let mut page_slot_by_global_expert_id = vec![u32::MAX; expert_capacity];
        for (subset_slot, &expert_id) in routed_expert_ids.iter().enumerate() {
            if let Some(slot) = page_slot_by_global_expert_id.get_mut(expert_id) {
                *slot = u32::try_from(subset_slot).unwrap_or(u32::MAX);
            }
        }
        let subset_manifest = QuantizedExpertPageManifest {
            expert_ids: routed_expert_ids.to_vec(),
            page_slot_by_global_expert_id,
            source_manifests: Vec::new(),
            payload_byte_count: subset_page.resident_payload_byte_count(),
        };
        Ok(Some((subset_page, subset_manifest)))
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
        let (streamed_weights, streamed_manifest) = self.stream_expert_page_from_storage(
            expert_pager,
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
                if seated {
                    performance_attribution.record_counter(
                        PerformanceCounter::ExpertResidencyReadThroughSeatedCompletePayloadBytes,
                        streamed_manifest.payload_byte_count,
                    );
                }
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

    /// Streams the unique routed experts for one forward without displacing a
    /// seated complete layer. A decode token's page is then offered for
    /// hot-expert retention so a later token can hit the warm table instead
    /// of re-reading storage (issue #372); the offer honors the active
    /// residency plan, so a layer the plan streams operation-local or is
    /// releasing stays unwarmed.
    ///
    /// A cold multi-token union seeds a retained table only when the plan funds
    /// a complete layer; existing partial tables use their missing-expert
    /// growth path. Prefill pages stay off the decode warm-insert queue, which
    /// previously pinned every chunk's union, including failed attempts, until
    /// generation started.
    pub(super) fn stream_operation_local_routed_experts(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_token_count: i32,
        routed_expert_ids: &[usize],
        production_default_paging: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), Qwen3_5ExecutionError>
    {
        let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
        // Multi-token chunks consult the warm table before touching storage:
        // streaming only the route's missing experts and growing the table
        // turns every later chunk into a table hit. Streaming the whole union
        // re-read experts already in RAM on every chunk — the 2026-10-23
        // flights held ~17 GB of partial tables yet still read 27.8 GB per
        // request. When the plan funds the layer's complete claim, the read
        // extends to every non-resident expert so the table converges to a
        // complete layer in one pass instead of a few experts per chunk.
        if route_token_count > 1 {
            if let Some(warm_table_page) = self.serve_multi_token_route_from_warm_table(
                expert_pager,
                layer_index,
                route_token_count,
                expert_capacity,
                production_default_paging,
                routed_expert_ids,
                performance_attribution,
            )? {
                return Ok(warm_table_page);
            }
        }
        let (streamed_weights, streamed_manifest) = self.stream_expert_page_from_storage(
            expert_pager,
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
            let warm_slot_count = self.hot_expert_warm_slot_count();
            let (resident_expert_count, layer_has_complete_table) = {
                let retained_experts = retained_experts.borrow();
                (
                    retained_experts.occupied_slot_count(layer_index),
                    retained_experts.has_complete_layer(layer_index, expert_capacity),
                )
            };
            // A multi-token chunk whose union covers the whole layer has
            // already paid the complete-layer read cost; offering it to
            // retained ownership is pure reuse. Without this offer every
            // later chunk re-read the same bytes from storage — the
            // 2026-10-23 Ornith flight streamed 389 GB for one 23,976-token
            // prompt while 26.95 GB of ceiling sat unused. Budget admission
            // inside the cache remains the only refusal path.
            if !layer_has_complete_table
                && should_commit_multi_token_union_as_complete_layer(
                    route_token_count,
                    production_default_paging,
                    self.expert_residency_target(layer_index),
                    routed_expert_ids.len(),
                    expert_capacity,
                )
            {
                let seated = retained_experts.borrow_mut().try_adopt_complete_layer(
                    &self.runtime,
                    layer_index,
                    routed_expert_ids,
                    &streamed_weights,
                )?;
                tracing::debug!(
                    layer_index,
                    seated,
                    routed_expert_count = routed_expert_ids.len(),
                    expert_capacity,
                    streamed_payload_bytes = streamed_manifest.payload_byte_count,
                    "multi-token union offered to retained ownership",
                );
                if seated {
                    performance_attribution.record_counter(
                        PerformanceCounter::ExpertResidencyReadThroughSeatedCompletePayloadBytes,
                        streamed_manifest.payload_byte_count,
                    );
                }
            } else if should_seed_partial_layer_on_prefill_read(
                route_token_count,
                production_default_paging,
                self.expert_residency_target(layer_index),
                resident_expert_count,
                routed_expert_ids.len(),
                expert_capacity,
            ) {
                let inserted_expert_count = retained_experts.borrow_mut().insert_streamed_experts(
                    &self.runtime,
                    layer_index,
                    routed_expert_ids,
                    &streamed_weights,
                    routed_expert_ids,
                    expert_capacity,
                )?;
                tracing::debug!(
                    layer_index,
                    inserted_expert_count,
                    routed_expert_count = routed_expert_ids.len(),
                    expert_capacity,
                    streamed_payload_bytes = streamed_manifest.payload_byte_count,
                    "multi-token prefill union offered to retained ownership as a completion seed",
                );
            }
            if route_token_count == 1 && warm_slot_count > 0 && !layer_has_complete_table {
                if should_commit_mandatory_routed_page(
                    route_token_count,
                    production_default_paging,
                    self.expert_residency_target(layer_index),
                    false,
                ) {
                    retained_experts
                        .borrow_mut()
                        .queue_pending_routed_expert_insert(
                            layer_index,
                            routed_expert_ids,
                            &streamed_weights,
                            warm_slot_count.min(expert_capacity),
                        )?;
                } else if performance_attribution.is_enabled() {
                    // Issue #537: keep this token's streamed route in leftover
                    // slots when the residency plan would have dropped it.
                    retained_experts
                        .borrow_mut()
                        .queue_pending_prefetch_insert(
                            layer_index,
                            routed_expert_ids,
                            &streamed_weights,
                            routed_expert_ids.len().min(expert_capacity),
                        )?;
                }
            }
        }
        Ok((streamed_weights, streamed_manifest))
    }

    /// Serves a multi-token routed union from the warm table, streaming only
    /// what the table lacks. Returns `None` when the layer has no table (the
    /// caller owns cold-union seeding or adoption) or when budget
    /// admission refused enough table growth that the route still is not
    /// covered (the caller falls back to the full union stream).
    #[allow(clippy::too_many_arguments)]
    fn serve_multi_token_route_from_warm_table(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_token_count: i32,
        expert_capacity: usize,
        production_default_paging: bool,
        routed_expert_ids: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<
        Option<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest)>,
        Qwen3_5ExecutionError,
    > {
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return Ok(None);
        };
        let Some(missing_route_expert_ids) = retained_experts
            .borrow()
            .missing_route_expert_ids(layer_index, routed_expert_ids)
        else {
            return Ok(None);
        };
        if missing_route_expert_ids.is_empty() {
            // The table already covers the whole union: serve the routed
            // subset from RAM. The subset page — not the whole table — is
            // the contract multi-token decoding needs, because the chunk
            // materializes exactly the page it multiplies; handing it the
            // full table would decode every resident expert whatever the
            // route size.
            retained_experts
                .borrow_mut()
                .record_routed_reads(layer_index, routed_expert_ids);
            if let Some(subset_page) =
                self.routed_subset_of_cached_page(layer_index, routed_expert_ids, expert_capacity)?
            {
                performance_attribution.record_counter(
                    PerformanceCounter::AvoidedCompleteLayerExpertSourcePayloadBytes,
                    subset_page.1.payload_byte_count,
                );
                return Ok(Some(subset_page));
            }
            return Ok(None);
        }
        let resident_expert_count = retained_experts.borrow().occupied_slot_count(layer_index);
        let read_expert_ids = if should_complete_partial_layer_on_prefill_read(
            production_default_paging,
            self.expert_residency_target(layer_index),
            resident_expert_count,
            expert_capacity,
        ) {
            retained_experts
                .borrow()
                .non_resident_expert_ids(layer_index, expert_capacity)
                .unwrap_or_else(|| missing_route_expert_ids.clone())
        } else {
            missing_route_expert_ids.clone()
        };
        let (streamed_weights, streamed_manifest) = self.stream_expert_page_from_storage(
            expert_pager,
            layer_index,
            &read_expert_ids,
            Qwen3_5ExpertStreamingRequestShape {
                route_token_count,
                routed_expert_count: read_expert_ids.len(),
            },
            performance_attribution,
        )?;
        retained_experts.borrow_mut().record_disk_load(
            read_expert_ids.len(),
            streamed_manifest.source_manifests.len(),
        );
        // Growth is bounded by the cache ceiling and the structural slot
        // count; the expert capacity is the largest honest slot bound, and
        // passing it deliberately rides the warm-insert counter for
        // completion reads — the growth cap and the insert-evidence counter
        // share this parameter, and refusing growth entirely to keep the
        // metric pure would defeat the completion this path exists for.
        // The routed experts are protected so completing the table can never
        // evict a resident expert this forward is about to serve.
        let inserted_expert_count = retained_experts.borrow_mut().insert_streamed_experts(
            &self.runtime,
            layer_index,
            &read_expert_ids,
            &streamed_weights,
            routed_expert_ids,
            expert_capacity,
        )?;
        if inserted_expert_count > 0
            && retained_experts
                .borrow()
                .has_complete_layer(layer_index, expert_capacity)
        {
            performance_attribution.record_counter(
                PerformanceCounter::ExpertResidencyReadThroughSeatedCompletePayloadBytes,
                streamed_manifest.payload_byte_count,
            );
        }
        if let Some(subset_page) = {
            retained_experts
                .borrow_mut()
                .record_routed_reads(layer_index, routed_expert_ids);
            self.routed_subset_of_cached_page(layer_index, routed_expert_ids, expert_capacity)?
        } {
            performance_attribution.record_counter(
                PerformanceCounter::AvoidedCompleteLayerExpertSourcePayloadBytes,
                subset_page.1.payload_byte_count,
            );
            return Ok(Some(subset_page));
        }
        // Budget admission refused part of the growth, so the table still
        // misses part of the route; the caller's full union stream stays
        // correct and merely re-reads what this pass could not seat.
        tracing::debug!(
            layer_index,
            inserted_expert_count,
            missing_route_expert_count = missing_route_expert_ids.len(),
            "warm-table growth could not cover the multi-token route"
        );
        Ok(None)
    }

    /// Returns the warm-table page when it covers every routed expert of this
    /// decode token (hot-expert cache hit), counting the served reads for the
    /// least-frequently-used eviction order. Complete layers are excluded:
    /// they were already served by the upstream full-hit disposition.
    pub(super) fn hot_expert_partial_page(
        &self,
        layer_index: usize,
        expert_capacity: usize,
        routed_expert_ids: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Option<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest)> {
        let retained_experts = self.retained_experts.as_ref()?;
        {
            let cache = retained_experts.borrow();
            if cache.has_complete_layer(layer_index, expert_capacity) {
                return None;
            }
            let full_coverage_page =
                cache.full_coverage_page(layer_index, routed_expert_ids, expert_capacity)?;
            drop(cache);
            retained_experts
                .borrow_mut()
                .record_routed_reads(layer_index, routed_expert_ids);
            performance_attribution
                .record_counter(PerformanceCounter::HotExpertPartialRouteHitCount, 1);
            Some(full_coverage_page)
        }
    }
}
