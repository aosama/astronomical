//! The single evaluation that completes a paged forward, and what it found.
//!
//! A reply token can be served two ways. The exact way resolves every sparse
//! layer's route on the host before building that layer, reading whatever is
//! missing from storage; it always succeeds and costs one graphics-processor
//! wait per layer that is not held whole. The optimistic way assumes every
//! routed expert is already in memory: each layer looks its experts up in
//! its table on the graphics processor and counts the ones that were not
//! there, and those counts join the token's single evaluation. When every
//! count is zero the token is finished without one extra wait; when any is
//! not, the token's outputs are wrong and the caller runs it again the exact
//! way. Optimistic serving is what lets memory be shared across layers: an `astro_expert_cache`
//! that takes a few idle experts out of nearly every layer would otherwise
//! make nearly every layer pay the wait.

use std::cell::{Cell, RefCell};

use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};
use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

/// One layer served on the graphics processor on the assumption that its
/// table held every routed expert.
#[derive(Debug)]
pub struct OptimisticLayerServing {
    /// How many of the token's routed experts the table did not hold; lazy
    /// until the token's evaluation. `None` for a table that holds its whole
    /// layer: it cannot miss, so nothing is computed or read back for it.
    pub missing_assignment_count: Option<MlxArray>,
    pub assignment_count: usize,
    pub table_payload_bytes: u64,
    pub expert_capacity: usize,
}

/// What the layers of one optimistic forward assumed, checked when the
/// forward is evaluated.
#[derive(Debug, Default)]
pub struct PagedForwardMissingRouteCollector {
    is_optimistic: Cell<bool>,
    /// Set when a layer could not be served optimistically after all, so the
    /// forward must be run again whatever the counts say.
    has_abandoned_optimistic_serving: Cell<bool>,
    optimistic_layer_servings: RefCell<Vec<OptimisticLayerServing>>,
}

impl PagedForwardMissingRouteCollector {
    pub fn clear(&self) {
        self.has_abandoned_optimistic_serving.set(false);
        self.optimistic_layer_servings.borrow_mut().clear();
    }

    pub fn set_optimistic(&self, is_optimistic: bool) {
        self.is_optimistic.set(is_optimistic);
    }

    pub fn is_optimistic(&self) -> bool {
        self.is_optimistic.get()
    }

    pub fn abandon_optimistic_serving(&self) {
        self.has_abandoned_optimistic_serving.set(true);
    }

    pub fn record_optimistic_layer(&self, layer_serving: OptimisticLayerServing) {
        self.optimistic_layer_servings
            .borrow_mut()
            .push(layer_serving);
    }

    fn take_optimistic_layer_servings(&self) -> Vec<OptimisticLayerServing> {
        std::mem::take(&mut *self.optimistic_layer_servings.borrow_mut())
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PagedRouteValidationOutcome {
    CompleteHit,
    /// An optimistic forward routed to an expert that was not in memory: its
    /// outputs are wrong and the forward must be run again the exact way.
    MissedExperts,
}

impl Qwen3_5Model {
    pub fn clear_paged_forward_missing_route_roots(&self) {
        self.paged_forward_missing_route_collector.clear();
    }

    pub fn begin_decode_astro_expert_cache_forward(&self) {
        if let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() {
            astro_expert_cache.borrow_mut().begin_decode_forward();
        }
    }

    pub fn end_decode_astro_expert_cache_forward(&self) {
        if let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() {
            astro_expert_cache.borrow_mut().end_decode_forward();
        }
    }

    /// Evaluates completion roots after every sparse layer loaded its exact route.
    pub fn evaluate_arrays_resolving_paged_routes(
        &self,
        evaluation_arrays: &[&MlxArray],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PagedRouteValidationOutcome, TextExecutionError> {
        // Copy references, not arrays. `completion_roots` merely gives MLX one
        // explicit list of graph roots whose dependencies include every eagerly
        // loaded expert page used by this forward.
        let mut completion_roots = Vec::with_capacity(evaluation_arrays.len());
        completion_roots.extend_from_slice(evaluation_arrays);
        // Issue #542: decode route arrays join this wait so finalization never
        // pays a second graphics-processor evaluation. Attribution-disabled
        // decode has an empty collector and must not allocate here.
        let pending_route_observation = performance_attribution
            .is_enabled()
            .then(|| self.route_observation.borrow());
        if let Some(pending_route_observation) = pending_route_observation.as_ref() {
            for pending_route_array in pending_route_observation.pending_route_array_refs() {
                completion_roots.push(pending_route_array);
            }
        }
        // Routes of seated layers multiplied on-device join the same wait so
        // demand recording afterwards is a host copy, not another evaluation.
        let deferred_expert_demand = self.deferred_expert_demand.borrow();
        completion_roots.extend(deferred_expert_demand.pending_route_array_refs());
        let optimistic_layer_servings = self
            .paged_forward_missing_route_collector
            .take_optimistic_layer_servings();
        extend_with_missing_assignment_counts(&mut completion_roots, &optimistic_layer_servings);

        // Attribute the blocking MLX evaluation boundary separately from graph
        // construction. With experimental solid-state-drive paging interval 0,
        // or any fully resident model, this single wait owns the entire
        // multi-layer multi-token tape, including first-use Metal compile and
        // any memory-pressure thrash.
        let completion_root_count = completion_roots.len();
        let eval_started_at = std::time::Instant::now();
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&completion_roots),
        )?;
        let eval_elapsed = eval_started_at.elapsed();
        drop(completion_roots);
        drop(deferred_expert_demand);
        drop(pending_route_observation);
        self.conclude_evaluated_forward(
            &optimistic_layer_servings,
            eval_elapsed,
            completion_root_count,
            performance_attribution,
        )
    }

    /// Decides what an evaluated forward was worth: nothing when it was
    /// optimistic and missed, otherwise its demand, hits and queued table
    /// writes are recorded.
    fn conclude_evaluated_forward(
        &self,
        optimistic_layer_servings: &[OptimisticLayerServing],
        eval_elapsed: std::time::Duration,
        completion_root_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PagedRouteValidationOutcome, TextExecutionError> {
        if self.optimistic_forward_missed_experts(optimistic_layer_servings)? {
            // Nothing this forward learned is true: its routes came from
            // wrong hidden states after the first missing expert.
            self.deferred_expert_demand
                .borrow_mut()
                .discard_pending_layer_routes();
            self.route_observation
                .borrow_mut()
                .discard_pending_layer_route_arrays();
            self.paged_forward_missing_route_collector.clear();
            return Ok(PagedRouteValidationOutcome::MissedExperts);
        }
        self.credit_optimistic_layer_hits(optimistic_layer_servings, performance_attribution);
        self.record_deferred_expert_demand(performance_attribution);
        if eval_elapsed > std::time::Duration::from_millis(500) {
            tracing::info!(
                eval_elapsed_millis = eval_elapsed.as_millis(),
                completion_root_count,
                "slow evaluate_arrays for paged forward"
            );
        }
        let flush_started_at = std::time::Instant::now();
        let written_expert_count = self.flush_pending_expert_slot_inserts_internal()?;
        if written_expert_count > 0 {
            performance_attribution.record_counter(
                PerformanceCounter::HotExpertWarmInsertCount,
                written_expert_count,
            );
        }
        let flush_elapsed = flush_started_at.elapsed();
        if flush_elapsed > std::time::Duration::from_millis(100) {
            tracing::info!(
                flush_elapsed_millis = flush_elapsed.as_millis(),
                "slow flush_pending_expert_slot_inserts after eval"
            );
        }
        self.paged_forward_missing_route_collector.clear();
        Ok(PagedRouteValidationOutcome::CompleteHit)
    }

    /// Drains queued slot inserts and records the warm-insert and prefetch
    /// evidence. The engine calls this once per token after the forward's
    /// arrays are evaluated, so this stays the single attribution seam for
    /// both hot-expert warming and the leftover-only #537 prefetch.
    pub fn record_warm_insert_and_prefetch_statistics(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        let written_expert_count = self.flush_pending_expert_slot_inserts_internal()?;
        if written_expert_count > 0 {
            performance_attribution.record_counter(
                PerformanceCounter::HotExpertWarmInsertCount,
                written_expert_count,
            );
        }
        let decode_slot_eviction_count =
            self.astro_expert_cache
                .as_ref()
                .map_or(0, |astro_expert_cache| {
                    astro_expert_cache
                        .borrow_mut()
                        .take_decode_slot_eviction_count()
                });
        if decode_slot_eviction_count > 0 {
            performance_attribution.record_counter(
                PerformanceCounter::DecodeAstroExpertCacheSlotEvictionCount,
                decode_slot_eviction_count,
            );
        }
        let (prefetch_issue_count, prefetch_capacity_drop_count, prefetch_payload_bytes) =
            self.take_previous_token_prefetch_statistics();
        if prefetch_issue_count > 0 {
            performance_attribution.record_counter(
                PerformanceCounter::PreviousTokenPrefetchIssueCount,
                prefetch_issue_count,
            );
            performance_attribution.record_counter(
                PerformanceCounter::PreviousTokenPrefetchByteCount,
                prefetch_payload_bytes,
            );
        }
        if prefetch_capacity_drop_count > 0 {
            performance_attribution.record_counter(
                PerformanceCounter::PreviousTokenPrefetchCapacityDropCount,
                prefetch_capacity_drop_count,
            );
        }
        Ok(())
    }

    pub fn take_previous_token_prefetch_statistics(&self) -> (u64, u64, u64) {
        self.astro_expert_cache
            .as_ref()
            .map_or((0, 0, 0), |astro_expert_cache| {
                astro_expert_cache
                    .borrow_mut()
                    .take_prefetch_flush_statistics()
            })
    }

    /// Counted variant of the post-evaluation warm insert flush.
    fn flush_pending_expert_slot_inserts_internal(&self) -> Result<u64, TextExecutionError> {
        if let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() {
            return astro_expert_cache
                .borrow_mut()
                .flush_pending_inserts(&self.runtime)
                .map_err(TextExecutionError::from);
        }
        Ok(0)
    }
}

impl Qwen3_5Model {
    /// Whether an optimistic forward routed to any expert its tables did not
    /// hold. The counts were roots of the evaluation that just finished, so
    /// reading them is a host copy, not another wait.
    fn optimistic_forward_missed_experts(
        &self,
        optimistic_layer_servings: &[OptimisticLayerServing],
    ) -> Result<bool, TextExecutionError> {
        if self
            .paged_forward_missing_route_collector
            .has_abandoned_optimistic_serving
            .get()
        {
            return Ok(true);
        }
        for optimistic_layer_serving in optimistic_layer_servings {
            let Some(missing_assignment_count) =
                optimistic_layer_serving.missing_assignment_count.as_ref()
            else {
                continue;
            };
            let missing_assignment_counts = missing_assignment_count.to_vec_u32()?;
            if missing_assignment_counts
                .iter()
                .any(|missing_assignment_count| *missing_assignment_count > 0)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Counts the hits of an optimistic forward once it is known to have
    /// been right, so a forward that is run again is never counted twice.
    fn credit_optimistic_layer_hits(
        &self,
        optimistic_layer_servings: &[OptimisticLayerServing],
        performance_attribution: &mut PerformanceAttribution,
    ) {
        for optimistic_layer_serving in optimistic_layer_servings {
            performance_attribution.record_counter(
                PerformanceCounter::RetainedRouteAssignmentHitCount,
                u64::try_from(optimistic_layer_serving.assignment_count).unwrap_or(u64::MAX),
            );
            self.record_resident_layer_decode_hit(
                optimistic_layer_serving.assignment_count,
                optimistic_layer_serving.table_payload_bytes,
                optimistic_layer_serving.expert_capacity,
                performance_attribution,
            );
        }
    }
}

/// Adds each partly held layer's miss count to the roots of the token's
/// evaluation, so checking them afterwards is a host copy.
fn extend_with_missing_assignment_counts<'roots>(
    completion_roots: &mut Vec<&'roots MlxArray>,
    optimistic_layer_servings: &'roots [OptimisticLayerServing],
) {
    for optimistic_layer_serving in optimistic_layer_servings {
        if let Some(missing_assignment_count) =
            optimistic_layer_serving.missing_assignment_count.as_ref()
        {
            completion_roots.push(missing_assignment_count);
        }
    }
}

#[cfg(test)]
#[path = "paged_route_resolution_tests.rs"]
mod paged_route_resolution_tests;
