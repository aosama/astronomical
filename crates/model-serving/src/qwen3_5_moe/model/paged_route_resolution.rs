//! Exact paged forward completion after every layer route was resolved eagerly.
//!
//! Older paging designs could build a forward with unresolved expert routes and
//! replay after discovering misses. The Rust streaming path resolves each sparse
//! layer before constructing its expert computation, so completion has only one
//! valid outcome: every route was a complete hit against the page selected for
//! that layer. The small compatibility types below keep that invariant explicit
//! at call sites without reviving replay state.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

#[derive(Debug, Default)]
pub(crate) struct PagedForwardMissingRouteCollector;

impl PagedForwardMissingRouteCollector {
    /// Intentionally a no-op: eager route resolution cannot accumulate misses.
    pub(crate) const fn clear(&self) {}
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum PagedRouteValidationOutcome {
    CompleteHit,
}

impl Qwen3_5Model {
    pub(crate) fn clear_paged_forward_missing_route_roots(&self) {
        self.paged_forward_missing_route_collector.clear();
    }

    /// Evaluates completion roots after every sparse layer loaded its exact route.
    pub(crate) fn evaluate_arrays_resolving_paged_routes(
        &self,
        evaluation_arrays: &[&MlxArray],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PagedRouteValidationOutcome, Qwen3_5ExecutionError> {
        // Copy references, not arrays. `completion_roots` merely gives MLX one
        // explicit list of graph roots whose dependencies include every eagerly
        // loaded expert page used by this forward.
        let mut completion_roots = Vec::with_capacity(evaluation_arrays.len());
        completion_roots.extend_from_slice(evaluation_arrays);

        // Attribute the blocking MLX evaluation boundary separately from graph
        // construction. With experimental solid-state-drive paging interval 0,
        // or any fully resident model, this single wait owns the entire
        // multi-layer multi-token tape, including first-use Metal compile and
        // any memory-pressure thrash.
        let eval_started_at = std::time::Instant::now();
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&completion_roots),
        )?;
        let eval_elapsed = eval_started_at.elapsed();
        if eval_elapsed > std::time::Duration::from_millis(500) {
            tracing::info!(
                eval_elapsed_millis = eval_elapsed.as_millis(),
                completion_root_count = completion_roots.len(),
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

    /// Writes queued miss experts into the slot table after GPU evaluation so
    /// `slice_update` can donate instead of copying a live gather buffer, and
    /// returns how many experts were newly written.
    pub(crate) fn flush_pending_expert_slot_inserts(&self) -> Result<u64, Qwen3_5ExecutionError> {
        self.flush_pending_expert_slot_inserts_internal()
    }

    /// Counted variant of the post-evaluation warm insert flush.
    fn flush_pending_expert_slot_inserts_internal(&self) -> Result<u64, Qwen3_5ExecutionError> {
        if let Some(retained_experts) = self.retained_experts.as_ref() {
            return retained_experts
                .borrow_mut()
                .flush_pending_inserts(&self.runtime)
                .map_err(Qwen3_5ExecutionError::from);
        }
        Ok(0)
    }
}
