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
use crate::{PerformanceAttribution, PerformanceOperation};

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
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&completion_roots),
        )?;
        self.paged_forward_missing_route_collector.clear();
        Ok(PagedRouteValidationOutcome::CompleteHit)
    }
}
