//! Exact paged forward completion after every layer route was resolved eagerly.
//!
//! Older paging designs could build a forward with unresolved expert routes and
//! replay after discovering misses. The Rust streaming path resolves each sparse
//! layer before constructing its expert computation, so completion has only one
//! valid outcome: every route was a complete hit against the page selected for
//! that layer. The small compatibility types below keep that invariant explicit
//! at call sites without reviving replay state.

use std::sync::atomic::{AtomicBool, Ordering};

use astronomical_runtime_integration::MlxRuntime;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};
use astronomical_mlx_c_rust::MlxArray;

/// Set after the first capture attempt so at most one prefill chunk is traced.
static METAL_CAPTURE_ATTEMPTED: AtomicBool = AtomicBool::new(false);

/// Starts the optional one-shot MLX Metal capture for the upcoming evaluation.
///
/// Returns whether a capture is active so the caller can pair it with
/// [`end_optional_metal_capture`]. Inert unless `ASTRONOMICAL_METAL_CAPTURE_PATH`
/// is set; if Metal's capture layer is not present the start fails, is logged,
/// and execution continues unchanged.
fn begin_optional_metal_capture(runtime: &MlxRuntime) -> bool {
    let Ok(metal_capture_path) = std::env::var("ASTRONOMICAL_METAL_CAPTURE_PATH") else {
        return false;
    };
    if METAL_CAPTURE_ATTEMPTED.swap(true, Ordering::SeqCst) {
        return false;
    }
    match runtime.start_metal_capture(&metal_capture_path) {
        Ok(()) => {
            tracing::info!(metal_capture_path, "MLX Metal capture started");
            true
        }
        Err(capture_error) => {
            tracing::warn!(
                metal_capture_path,
                "MLX Metal capture did not start; run the worker with MTL_CAPTURE_ENABLED=1: {capture_error}"
            );
            false
        }
    }
}

/// Stops the capture started by [`begin_optional_metal_capture`] when active.
fn end_optional_metal_capture(runtime: &MlxRuntime, is_capturing: bool) {
    if !is_capturing {
        return;
    }
    if let Err(capture_error) = runtime.stop_metal_capture() {
        tracing::warn!("MLX Metal capture did not stop cleanly: {capture_error}");
    }
}

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

        // Attribute the blocking MLX evaluation boundary separately from graph
        // construction. With experimental solid-state-drive paging interval 0,
        // or any fully resident model, this single wait owns the entire
        // multi-layer multi-token tape, including first-use Metal compile and
        // any memory-pressure thrash.
        //
        // Optional one-shot Metal capture: when the worker runs with
        // `MTL_CAPTURE_ENABLED=1` and `ASTRONOMICAL_METAL_CAPTURE_PATH` is set,
        // the first chunk-terminal evaluation is captured to an Xcode
        // `.gputrace` bundle. It is inert otherwise, so it can never change a
        // serving request's result.
        let is_capturing_this_eval = begin_optional_metal_capture(&self.runtime);
        let eval_started_at = std::time::Instant::now();
        let eval_result = performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| self.runtime.evaluate_arrays(&completion_roots),
        );
        end_optional_metal_capture(&self.runtime, is_capturing_this_eval);
        eval_result?;
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

    /// Drains queued slot inserts and records the warm-insert and prefetch
    /// evidence. The engine calls this once per token after the forward's
    /// arrays are evaluated, so this stays the single attribution seam for
    /// both hot-expert warming and the leftover-only #537 prefetch.
    pub(crate) fn record_warm_insert_and_prefetch_statistics(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let written_expert_count = self.flush_pending_expert_slot_inserts_internal()?;
        if written_expert_count > 0 {
            performance_attribution.record_counter(
                PerformanceCounter::HotExpertWarmInsertCount,
                written_expert_count,
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

    pub(crate) fn take_previous_token_prefetch_statistics(&self) -> (u64, u64, u64) {
        self.retained_experts
            .as_ref()
            .map_or((0, 0, 0), |retained_experts| {
                retained_experts
                    .borrow_mut()
                    .take_prefetch_flush_statistics()
            })
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
