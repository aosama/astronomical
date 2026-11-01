//! Rust-led bounded expert streaming from SSD.
//!
//! Storage-to-weights boundary:
//! selected expert IDs -> bounded manifest -> memory admission -> lazy MLX arrays.
//! Prefill and decode both pass only the expert IDs this forward routed.

use crate::{
    AllocationAdmissionDecision, PerformanceAttribution, PerformanceCounter, PerformanceOperation,
};
use std::mem::size_of;

use super::{ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5PagedExpertWeights};
use crate::expert_paging::{
    QuantizedExpertPageManifest, build_quantized_expert_page_manifest_from_plan,
    load_quantized_expert_page,
};
use crate::memory::{
    SessionStreamingPhase, StreamedPageSourcePlan, record_streamed_page_source_plan_in_both_ledgers,
};
use crate::qwen3_5_streaming::expert_streaming::paged_expert_weights;

/// Request dimensions needed to attribute one source plan without retaining routes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen3_5ExpertStreamingRequestShape {
    pub(crate) route_token_count: i32,
    pub(crate) routed_expert_count: usize,
}

impl Qwen3_5ExpertPager {
    /// Loads the exact expert IDs selected by Rust for this forward.
    ///
    /// `serving_phase` is the engine's real phase state — never derived from
    /// the route's token count here, because speculative and multi-token
    /// decode forwards exist (issue #1715).
    pub(crate) fn stream_selected_expert_page_from_ssd(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        request_shape: Qwen3_5ExpertStreamingRequestShape,
        serving_phase: SessionStreamingPhase,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), ExpertPagingError> {
        // `layer_plan` is startup-validated immutable geometry. Building a page
        // manifest from it is pure planning and performs no model-payload I/O.
        let layer_plan = self.layer_plan(layer_index)?;
        let page_manifest = build_quantized_expert_page_manifest_from_plan(layer_plan, expert_ids)?;
        let decoded_codebook_weight_shapes =
            layer_plan.jangtq2_decoded_weight_page_shapes(expert_ids.len())?;
        let pending_page_allocation_bytes = pending_streamed_page_allocation_bytes(
            page_manifest.payload_byte_count,
            &decoded_codebook_weight_shapes,
            request_shape.route_token_count,
            self.decode_tokens_multiply_packed_pages(),
        )?;
        require_streamed_page_admission(
            runtime,
            self,
            layer_index,
            pending_page_allocation_bytes,
            performance_attribution,
        )?;
        let expert_read_metrics =
            self.expert_read_metrics_for_streamed_page(performance_attribution);
        // The physical-read baseline must precede the first expert read so the
        // session's physical figure covers exactly the expert-streaming era.
        self.ensure_session_process_io_baseline();
        let read_elapsed_before_load_nanoseconds = self
            .session_read_metrics
            .snapshot()
            .total_read_elapsed_nanoseconds;
        let mut loaded_tensors = performance_attribution
            .measure_operation(
                PerformanceOperation::RustExpertStreamingLayerPreparation,
                |_performance_attribution| {
                    load_quantized_expert_page(
                        runtime,
                        &page_manifest,
                        Some(std::sync::Arc::clone(&expert_read_metrics)),
                    )
                },
            )
            .map_err(|error| ExpertPagingError::Runtime {
                description: error.to_string(),
            })?;
        // This conversion consumes named tensors from the map. Any absent weight,
        // scale, or bias fails before model execution can observe a partial page.
        let streamed_weights =
            paged_expert_weights::build_paged_expert_weights(&mut loaded_tensors, layer_plan)?;
        self.record_streamed_page_session_evidence(
            layer_index,
            request_shape,
            serving_phase,
            &page_manifest,
            read_elapsed_before_load_nanoseconds,
            performance_attribution,
        );
        // This counter is logical selected payload. Positional-read counters and
        // process-attributed physical I/O answer different questions and remain
        // separate in performance reports.
        performance_attribution.record_counter(
            PerformanceCounter::RustExpertStreamingPayloadByteCount,
            page_manifest.payload_byte_count,
        );
        Ok((streamed_weights, page_manifest))
    }

    /// Records one loaded page in both ledgers and in the phase-scoped
    /// read-time window. One seam, so the attribution counters, the session
    /// phase bytes, and the session read seconds cannot drift apart.
    fn record_streamed_page_session_evidence(
        &self,
        layer_index: usize,
        request_shape: Qwen3_5ExpertStreamingRequestShape,
        serving_phase: SessionStreamingPhase,
        page_manifest: &QuantizedExpertPageManifest,
        read_elapsed_before_load_nanoseconds: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let streamed_page_source_plan = StreamedPageSourcePlan {
            layer_index,
            route_token_count: request_shape.route_token_count,
            serving_phase,
            routed_expert_count: request_shape.routed_expert_count,
            streamed_expert_count: page_manifest.expert_ids.len(),
            source_shard_count: page_manifest.source_manifests.len(),
            payload_byte_count: page_manifest.payload_byte_count,
        };
        let read_elapsed_after_load_nanoseconds = self
            .session_read_metrics
            .snapshot()
            .total_read_elapsed_nanoseconds;
        let page_read_elapsed_nanoseconds = read_elapsed_after_load_nanoseconds
            .saturating_sub(read_elapsed_before_load_nanoseconds);
        let mut session_counters = self.session_counters.borrow_mut();
        session_counters
            .record_phase_source_read_elapsed(serving_phase, page_read_elapsed_nanoseconds);
        record_streamed_page_source_plan_in_both_ledgers(
            &streamed_page_source_plan,
            performance_attribution,
            &mut session_counters,
        );
    }

    /// Prefer the attribution's private per-request aggregate so diagnostics
    /// report deltas stay exact, and install the session aggregate as its
    /// mirror peer so the always-on observatory totals see the same reads.
    /// When diagnostics are off the session aggregate feeds the readers directly.
    fn expert_read_metrics_for_streamed_page(
        &self,
        performance_attribution: &PerformanceAttribution,
    ) -> std::sync::Arc<astronomical_runtime_integration::PositionalFileReadMetrics> {
        match performance_attribution.positional_file_read_metrics() {
            Some(attribution_read_metrics) => {
                attribution_read_metrics
                    .install_peer_read_metrics(self.session_expert_read_metrics());
                attribution_read_metrics
            }
            None => self.session_expert_read_metrics(),
        }
    }
}

/// Pure: the peak allocation this page must be admitted against.
///
/// A single decode token's page is multiplied in packed form by the fused
/// gathered projection kernel, so only its source bytes are admitted. Every
/// other page (multi-token prefill, or a worker whose kernel probe failed)
/// decodes a transient bfloat16 copy at forward time: the memory owner then
/// prices the persistent bfloat16 output with the packed page live throughout.
fn pending_streamed_page_allocation_bytes(
    page_payload_byte_count: u64,
    decoded_codebook_weight_shapes: &[Vec<usize>],
    route_token_count: i32,
    decode_tokens_multiply_packed_pages: bool,
) -> Result<u64, ExpertPagingError> {
    let page_is_multiplied_packed = route_token_count == 1 && decode_tokens_multiply_packed_pages;
    if page_is_multiplied_packed {
        return Ok(page_payload_byte_count);
    }
    crate::memory::expert_page_peak_allocation_bytes(
        page_payload_byte_count,
        decoded_codebook_weight_shapes,
        size_of::<u16>(),
    )
    .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
        description: "expert page plus jangtq2 decoded output byte count overflowed".to_owned(),
    })
}

/// Admission must happen before opening/loading source ranges. It checks the
/// exact payload against current active memory and the shared ceiling; forward
/// admission has already reclaimed experts for persistent and transient
/// growth. Loading first would spend I/O on an unusable page.
fn require_streamed_page_admission(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    expert_pager: &Qwen3_5ExpertPager,
    layer_index: usize,
    pending_page_allocation_bytes: u64,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), ExpertPagingError> {
    let allocation_admission_decision = performance_attribution.measure_operation(
        PerformanceOperation::MemoryAdmissionSnapshot,
        |_performance_attribution| {
            expert_pager.memory_budget.require_admission(
                runtime,
                &format!("rust_streamed_expert_layer_{layer_index}"),
                pending_page_allocation_bytes,
            )
        },
    )?;
    if allocation_admission_decision != AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit {
        return Ok(());
    }
    // Policy recommends cleanup; this paging owner executes the mechanism
    // and asks policy to re-evaluate the fresh counters before loading.
    performance_attribution.measure_operation(
        PerformanceOperation::MlxAllocatorCacheCleanup,
        |_performance_attribution| runtime.synchronize_gpu_stream_and_clear_allocator_cache(),
    )?;
    performance_attribution.measure_operation(
        PerformanceOperation::MemoryAdmissionSnapshot,
        |_performance_attribution| {
            expert_pager.memory_budget.require_admission(
                runtime,
                &format!("rust_streamed_expert_layer_{layer_index}_after_cleanup"),
                pending_page_allocation_bytes,
            )
        },
    )?;
    Ok(())
}
