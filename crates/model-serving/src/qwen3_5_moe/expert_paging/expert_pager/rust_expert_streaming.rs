//! Rust-led bounded expert streaming from SSD.
//!
//! Storage-to-weights boundary:
//! selected expert IDs -> bounded manifest -> memory admission -> lazy MLX arrays.
//! Prefill and decode both pass only the expert IDs this forward routed.

use crate::{
    AllocationAdmissionDecision, PerformanceAttribution, PerformanceCounter, PerformanceOperation,
};

use super::{ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5PagedExpertWeights};
use crate::expert_paging::{
    QuantizedExpertPageManifest, assemble_streaming_expert_page_tensors,
    build_quantized_expert_page_manifest_from_plan, build_streaming_expert_page_manifest,
    load_quantized_expert_page,
};
use crate::qwen3_5_moe::expert_paging::paged_expert_weights::build_paged_expert_weights;

/// Request dimensions needed to attribute one source plan without retaining routes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen3_5ExpertStreamingRequestShape {
    pub(crate) route_token_count: i32,
    pub(crate) routed_expert_count: usize,
}

impl Qwen3_5ExpertPager {
    /// Loads the exact expert IDs selected by Rust for this forward.
    pub(crate) fn load_rust_streamed_experts(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        request_shape: Qwen3_5ExpertStreamingRequestShape,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), ExpertPagingError> {
        // `layer_plan` is startup-validated immutable geometry. Building a page
        // manifest from it is pure planning and performs no model-payload I/O.
        let layer_plan = self.layer_plan(layer_index)?;
        let page_manifest = if layer_index
            < self
                .streaming_expert_pack_sources
                .as_ref()
                .map(|streaming_sources| streaming_sources.decoder_layer_count())
                .unwrap_or(0)
        {
            let streaming_sources = self
                .streaming_expert_pack_sources
                .as_ref()
                .expect("layer index checked against streaming coverage above");
            build_streaming_expert_page_manifest(
                layer_plan,
                streaming_sources.expert_file_paths(layer_index),
                expert_ids,
            )?
        } else {
            build_quantized_expert_page_manifest_from_plan(layer_plan, expert_ids)?
        };
        // Admission must happen before opening/loading source ranges. It checks
        // the exact payload against current active memory and the shared ceiling;
        // forward admission has already reclaimed experts for persistent and
        // transient growth. Loading first would spend I/O on an unusable page.
        let allocation_admission_decision = performance_attribution.measure_operation(
            PerformanceOperation::MemoryAdmissionSnapshot,
            |_performance_attribution| {
                self.memory_budget.require_admission(
                    runtime,
                    &format!("rust_streamed_expert_layer_{layer_index}"),
                    page_manifest.payload_byte_count,
                )
            },
        )?;
        if allocation_admission_decision
            == AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit
        {
            // Policy recommends cleanup; this paging owner executes the mechanism
            // and asks policy to re-evaluate the fresh counters before loading.
            performance_attribution.measure_operation(
                PerformanceOperation::MlxAllocatorCacheCleanup,
                |_performance_attribution| {
                    runtime.synchronize_gpu_stream_and_clear_allocator_cache()
                },
            )?;
            performance_attribution.measure_operation(
                PerformanceOperation::MemoryAdmissionSnapshot,
                |_performance_attribution| {
                    self.memory_budget.require_admission(
                        runtime,
                        &format!("rust_streamed_expert_layer_{layer_index}_after_cleanup"),
                        page_manifest.payload_byte_count,
                    )
                },
            )?;
        }
        let mut loaded_tensors = performance_attribution
            .measure_operation(
                PerformanceOperation::RustExpertStreamingLayerPreparation,
                |performance_attribution| {
                    load_quantized_expert_page(
                        runtime,
                        &page_manifest,
                        performance_attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(|error| ExpertPagingError::Runtime {
                description: error.to_string(),
            })?;
        let mut loaded_tensors = if self
            .streaming_expert_pack_sources
            .as_ref()
            .is_some_and(|streaming_sources| layer_index < streaming_sources.decoder_layer_count())
        {
            let assembled_tensors = assemble_streaming_expert_page_tensors(
                runtime,
                &mut loaded_tensors,
                layer_plan,
                page_manifest.expert_ids.len(),
            )
            .map_err(|error| ExpertPagingError::Runtime {
                description: error.to_string(),
            })?;
            assembled_tensors
        } else {
            loaded_tensors
        };
        // This conversion consumes named tensors from the map. Any absent weight,
        // scale, or bias fails before model execution can observe a partial page.
        let streamed_weights = build_paged_expert_weights(&mut loaded_tensors, layer_plan)?;
        performance_attribution.record_expert_streaming_source_plan(
            layer_index,
            request_shape.route_token_count,
            request_shape.routed_expert_count,
            expert_ids.len(),
            page_manifest.source_manifests.len(),
            page_manifest.payload_byte_count,
            self.streaming_expert_pack_sources
                .as_ref()
                .is_some_and(|streaming_sources| {
                    layer_index < streaming_sources.decoder_layer_count()
                }),
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

    /// Streams every expert of one layer through the per-expert packs.
    ///
    /// This is the complete-layer materialization path for streaming revisions:
    /// their retained descriptors are `.apack` pack files, not SafeTensors
    /// shards, so the shard-source resident promotion route cannot open them.
    /// The returned page holds the full `[expert_capacity, ...]` arrays in
    /// normalized expert order, ready for the identical fusion and validation
    /// the shard-path resident materialization applies.
    pub(crate) fn load_complete_streaming_expert_layer(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        layer_index: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), ExpertPagingError> {
        let layer_plan = self.layer_plan(layer_index)?;
        let complete_expert_ids: Vec<usize> = (0..layer_plan.expert_capacity).collect();
        self.load_rust_streamed_experts(
            runtime,
            layer_index,
            &complete_expert_ids,
            Qwen3_5ExpertStreamingRequestShape {
                route_token_count: 0,
                routed_expert_count: layer_plan.expert_capacity,
            },
            performance_attribution,
        )
    }
}
