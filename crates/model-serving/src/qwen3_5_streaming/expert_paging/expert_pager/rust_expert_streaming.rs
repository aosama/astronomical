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
use crate::qwen3_5_streaming::expert_paging::paged_expert_weights;

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
        let page_manifest = build_quantized_expert_page_manifest_from_plan(layer_plan, expert_ids)?;
        // Admission must happen before opening/loading source ranges. It checks
        // the exact payload against current active memory and the shared ceiling;
        // forward admission has already reclaimed experts for persistent and
        // transient growth. Loading first would spend I/O on an unusable page.
        let decoded_codebook_weight_shapes =
            layer_plan.jangtq2_decoded_weight_page_shapes(expert_ids.len())?;
        let decode_workspace_tensor_shapes =
            layer_plan.jangtq2_decode_workspace_page_shapes(expert_ids.len())?;
        let output_assembly_copy_count =
            layer_plan.jangtq2_decoded_output_assembly_copy_count(expert_ids.len());
        let decoder_index_tensor_shapes = decoded_codebook_weight_shapes
            .iter()
            .map(|decoded_weight_shape| {
                vec![
                    decoded_weight_shape[2]
                        .max(decoded_weight_shape[0].saturating_mul(decoded_weight_shape[1])),
                ]
            })
            .collect::<Vec<_>>();
        // The memory owner calculates persistent bfloat16 output storage plus
        // the largest one-block decode workspace and the decoder's small
        // auxiliary arrays. The packed page remains live while it is decoded.
        let pending_page_allocation_bytes = crate::memory::expert_page_peak_allocation_bytes(
            page_manifest.payload_byte_count,
            &decoded_codebook_weight_shapes,
            &decode_workspace_tensor_shapes,
            output_assembly_copy_count,
            size_of::<u16>(),
            crate::jangtq2_qwen35::mlx_decode::MAXIMUM_SIMULTANEOUS_DECODE_TEMPORARY_ARRAY_COUNT,
            size_of::<u32>(),
            &decoder_index_tensor_shapes,
            crate::jangtq2_qwen35::mlx_decode::JANGTQ2_DECODE_AUXILIARY_ARRAY_COUNT,
            size_of::<u32>(),
        )
        .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
            description: "expert page plus jangtq2 decode workspace byte count overflowed"
                .to_owned(),
        })?;
        let allocation_admission_decision = performance_attribution.measure_operation(
            PerformanceOperation::MemoryAdmissionSnapshot,
            |_performance_attribution| {
                self.memory_budget.require_admission(
                    runtime,
                    &format!("rust_streamed_expert_layer_{layer_index}"),
                    pending_page_allocation_bytes,
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
                        pending_page_allocation_bytes,
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
        // This conversion consumes named tensors from the map. Any absent weight,
        // scale, or bias fails before model execution can observe a partial page.
        let streamed_weights = paged_expert_weights::build_paged_expert_weights(
            runtime,
            &mut loaded_tensors,
            layer_plan,
        )?;
        performance_attribution.record_expert_streaming_source_plan(
            layer_index,
            request_shape.route_token_count,
            request_shape.routed_expert_count,
            expert_ids.len(),
            page_manifest.source_manifests.len(),
            page_manifest.payload_byte_count,
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
}
