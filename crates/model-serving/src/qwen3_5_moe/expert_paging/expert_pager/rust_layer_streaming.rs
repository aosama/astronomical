//! Rust-led bounded expert layer streaming.
//!
//! This is the complete storage-to-page orchestration boundary:
//!
//! `validated layer plan -> exact selected IDs -> bounded manifest -> memory
//! admission -> lazy MLX arrays -> typed Qwen expert weights`.
//!
//! It intentionally serves two execution shapes through one exact mechanism:
//! multi-token prefill passes every expert ID and receives a complete temporary
//! layer, while one-token decode passes only router-selected top-K IDs. Neither
//! shape changes quantization, expert order, or source precision.

use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

use super::{ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5PagedExpertWeights};
use crate::expert_paging::{
    QuantizedExpertPageManifest, build_quantized_expert_page_manifest_from_plan,
    load_quantized_expert_page,
};
use crate::qwen3_5_moe::expert_paging::paged_expert_weights::build_paged_expert_weights;

impl Qwen3_5ExpertPager {
    /// Loads one exact page selected by Rust. Multi-token callers pass all expert
    /// identifiers and therefore stream one complete layer; decode passes top-K.
    pub(crate) fn load_rust_streamed_expert_layer(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), ExpertPagingError> {
        // `layer_plan` is startup-validated immutable geometry. Building a page
        // manifest from it is pure planning and performs no model-payload I/O.
        let layer_plan = self.layer_plan(layer_index)?;
        let page_manifest = build_quantized_expert_page_manifest_from_plan(layer_plan, expert_ids)?;
        // Admission must happen before opening/loading source ranges. It reserves
        // the exact payload against current active memory and the forward reserve
        // published by adaptive admission. Loading first would discover pressure
        // only after spending I/O and constructing arrays that cannot be used.
        performance_attribution.measure_operation(
            PerformanceOperation::MemoryAdmissionSnapshot,
            |_performance_attribution| {
                self.memory_budget.check(
                    runtime,
                    &format!("rust_streamed_expert_layer_{layer_index}"),
                    page_manifest.payload_byte_count,
                )
            },
        )?;
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
        let streamed_weights = build_paged_expert_weights(&mut loaded_tensors, layer_plan)?;
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
