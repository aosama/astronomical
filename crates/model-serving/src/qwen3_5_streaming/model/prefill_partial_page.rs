//! Builds one prefill page from warm-table rows plus experts already streamed.
//!
//! When the ceiling refuses to finish a partial layer, the misses for this
//! chunk are already in memory. Re-reading the whole routed union from the
//! file would load experts the table is holding. Concatenating the two sides
//! keeps the single-page prefill forward, which decodes exactly the page it
//! multiplies, so the page contains only the routed union.

use crate::expert_paging::{ExpertWeightPage, QuantizedExpertPageManifest};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;
use crate::{PerformanceAttribution, PerformanceCounter};

use super::Qwen3_5Model;
use super::prefill_partial_layer_read::{
    assembled_prefill_expert_order, page_slots_for_expert_ids, split_prefill_route,
};

impl Qwen3_5Model {
    /// Assembles the routed union without another storage read.
    ///
    /// The retained half is counted as a read the table avoided. The missing
    /// half was already counted when it was streamed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn assemble_refused_prefill_page(
        &self,
        layer_index: usize,
        expert_capacity: usize,
        routed_expert_ids: &[usize],
        missing_route_expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        streamed_manifest: &QuantizedExpertPageManifest,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), Qwen3_5ExecutionError>
    {
        let route_split = split_prefill_route(routed_expert_ids, missing_route_expert_ids);
        if route_split.missing_expert_ids.is_empty() {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "a covered prefill route had no warm-table page",
            });
        }
        let missing_page = select_streamed_prefill_experts(
            &self.runtime,
            streamed_weights,
            streamed_manifest,
            &route_split.missing_expert_ids,
        )?;
        if route_split.retained_expert_ids.is_empty() {
            return manifest_for_assembled_page(
                expert_capacity,
                &route_split.retained_expert_ids,
                &route_split.missing_expert_ids,
                missing_page,
            );
        }
        let Some((retained_page, _)) = self.routed_subset_of_cached_page(
            layer_index,
            &route_split.retained_expert_ids,
            expert_capacity,
        )?
        else {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "a retained prefill expert was missing from the warm table",
            });
        };
        self.record_avoided_resident_page(
            retained_page.resident_payload_byte_count(),
            performance_attribution,
        );
        let assembled_page =
            retained_page.concatenate_expert_pages(&self.runtime, &missing_page)?;
        manifest_for_assembled_page(
            expert_capacity,
            &route_split.retained_expert_ids,
            &route_split.missing_expert_ids,
            assembled_page,
        )
    }

    pub(super) fn record_avoided_resident_page(
        &self,
        payload_byte_count: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        performance_attribution.record_counter(
            PerformanceCounter::AvoidedCompleteLayerExpertSourcePayloadBytes,
            payload_byte_count,
        );
        if let Some(expert_pager) = self.expert_pager.as_ref() {
            expert_pager.record_session_avoided_source_bytes(payload_byte_count);
        }
    }
}

fn select_streamed_prefill_experts(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    streamed_weights: &Qwen3_5PagedExpertWeights,
    streamed_manifest: &QuantizedExpertPageManifest,
    expert_ids: &[usize],
) -> Result<Qwen3_5PagedExpertWeights, Qwen3_5ExecutionError> {
    let Some(page_slots) =
        page_slots_for_expert_ids(&streamed_manifest.page_slot_by_global_expert_id, expert_ids)
    else {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "a streamed prefill expert is absent from its page",
        });
    };
    streamed_weights
        .select_experts(runtime, &page_slots)
        .map_err(Qwen3_5ExecutionError::from)
}

fn manifest_for_assembled_page(
    expert_capacity: usize,
    retained_expert_ids: &[usize],
    missing_expert_ids: &[usize],
    assembled_page: Qwen3_5PagedExpertWeights,
) -> Result<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest), Qwen3_5ExecutionError> {
    let Some(expert_order) =
        assembled_prefill_expert_order(retained_expert_ids, missing_expert_ids, expert_capacity)
    else {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "a prefill expert id is outside its layer",
        });
    };
    let payload_byte_count = assembled_page.resident_payload_byte_count();
    Ok((
        assembled_page,
        QuantizedExpertPageManifest {
            expert_ids: expert_order.expert_ids,
            page_slot_by_global_expert_id: expert_order.page_slot_by_global_expert_id,
            source_manifests: Vec::new(),
            payload_byte_count,
        },
    ))
}
