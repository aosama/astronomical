//! The retained-ownership offer and decode warm-insert steps of the
//! operation-local routed-expert stream. Split from `expert_reuse.rs` to keep
//! that selector under the repository's per-file source-size gate; these two
//! steps run after a storage stream completes and decide whether the freshly
//! streamed tensors become retained memory or merely serve this one forward.

use std::cell::RefCell;

use crate::expert_paging::QuantizedExpertPageManifest;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::expert_streaming::WarmExpertSlotTableCache;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{
    PerformanceAttribution, PerformanceCounter, should_commit_mandatory_routed_page,
    should_commit_multi_token_union_as_complete_layer, should_seed_partial_layer_on_prefill_read,
};

impl Qwen3_5Model {
    /// Offers a just-streamed multi-token union to retained ownership: either
    /// adopting the whole layer when the union already covers it, or seeding a
    /// partial table on a prefill read. A multi-token chunk whose union covers
    /// the whole layer has already paid the complete-layer read cost, so
    /// adopting it is pure reuse — without this offer every later chunk
    /// re-read the same bytes from storage (the 2026-10-23 Ornith flight
    /// streamed 389 GB for one 23,976-token prompt while 26.95 GB of ceiling
    /// sat unused). Budget admission inside the cache remains the only refusal
    /// path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn offer_multi_token_union_to_retained_ownership(
        &self,
        retained_experts: &RefCell<WarmExpertSlotTableCache>,
        layer_index: usize,
        expert_capacity: usize,
        route_token_count: i32,
        production_default_paging: bool,
        routed_expert_ids: &[usize],
        resident_expert_count: usize,
        layer_has_complete_table: bool,
        streamed_weights: &Qwen3_5PagedExpertWeights,
        streamed_manifest: &QuantizedExpertPageManifest,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
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
                streamed_weights,
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
                streamed_weights,
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
        Ok(())
    }

    /// Queues a single-token decode page for warm retention when hot slots are
    /// funded and the layer is not already complete. Under the active residency
    /// plan the routed page is queued mandatorily; when attribution is enabled
    /// the plan's refusal is instead served by the leftover-slot prefetch path
    /// (issue #537) so the token's streamed route still warms.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn queue_decode_warm_inserts(
        &self,
        retained_experts: &RefCell<WarmExpertSlotTableCache>,
        layer_index: usize,
        expert_capacity: usize,
        route_token_count: i32,
        production_default_paging: bool,
        routed_expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        warm_slot_count: usize,
        layer_has_complete_table: bool,
        performance_attribution: &PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
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
                        streamed_weights,
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
                        streamed_weights,
                        routed_expert_ids.len().min(expert_capacity),
                    )?;
            }
        }
        Ok(())
    }
}
