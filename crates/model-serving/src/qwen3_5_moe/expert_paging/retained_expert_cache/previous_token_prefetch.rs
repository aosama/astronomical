//! Leftover-slot inserts for previous-token prefetch (issue #537).
//!
//! These methods never evict an occupied warm slot. A miss that does not fit
//! leftover capacity is dropped so the current native route keeps its pages.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::RetainedExpertCache;
use crate::qwen3_5_moe::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;
use crate::qwen3_5_moe::expert_paging::retained_expert_cache::RoutedExpertCoverage;

impl RetainedExpertCache {
    /// Writes streamed experts into leftover slots only. Occupied slots stay.
    pub fn insert_into_free_slots_only(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        warm_slot_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        if expert_ids.is_empty() {
            return Ok(0);
        }
        let has_table = self
            .tables_by_layer
            .get(layer_index)
            .is_some_and(|table| table.is_some());
        if !has_table {
            // A new leftover table is sized to this token's routed set so it
            // cannot steal padded capacity from another layer.
            return self.insert_streamed_experts(
                runtime,
                layer_index,
                expert_ids,
                streamed_weights,
                &[],
                warm_slot_count.min(expert_ids.len()),
            );
        }
        let table = self
            .tables_by_layer
            .get_mut(layer_index)
            .and_then(|table| table.as_mut())
            .expect("the layer table was just checked");
        let free_slots: Vec<usize> = (0..table.slot_count())
            .filter(|slot| table.expert_id_by_slot[*slot].is_none())
            .collect();
        let mut written_expert_count = 0_usize;
        let mut free_slot_iter = free_slots.into_iter();
        for (expert_row, expert_id) in expert_ids.iter().copied().enumerate() {
            if table.slot_by_expert_id.contains_key(&expert_id) {
                continue;
            }
            let Some(slot) = free_slot_iter.next() else {
                break;
            };
            super::slot_writes::write_expert_into_slot(
                runtime,
                &mut table.weights,
                streamed_weights,
                expert_row,
                slot,
            )?;
            table.expert_id_by_slot[slot] = Some(expert_id);
            table.slot_by_expert_id.insert(expert_id, slot);
            table.read_count_by_slot[slot] = 0;
            table.occupied_slot_count += 1;
            table.payload_bytes = table
                .payload_bytes
                .saturating_add(table.per_expert_payload_bytes);
            written_expert_count += 1;
        }
        Ok(written_expert_count)
    }

    #[must_use]
    pub fn take_prefetch_flush_statistics(&mut self) -> (u64, u64, u64) {
        (
            std::mem::take(&mut self.prefetch_issue_count),
            std::mem::take(&mut self.prefetch_capacity_drop_count),
            std::mem::take(&mut self.prefetch_payload_bytes),
        )
    }

    pub fn mark_prefetched_experts(&mut self, layer_index: usize, expert_ids: &[usize]) {
        for expert_id in expert_ids {
            self.prefetched_unconsumed_experts
                .insert((layer_index, *expert_id));
        }
    }

    /// Scores the next token's route against leftover prefetch occupancy.
    #[must_use]
    pub fn consume_prefetch_coverage(
        &mut self,
        layer_index: usize,
        route_coverage: &RoutedExpertCoverage,
    ) -> (u64, u64) {
        let mut prefetch_hit_count = 0_u64;
        let mut prefetch_miss_count = 0_u64;
        for expert_id in &route_coverage.retained_expert_ids {
            if self
                .prefetched_unconsumed_experts
                .remove(&(layer_index, *expert_id))
            {
                prefetch_hit_count = prefetch_hit_count.saturating_add(1);
            }
        }
        for expert_id in &route_coverage.missing_expert_ids {
            if self
                .prefetched_unconsumed_experts
                .remove(&(layer_index, *expert_id))
            {
                prefetch_miss_count = prefetch_miss_count.saturating_add(1);
            }
        }
        (prefetch_hit_count, prefetch_miss_count)
    }
}
