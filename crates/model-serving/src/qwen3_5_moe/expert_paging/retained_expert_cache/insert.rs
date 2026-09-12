//! Warm-table creation and routed-expert slot inserts for the retained cache.
//!
//! A decode miss streams one token's routed experts, and this module decides
//! whether that page becomes a retained warm table. The first insert may create
//! a zero-padded table sized to the policy capacity so later tokens accumulate
//! hot experts without churning; later inserts write each new expert into a
//! free or least-read unprotected slot. Budget admission is the only refusal
//! reason: a page the machine cannot hold stays operation-local.

use std::collections::HashMap;

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::slot_writes::{RetainedReferenceOk, create_warm_table_weights, write_expert_into_slot};
use super::{ExpertSlotTable, RetainedExpertCache};
use crate::expert_paging::ExpertWeightPage;
use crate::memory::merge_predicted_experts_into_protected_set;
use crate::qwen3_5_moe::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;

impl RetainedExpertCache {
    /// Inserts streamed experts into the layer's slot table. The first insert
    /// creates the table at the warm-slot capacity (zero-padded beyond the
    /// routed rows) when the policy capacity exceeds the routed set, so later
    /// decode tokens can accumulate hot experts without churning; a routed set
    /// at or above the capacity adopts the streamed page as-is. Later inserts
    /// write each new expert into a free or least-read unprotected slot via
    /// `slice_update`. Returns how many experts were newly written.
    pub fn insert_streamed_experts(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        protected_expert_ids: &[usize],
        warm_slot_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        if expert_ids.is_empty() {
            return Ok(0);
        }
        let predicted_retention_expert_ids = self.predicted_retention_experts(layer_index).to_vec();
        let protected_expert_ids = merge_predicted_experts_into_protected_set(
            protected_expert_ids,
            &predicted_retention_expert_ids,
        );
        let protected_expert_ids = protected_expert_ids.as_slice();
        if self
            .tables_by_layer
            .get(layer_index)
            .is_none_or(|table| table.is_none())
        {
            // A warm capacity above the routed set needs a zero-padded table
            // so later decode tokens can accumulate hot experts without
            // churning; a routed set filling the capacity adopts as-is.
            let capacity = warm_slot_count.max(expert_ids.len());
            let per_expert_payload_bytes = streamed_weights
                .resident_payload_byte_count()
                .checked_div(u64::try_from(expert_ids.len()).unwrap_or(1))
                .unwrap_or(0);
            let needs_padding = capacity > expert_ids.len();
            let estimated_full_padded_payload_bytes = if needs_padding {
                // Budget gate before allocation: refusing keeps the stream
                // operation-local without allocating the padded table first.
                per_expert_payload_bytes.saturating_mul(u64::try_from(capacity).unwrap_or(u64::MAX))
            } else {
                streamed_weights.resident_payload_byte_count()
            };
            if !self.can_admit(estimated_full_padded_payload_bytes) {
                // Leave the streamed page operation-local. Evicting another complete
                // layer to seat this one would thrash the sequential decoder order.
                return Ok(0);
            }
            let mut weights = if needs_padding {
                create_warm_table_weights(runtime, streamed_weights, capacity)?
            } else {
                streamed_weights.retained_reference_ok()
            };
            if needs_padding {
                // The padded table starts zero-filled: copy each streamed
                // routed row into its leading slot. Without this write the
                // slot map would claim experts the tensor never received.
                for expert_row in 0..expert_ids.len() {
                    write_expert_into_slot(
                        runtime,
                        &mut weights,
                        streamed_weights,
                        expert_row,
                        expert_row,
                    )?;
                }
            }
            let payload_bytes = streamed_weights.resident_payload_byte_count();
            let full_padded_payload_bytes = weights.resident_payload_byte_count();
            self.resident_payload_bytes = self
                .resident_payload_bytes
                .saturating_add(full_padded_payload_bytes);
            // Every slot must exist in the map: free slots are `None` so the
            // insert path can find them, and occupied slots map their expert.
            let mut slot_by_expert_id = HashMap::with_capacity(capacity);
            let mut expert_id_by_slot = Vec::with_capacity(capacity);
            for slot in 0..capacity {
                let maybe_expert_id = expert_ids.get(slot).copied();
                expert_id_by_slot.push(maybe_expert_id);
                if let Some(expert_id) = maybe_expert_id {
                    slot_by_expert_id.insert(expert_id, slot);
                }
            }
            let table = ExpertSlotTable {
                weights,
                expert_id_by_slot,
                slot_by_expert_id,
                read_count_by_slot: vec![0; capacity],
                occupied_slot_count: expert_ids.len(),
                per_expert_payload_bytes,
                payload_bytes,
                full_padded_payload_bytes,
            };
            if let Some(slot) = self.tables_by_layer.get_mut(layer_index) {
                *slot = Some(table);
            }
            // Warm-insert evidence counts only hot-expert warming (a nonzero
            // warm capacity); complete-layer adoption is whole-layer caching.
            if warm_slot_count > 0 {
                self.warm_expert_insert_count = self
                    .warm_expert_insert_count
                    .saturating_add(u64::try_from(expert_ids.len()).unwrap_or(u64::MAX));
            }
            self.record_retention_hint_acceptances(layer_index);
            return Ok(expert_ids.len());
        }
        let table = self
            .tables_by_layer
            .get_mut(layer_index)
            .and_then(|table| table.as_mut())
            .expect("slot table was just checked");
        let mut protected_expert_ids = protected_expert_ids.to_vec();
        protected_expert_ids.sort_unstable();
        protected_expert_ids.dedup();
        // Plan every slot before writing: free slots first, then least-read
        // victims. Planning up front keeps experts inserted by the same flush
        // from evicting each other — with equal (zero) read counts a naive
        // pick-lowest-slot loop would let the second insert evict the first.
        // Victims also exclude slots holding any incoming routed expert: such
        // an expert is contained now, and an eviction would demote it to a
        // late surprise miss that exhausts the plan.
        let new_expert_rows: Vec<(usize, usize)> = expert_ids
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, expert_id)| !table.slot_by_expert_id.contains_key(expert_id))
            .collect();
        let new_expert_count = new_expert_rows.len();
        if new_expert_count == 0 {
            self.record_retention_hint_acceptances(layer_index);
            return Ok(0);
        }
        let free_slots: Vec<usize> = (0..table.slot_count())
            .filter(|slot| table.expert_id_by_slot[*slot].is_none())
            .collect();
        let mut evictable_slots: Vec<usize> = (0..table.slot_count())
            .filter(|slot| {
                table.expert_id_by_slot[*slot].is_some_and(|retained_expert_id| {
                    !expert_ids.contains(&retained_expert_id)
                        && protected_expert_ids
                            .binary_search(&retained_expert_id)
                            .is_err()
                })
            })
            .collect();
        if new_expert_count > free_slots.len() + evictable_slots.len() {
            self.record_retention_hint_acceptances(layer_index);
            return Ok(0);
        }
        let mut planned_slots: Vec<usize> =
            free_slots.iter().copied().take(new_expert_count).collect();
        if planned_slots.len() < new_expert_count {
            evictable_slots.sort_unstable_by_key(|slot| {
                table
                    .read_count_by_slot
                    .get(*slot)
                    .copied()
                    .unwrap_or(u64::MAX)
            });
            for victim_slot in evictable_slots {
                if planned_slots.len() == new_expert_count {
                    break;
                }
                planned_slots.push(victim_slot);
            }
        }
        let mut planned_slot_iter = planned_slots.into_iter();
        let mut written_expert_count = 0_usize;
        for (expert_row, expert_id) in new_expert_rows {
            let slot = planned_slot_iter
                .next()
                .expect("planned slots match the counted new expert set");
            let evicted = table.expert_id_by_slot[slot].take();
            if let Some(evicted_id) = evicted {
                table.slot_by_expert_id.remove(&evicted_id);
                // Evicting an existing expert: the slot data remains in the tensor
                // but we no longer count it as occupied payload.
                table.payload_bytes = table
                    .payload_bytes
                    .saturating_sub(table.per_expert_payload_bytes);
            } else {
                table.occupied_slot_count += 1;
                // Filling a free slot: the zero padding is overwritten by real expert
                // data, so we count the per-expert payload.
                table.payload_bytes = table
                    .payload_bytes
                    .saturating_add(table.per_expert_payload_bytes);
            }
            let write_started_at = std::time::Instant::now();
            write_expert_into_slot(
                runtime,
                &mut table.weights,
                streamed_weights,
                expert_row,
                slot,
            )?;
            let write_elapsed = write_started_at.elapsed();
            if write_elapsed > std::time::Duration::from_millis(5) {
                tracing::info!(
                    layer_index,
                    expert_id,
                    slot,
                    write_elapsed_millis = write_elapsed.as_millis(),
                    "slow write_expert_into_slot"
                );
            }
            table.expert_id_by_slot[slot] = Some(expert_id);
            table.slot_by_expert_id.insert(expert_id, slot);
            table.read_count_by_slot[slot] = 0;
            written_expert_count += 1;
        }
        if warm_slot_count > 0 {
            self.warm_expert_insert_count = self
                .warm_expert_insert_count
                .saturating_add(u64::try_from(written_expert_count).unwrap_or(u64::MAX));
        }
        self.record_retention_hint_acceptances(layer_index);
        Ok(written_expert_count)
    }
}
