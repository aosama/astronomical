//! Warm-table creation, growth, and routed-expert slot inserts for the
//! retained cache.
//!
//! A decode miss streams one token's routed experts, and this module decides
//! whether that page becomes a retained warm table. The first insert adopts
//! the streamed page as-is (no padding); when the table fills, it doubles in
//! place while the budget admits the growth, so resident bytes always track
//! demonstrated demand within a factor of two. Eagerly padding a new table to
//! the whole affordable capacity charged the budget for zeros and starved
//! sibling layers' warming after the first token (issue #955). Budget
//! admission is the only refusal reason: a page the machine cannot hold stays
//! operation-local.
//!
//! Every write path ends in
//! [`materialize_and_detach_table_weights`][super::slot_writes::materialize_and_detach_table_weights]:
//! MLX retains an array's graph inputs for its lifetime, so a retained table
//! built from lazy row-writes would pin each source buffer — the streamed
//! page of a fill, the retired tensor of a growth — in active memory until
//! the table dies. Evaluating and detaching keeps the published memory
//! decomposition exact.

use std::collections::HashMap;

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::slot_writes::{
    RetainedReferenceOk, create_warm_table_weights, materialize_and_detach_table_weights,
    write_expert_into_slot,
};
use super::{ExpertSlotTable, RetainedExpertCache};
use crate::expert_paging::ExpertWeightPage;
use crate::qwen3_5_streaming::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;

impl RetainedExpertCache {
    /// Inserts streamed experts into the layer's slot table. The first insert
    /// adopts the streamed page as-is so the table holds exactly the routed
    /// experts; later inserts grow the table (doubling, bounded by the policy
    /// warm capacity) or write into free and least-read slots. Returns how
    /// many experts were newly written.
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
        if self
            .tables_by_layer
            .get(layer_index)
            .is_none_or(|table| table.is_none())
        {
            let routed_expert_count = expert_ids.len();
            let per_expert_payload_bytes = streamed_weights
                .resident_payload_byte_count()
                .checked_div(u64::try_from(routed_expert_count).unwrap_or(1))
                .unwrap_or(0);
            if !self.can_admit(streamed_weights.resident_payload_byte_count()) {
                // Leave the streamed page operation-local. Evicting another complete
                // layer to seat this one would thrash the sequential decoder order.
                return Ok(0);
            }
            // Adopt the streamed page as-is: the table starts at the routed set,
            // and growth below expands it only as demand proves itself.
            let weights = streamed_weights.retained_reference_ok();
            // Today's streamed pages are materialized file reads so this is a
            // no-op evaluation, but a future lazy streaming pipeline must not
            // pin its sources for the table's lifetime: adoption gets the same
            // graph hygiene as every write path.
            materialize_and_detach_table_weights(runtime, &weights)?;
            // The residency plan validates payload as per-expert bytes times
            // occupancy, so the accounting must use exactly that product even
            // if the streamed page's raw byte count carries extra alignment.
            let payload_bytes = per_expert_payload_bytes
                .saturating_mul(u64::try_from(routed_expert_count).unwrap_or(0));
            let full_padded_payload_bytes = weights.resident_payload_byte_count();
            self.resident_payload_bytes = self
                .resident_payload_bytes
                .saturating_add(full_padded_payload_bytes);
            let mut slot_by_expert_id = HashMap::with_capacity(routed_expert_count);
            let mut expert_id_by_slot = Vec::with_capacity(routed_expert_count);
            for (slot, expert_id) in expert_ids.iter().enumerate() {
                expert_id_by_slot.push(Some(*expert_id));
                slot_by_expert_id.insert(*expert_id, slot);
            }
            let table = ExpertSlotTable {
                weights,
                expert_id_by_slot,
                slot_by_expert_id,
                read_count_by_slot: vec![0; routed_expert_count],
                occupied_slot_count: routed_expert_count,
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
                    .saturating_add(u64::try_from(routed_expert_count).unwrap_or(u64::MAX));
            }
            return Ok(routed_expert_count);
        }
        let cache_ceiling_bytes = self.effective_maximum_resident_payload_bytes();
        let table = self
            .tables_by_layer
            .get_mut(layer_index)
            .and_then(|table| table.as_mut())
            .expect("slot table was just checked");
        let mut resident_payload_bytes = self.resident_payload_bytes;
        let mut protected_expert_ids = protected_expert_ids.to_vec();
        protected_expert_ids.sort_unstable();
        protected_expert_ids.dedup();
        // Plan every slot before writing: free slots first, then growth, then
        // least-read victims. Planning up front keeps experts inserted by the
        // same flush from evicting each other — with equal (zero) read counts a
        // naive pick-lowest-slot loop would let the second insert evict the
        // first. Victims also exclude slots holding any incoming routed expert:
        // such an expert is contained now, and an eviction would demote it to a
        // late surprise miss that exhausts the plan.
        let new_expert_rows: Vec<(usize, usize)> = expert_ids
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, expert_id)| !table.slot_by_expert_id.contains_key(expert_id))
            .collect();
        let new_expert_count = new_expert_rows.len();
        if new_expert_count == 0 {
            return Ok(0);
        }
        let mut free_slots: Vec<usize> = (0..table.slot_count())
            .filter(|slot| table.expert_id_by_slot[*slot].is_none())
            .collect();
        let free_slot_deficit = new_expert_count.saturating_sub(free_slots.len());
        if free_slot_deficit > 0 {
            let growth_outcome = grow_table_within_budget(
                runtime,
                table,
                resident_payload_bytes,
                cache_ceiling_bytes,
                warm_slot_count,
                free_slot_deficit,
            )?;
            if let Some(grown_resident_payload_bytes) = growth_outcome {
                resident_payload_bytes = grown_resident_payload_bytes;
                free_slots = (0..table.slot_count())
                    .filter(|slot| table.expert_id_by_slot[*slot].is_none())
                    .collect();
                // The table tensor already grew, so the cache-wide accounting
                // must move with it even if slot planning below gives up:
                // undercounting here would let later can_admit checks overfill
                // the ceiling.
                self.resident_payload_bytes = resident_payload_bytes;
            }
        }
        let mut evictable_slots: Vec<usize> = Vec::new();
        if free_slots.len() < new_expert_count {
            evictable_slots = (0..table.slot_count())
                .filter(|slot| {
                    table.expert_id_by_slot[*slot].is_some_and(|retained_expert_id| {
                        !expert_ids.contains(&retained_expert_id)
                            && protected_expert_ids
                                .binary_search(&retained_expert_id)
                                .is_err()
                    })
                })
                .collect();
            evictable_slots.sort_unstable_by_key(|slot| {
                table
                    .read_count_by_slot
                    .get(*slot)
                    .copied()
                    .unwrap_or(u64::MAX)
            });
        }
        if new_expert_count > free_slots.len() + evictable_slots.len() {
            return Ok(0);
        }
        let mut planned_slots: Vec<usize> =
            free_slots.iter().copied().take(new_expert_count).collect();
        if planned_slots.len() < new_expert_count {
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
                // The victim's slot is refilled by this same iteration, so the
                // occupied payload stays net-constant: decrementing here leaked
                // one expert of payload per evict-refill cycle and drifted the
                // table's accounting away from its slot map (issue #955).
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
        self.resident_payload_bytes = resident_payload_bytes;
        // The fills left the table arrays with lazy graphs that pin this
        // token's streamed page for the table's lifetime. Materialize the
        // writes and strip the provenance so the page frees with the forward.
        materialize_and_detach_table_weights(runtime, &table.weights)?;
        Ok(written_expert_count)
    }
}

/// Grows a warm table while the budget admits the expansion and the policy
/// warm capacity allows it, copying every occupied expert row into the new
/// tensor. The target size doubles the table but never drops below the
/// current deficit and never exceeds the policy capacity. The retired tensor
/// frees only after the copies evaluate and the grown arrays detach from
/// their lazy graph, so the cache-wide accounting tracks real tensor bytes
/// throughout. Returns the updated cache-wide resident payload, or `None`
/// when the table could not grow.
fn grow_table_within_budget(
    runtime: &MlxRuntime,
    table: &mut ExpertSlotTable,
    resident_payload_bytes: u64,
    cache_ceiling_bytes: u64,
    warm_slot_count: usize,
    free_slot_deficit: usize,
) -> Result<Option<u64>, MlxRuntimeError> {
    if warm_slot_count == 0 || table.slot_count() >= warm_slot_count {
        return Ok(None);
    }
    let growth_target_slot_count = (table
        .slot_count()
        .saturating_mul(2)
        .max(table.slot_count() + free_slot_deficit))
    .min(warm_slot_count);
    // Budget gate before allocation: the per-expert estimate refuses a growth
    // the ceiling cannot hold without allocating the tensor first.
    let estimated_growth_payload_bytes = table
        .per_expert_payload_bytes
        .saturating_mul(u64::try_from(growth_target_slot_count - table.slot_count()).unwrap_or(0));
    if estimated_growth_payload_bytes > 0
        && resident_payload_bytes.saturating_add(estimated_growth_payload_bytes)
            > cache_ceiling_bytes
    {
        return Ok(None);
    }
    let previous_slot_count = table.slot_count();
    let previous_full_padded_payload_bytes = table.full_padded_payload_bytes;
    let mut grown_weights =
        create_warm_table_weights(runtime, &table.weights, growth_target_slot_count)?;
    for previous_slot in 0..previous_slot_count {
        if table.expert_id_by_slot[previous_slot].is_some() {
            write_expert_into_slot(
                runtime,
                &mut grown_weights,
                &table.weights,
                previous_slot,
                previous_slot,
            )?;
        }
    }
    // Until the copies evaluate, the retired tensor stays alive in active
    // memory while the cache claims only its replacement, and the lazy graph
    // would pin it for the grown table's lifetime. Growth is rare, so paying
    // one eager evaluation and a detach here keeps both the buffers and the
    // published decomposition exact (issue #955).
    materialize_and_detach_table_weights(runtime, &grown_weights)?;
    let grown_full_padded_payload_bytes = grown_weights.resident_payload_byte_count();
    let measured_growth_payload_bytes =
        grown_full_padded_payload_bytes.saturating_sub(previous_full_padded_payload_bytes);
    let projected_resident_payload_bytes =
        resident_payload_bytes.saturating_add(measured_growth_payload_bytes);
    if measured_growth_payload_bytes > estimated_growth_payload_bytes
        && projected_resident_payload_bytes > cache_ceiling_bytes
    {
        // The measured growth exceeded the pre-allocation estimate and no
        // longer fits. The table itself already grew, so keep the honest
        // accounting and let the ceiling's own reclamation shed on demand.
        tracing::warn!(
            measured_growth_payload_bytes,
            estimated_growth_payload_bytes,
            projected_resident_payload_bytes,
            cache_ceiling_bytes,
            "warm-table growth measured above the budget estimate"
        );
    }
    table.weights = grown_weights;
    table
        .expert_id_by_slot
        .resize(growth_target_slot_count, None);
    table.read_count_by_slot.resize(growth_target_slot_count, 0);
    table.full_padded_payload_bytes = grown_full_padded_payload_bytes;
    tracing::debug!(
        previous_slot_count,
        growth_target_slot_count,
        measured_growth_payload_bytes,
        "grew a decode warm table for demonstrated demand"
    );
    Ok(Some(projected_resident_payload_bytes))
}
