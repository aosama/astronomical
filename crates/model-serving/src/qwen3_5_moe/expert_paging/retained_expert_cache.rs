//! Per-layer complete expert pages for SSD-streamed MoE.
//!
//! Prefill streams every expert of a decoder index and seats that complete page
//! when leftover RAM allows. A seated complete layer is a cache hit for every
//! later route. Layers that do not fit stay operation-local so sequential
//! decoder order does not thrash pinned complete layers.

use std::collections::HashMap;

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::expert_paging::{
    ExpertWeightPage, QuantizedExpertPageManifest, RetainedExpertReclamation,
};
use crate::memory::{CurrentExpertLayerResidency, RetainedExpertPageClass};
use crate::qwen3_5_moe::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;

mod reclamation;
mod slot_writes;

#[cfg(all(test, feature = "direct-mlx"))]
mod tests;
use slot_writes::{RetainedReferenceOk, create_warm_table_weights, write_expert_into_slot};

/// One layer's slot table: a preallocated weight tensor and its slot map.
#[derive(Debug)]
struct ExpertSlotTable {
    weights: Qwen3_5PagedExpertWeights,
    expert_id_by_slot: Vec<Option<usize>>,
    slot_by_expert_id: HashMap<usize, usize>,
    read_count_by_slot: Vec<u64>,
    occupied_slot_count: usize,
    /// Per-expert payload bytes. Used when decode inserts or evicts a slot.
    per_expert_payload_bytes: u64,
    /// Occupied-expert payload. The residency plan validates this against
    /// occupied_count × geometry.expert_payload_bytes.
    payload_bytes: u64,
    /// GPU memory consumed by the weight tensors.
    full_padded_payload_bytes: u64,
}

impl ExpertSlotTable {
    fn slot_count(&self) -> usize {
        self.expert_id_by_slot.len()
    }
}

/// Streamed miss page waiting until the current forward has evaluated.
#[derive(Debug)]
struct PendingSlotInsert {
    layer_index: usize,
    expert_ids: Vec<usize>,
    weights: Qwen3_5PagedExpertWeights,
    /// Slot capacity for the table created by this insert, decided by the
    /// memory package's warm-capacity policy at queue time.
    warm_slot_count: usize,
}

/// RAM-resident expert slot tables keyed by decoder index.
#[derive(Debug)]
pub struct RetainedExpertCache {
    tables_by_layer: Vec<Option<ExpertSlotTable>>,
    pending_inserts: Vec<PendingSlotInsert>,
    expert_demand_counts_by_layer: Vec<Vec<u64>>,
    demand_assignment_weight: u64,
    resident_payload_bytes: u64,
    normal_maximum_resident_payload_bytes: u64,
    request_pressure_maximum_resident_payload_bytes: Option<u64>,
    eviction_count: u64,
    disk_page_load_count: u64,
    disk_batch_load_count: u64,
    /// Routed experts written into warm tables (hot-expert caching evidence).
    warm_expert_insert_count: u64,
}

impl RetainedExpertCache {
    #[must_use]
    pub fn new(layer_count: usize) -> Self {
        Self {
            tables_by_layer: (0..layer_count).map(|_| None).collect(),
            pending_inserts: Vec::new(),
            expert_demand_counts_by_layer: (0..layer_count).map(|_| Vec::new()).collect(),
            demand_assignment_weight: 1,
            resident_payload_bytes: 0,
            normal_maximum_resident_payload_bytes: 0,
            request_pressure_maximum_resident_payload_bytes: None,
            eviction_count: 0,
            disk_page_load_count: 0,
            disk_batch_load_count: 0,
            warm_expert_insert_count: 0,
        }
    }

    /// Returns the preallocated weights and a slot-indexed manifest for a hit.
    /// The gather path uses slot ids derived from `page_slot_by_global_expert_id`.
    #[must_use]
    pub fn packed_page(
        &self,
        layer_index: usize,
        expert_ids: &[usize],
        expert_capacity: usize,
    ) -> Option<(Qwen3_5PagedExpertWeights, QuantizedExpertPageManifest)> {
        let table = self.tables_by_layer.get(layer_index)?.as_ref()?;
        if !expert_ids
            .iter()
            .all(|expert_id| table.slot_by_expert_id.contains_key(expert_id))
        {
            return None;
        }
        let mut page_slot_by_global_expert_id = vec![u32::MAX; expert_capacity];
        for (slot, expert_id) in table
            .expert_id_by_slot
            .iter()
            .enumerate()
            .filter_map(|(slot, maybe_id)| maybe_id.map(|id| (slot, id)))
        {
            if expert_id < page_slot_by_global_expert_id.len() {
                page_slot_by_global_expert_id[expert_id] = u32::try_from(slot).unwrap_or(u32::MAX);
            }
        }
        let mut expert_ids: Vec<usize> = table.slot_by_expert_id.keys().copied().collect();
        expert_ids.sort_unstable();
        let manifest = QuantizedExpertPageManifest {
            expert_ids,
            page_slot_by_global_expert_id,
            source_manifests: Vec::new(),
            payload_byte_count: table.payload_bytes,
        };
        Some((table.weights.retained_reference_ok(), manifest))
    }

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
        Ok(written_expert_count)
    }

    /// Seats a complete expert layer if it fits leftover budget. Returns false
    /// when the page must stay operation-local so other complete layers survive.
    pub fn try_adopt_complete_layer(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
    ) -> Result<bool, MlxRuntimeError> {
        if self.has_complete_layer(layer_index, expert_ids.len()) {
            return Ok(true);
        }
        let incoming_payload_bytes = streamed_weights.resident_payload_byte_count();
        let existing_payload_bytes = self
            .tables_by_layer
            .get(layer_index)
            .and_then(|table| table.as_ref())
            .map_or(0, |table| table.full_padded_payload_bytes);
        let projected_payload_bytes = self
            .resident_payload_bytes
            .saturating_sub(existing_payload_bytes)
            .saturating_add(incoming_payload_bytes);
        if projected_payload_bytes > self.effective_maximum_resident_payload_bytes() {
            return Ok(false);
        }
        self.remove_layer(layer_index);
        self.insert_streamed_experts(runtime, layer_index, expert_ids, streamed_weights, &[], 0)
            .map(|written_count| written_count > 0)
    }

    #[must_use]
    pub fn has_complete_layer(&self, layer_index: usize, expert_capacity: usize) -> bool {
        let Some(Some(table)) = self.tables_by_layer.get(layer_index) else {
            return false;
        };
        expert_capacity > 0 && table.occupied_slot_count == expert_capacity
    }

    /// Queues the streamed routed experts of one decode forward for hot-expert
    /// retention. The queue drains after the forward's arrays are evaluated, so
    /// warming never stalls the token that produced the experts. Whole routed
    /// sets are queued even when the table already holds some of them; the
    /// insert skips contained experts and the rows stay aligned with the
    /// streamed page.
    pub fn queue_pending_routed_expert_insert(
        &mut self,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        warm_slot_count: usize,
    ) -> Result<(), MlxRuntimeError> {
        if expert_ids.is_empty() {
            return Ok(());
        }
        self.pending_inserts.push(PendingSlotInsert {
            layer_index,
            expert_ids: expert_ids.to_vec(),
            weights: streamed_weights.retained_reference_ok(),
            warm_slot_count,
        });
        Ok(())
    }

    /// Counts one served read for each routed expert present in the layer's
    /// table, feeding the least-frequently-used eviction order so a stable hot
    /// set outlives one-off routing noise.
    pub fn record_routed_reads(&mut self, layer_index: usize, expert_ids: &[usize]) {
        let Some(Some(table)) = self.tables_by_layer.get_mut(layer_index) else {
            return;
        };
        for expert_id in expert_ids {
            if let Some(slot) = table.slot_by_expert_id.get(expert_id)
                && let Some(read_count) = table.read_count_by_slot.get_mut(*slot)
            {
                *read_count = read_count.saturating_add(1);
            }
        }
    }

    /// Writes queued miss experts into their layer tables after GPU evaluation
    /// and returns how many experts were newly written.
    pub fn flush_pending_inserts(&mut self, runtime: &MlxRuntime) -> Result<u64, MlxRuntimeError> {
        let flush_started_at = std::time::Instant::now();
        let pending_inserts = std::mem::take(&mut self.pending_inserts);
        let pending_count = pending_inserts.len();
        let total_expert_count: usize = pending_inserts.iter().map(|pi| pi.expert_ids.len()).sum();
        let mut written_expert_count = 0_u64;
        for pending_insert in pending_inserts {
            let insert_started_at = std::time::Instant::now();
            let expert_count = pending_insert.expert_ids.len();
            let written_count = self.insert_streamed_experts(
                runtime,
                pending_insert.layer_index,
                &pending_insert.expert_ids,
                &pending_insert.weights,
                &[],
                pending_insert.warm_slot_count,
            )?;
            written_expert_count = written_expert_count
                .saturating_add(u64::try_from(written_count).unwrap_or(u64::MAX));
            let insert_elapsed = insert_started_at.elapsed();
            if insert_elapsed > std::time::Duration::from_millis(10) {
                tracing::info!(
                    layer_index = pending_insert.layer_index,
                    expert_count,
                    insert_elapsed_millis = insert_elapsed.as_millis(),
                    "slow slot table insert after flush"
                );
            }
        }
        let flush_elapsed = flush_started_at.elapsed();
        if flush_elapsed > std::time::Duration::from_millis(100) {
            tracing::info!(
                pending_count,
                total_expert_count,
                flush_elapsed_millis = flush_elapsed.as_millis(),
                "flushed pending expert slot inserts"
            );
        }
        Ok(written_expert_count)
    }

    pub fn remove_layer(&mut self, layer_index: usize) -> bool {
        let Some(Some(removed)) = self
            .tables_by_layer
            .get_mut(layer_index)
            .map(|slot| slot.take())
        else {
            return false;
        };
        self.resident_payload_bytes = self
            .resident_payload_bytes
            .saturating_sub(removed.full_padded_payload_bytes);
        self.eviction_count = self.eviction_count.saturating_add(1);
        true
    }

    pub fn record_expert_demand(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_expert_ids: &[usize],
    ) {
        let Some(demand) = self.expert_demand_counts_by_layer.get_mut(layer_index) else {
            return;
        };
        if demand.len() < expert_capacity {
            demand.resize(expert_capacity, 0);
        }
        let weight = self.demand_assignment_weight.max(1);
        for expert_id in selected_expert_ids {
            if let Some(count) = demand.get_mut(*expert_id) {
                *count = count.saturating_add(weight);
            }
        }
    }

    pub fn clear_expert_demand(&mut self) {
        for demand in &mut self.expert_demand_counts_by_layer {
            demand.fill(0);
        }
        self.demand_assignment_weight = 1;
    }

    pub fn record_disk_load(&mut self, expert_count: usize, batch_count: usize) {
        self.disk_page_load_count = self
            .disk_page_load_count
            .saturating_add(u64::try_from(expert_count).unwrap_or(u64::MAX));
        self.disk_batch_load_count = self
            .disk_batch_load_count
            .saturating_add(u64::try_from(batch_count).unwrap_or(u64::MAX));
    }

    #[must_use]
    pub fn topology_snapshot(&self, expert_capacity: usize) -> Vec<CurrentExpertLayerResidency> {
        self.tables_by_layer
            .iter()
            .enumerate()
            .filter_map(|(layer_index, table)| {
                let table = table.as_ref()?;
                let mut expert_ids: Vec<usize> = table.slot_by_expert_id.keys().copied().collect();
                expert_ids.sort_unstable();
                // A slot table that has grown to hold every expert is effectively
                // a complete layer and must be classified as StableCompleteLayer
                // so the residency validation passes (ElasticRoutedExperts requires
                // retained_count < expert_capacity, which is false when all are held).
                let class = if expert_ids.len() >= expert_capacity {
                    RetainedExpertPageClass::StableCompleteLayer
                } else {
                    RetainedExpertPageClass::ElasticRoutedExperts
                };
                let covered_weighted_demand = expert_ids
                    .iter()
                    .filter_map(|expert_id| {
                        self.expert_demand_counts_by_layer[layer_index].get(*expert_id)
                    })
                    .copied()
                    .fold(0_u64, u64::saturating_add);
                Some(CurrentExpertLayerResidency {
                    layer_index,
                    class,
                    retained_expert_ids: expert_ids,
                    payload_bytes: table.payload_bytes,
                    covered_weighted_demand,
                })
            })
            .collect()
    }

    pub fn update_maximum_resident_payload_bytes(
        &mut self,
        maximum_payload_bytes: u64,
    ) -> RetainedExpertReclamation {
        self.normal_maximum_resident_payload_bytes = maximum_payload_bytes;
        // Clear any request-pressure override so eviction uses the normal
        // ceiling. Request-pressure overrides are for generation, not
        // residency planning.
        self.request_pressure_maximum_resident_payload_bytes = None;
        self.reclaim_to_effective_ceiling()
    }
}
