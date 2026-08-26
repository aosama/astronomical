//! Per-layer complete expert pages for SSD-streamed MoE.
//!
//! Prefill streams every expert of a decoder index and seats that complete page
//! when leftover RAM allows. A seated complete layer is a cache hit for every
//! later route. Layers that do not fit stay operation-local so sequential
//! decoder order does not thrash pinned complete layers.

use std::collections::HashMap;

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::expert_paging::{
    ExpertWeightMemoryCacheStatistics, ExpertWeightPage, QuantizedExpertPageManifest,
    RetainedExpertReclamation,
};
use crate::memory::{CurrentExpertLayerResidency, RetainedExpertPageClass};
use crate::qwen3_5_moe::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;

mod slot_writes;
use slot_writes::{RetainedReferenceOk, write_expert_into_slot};

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
    /// adopts the streamed unique-expert page as-is. Later inserts write each
    /// new expert into a free or least-read unprotected slot via `slice_update`.
    pub fn insert_streamed_experts(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        protected_expert_ids: &[usize],
        _expert_capacity: usize,
    ) -> Result<bool, MlxRuntimeError> {
        if expert_ids.is_empty() {
            return Ok(true);
        }
        if self
            .tables_by_layer
            .get(layer_index)
            .is_none_or(|table| table.is_none())
        {
            let weights = Qwen3_5PagedExpertWeights {
                gate_projection: streamed_weights.gate_projection.retained_reference()?,
                up_projection: streamed_weights.up_projection.retained_reference()?,
                down_projection: streamed_weights.down_projection.retained_reference()?,
            };
            let capacity = expert_ids.len();
            let per_expert_payload_bytes = streamed_weights
                .resident_payload_byte_count()
                .checked_div(u64::try_from(expert_ids.len()).unwrap_or(1))
                .unwrap_or(0);
            let payload_bytes = streamed_weights.resident_payload_byte_count();
            let full_padded_payload_bytes = weights.resident_payload_byte_count();
            if !self.can_admit(full_padded_payload_bytes) {
                // Leave the streamed page operation-local. Evicting another complete
                // layer to seat this one would thrash the sequential decoder order.
                return Ok(false);
            }
            self.resident_payload_bytes = self
                .resident_payload_bytes
                .saturating_add(full_padded_payload_bytes);
            let mut slot_by_expert_id = HashMap::with_capacity(capacity);
            let mut expert_id_by_slot = Vec::with_capacity(capacity);
            for (slot, expert_id) in expert_ids.iter().copied().enumerate() {
                slot_by_expert_id.insert(expert_id, slot);
                expert_id_by_slot.push(Some(expert_id));
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
            return Ok(true);
        }
        let table = self
            .tables_by_layer
            .get_mut(layer_index)
            .and_then(|table| table.as_mut())
            .expect("slot table was just checked");
        let mut protected_expert_ids = protected_expert_ids.to_vec();
        protected_expert_ids.sort_unstable();
        protected_expert_ids.dedup();
        let new_expert_count = expert_ids
            .iter()
            .filter(|expert_id| !table.slot_by_expert_id.contains_key(expert_id))
            .count();
        let evictable_slot_count = (0..table.slot_count())
            .filter(|slot| {
                table.expert_id_by_slot[*slot]
                    .is_none_or(|expert_id| protected_expert_ids.binary_search(&expert_id).is_err())
            })
            .count();
        if new_expert_count > evictable_slot_count {
            return Ok(false);
        }
        for (expert_row, expert_id) in expert_ids.iter().copied().enumerate() {
            if table.slot_by_expert_id.contains_key(&expert_id) {
                continue;
            }
            let slot = Self::select_slot_for_insert(table, &protected_expert_ids)
                .expect("evictable slot was counted before insert");
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
        }
        Ok(true)
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
    }

    #[must_use]
    pub fn has_complete_layer(&self, layer_index: usize, expert_capacity: usize) -> bool {
        let Some(Some(table)) = self.tables_by_layer.get(layer_index) else {
            return false;
        };
        expert_capacity > 0 && table.occupied_slot_count == expert_capacity
    }

    /// Writes queued miss experts into their layer tables after GPU evaluation.
    pub fn flush_pending_inserts(&mut self, runtime: &MlxRuntime) -> Result<(), MlxRuntimeError> {
        let flush_started_at = std::time::Instant::now();
        let pending_inserts = std::mem::take(&mut self.pending_inserts);
        let pending_count = pending_inserts.len();
        let total_expert_count: usize = pending_inserts.iter().map(|pi| pi.expert_ids.len()).sum();
        for pending_insert in pending_inserts {
            let insert_started_at = std::time::Instant::now();
            let expert_count = pending_insert.expert_ids.len();
            self.insert_streamed_experts(
                runtime,
                pending_insert.layer_index,
                &pending_insert.expert_ids,
                &pending_insert.weights,
                &[],
                // expert_capacity is only used when creating a new table. By the
                // time flush runs, the table already exists from a prior insert,
                // so this value is unused.
                0,
            )?;
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
        Ok(())
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
                Some(CurrentExpertLayerResidency {
                    layer_index,
                    class,
                    retained_expert_ids: expert_ids,
                    payload_bytes: table.payload_bytes,
                    covered_weighted_demand: 0,
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

    pub fn limit_for_request_pressure(&mut self, reclamation_target_bytes: u64) -> bool {
        let pressure_maximum = self
            .resident_payload_bytes
            .saturating_sub(reclamation_target_bytes);
        self.limit_for_request_pressure_to_maximum(pressure_maximum)
    }

    pub fn limit_for_request_pressure_to_maximum(
        &mut self,
        pressure_maximum_resident_payload_bytes: u64,
    ) -> bool {
        self.request_pressure_maximum_resident_payload_bytes =
            Some(pressure_maximum_resident_payload_bytes);
        self.reclaim_to_effective_ceiling().released_payload_bytes() > 0
    }

    pub fn resume_after_request_pressure(&mut self) -> bool {
        self.request_pressure_maximum_resident_payload_bytes
            .take()
            .is_some()
    }

    pub fn release_all(&mut self) -> bool {
        let had_experts = self.resident_payload_bytes > 0;
        for layer_index in 0..self.tables_by_layer.len() {
            self.remove_layer(layer_index);
        }
        had_experts
    }

    #[must_use]
    pub fn statistics(&self) -> ExpertWeightMemoryCacheStatistics {
        let occupied_layer_count = self
            .tables_by_layer
            .iter()
            .filter(|table| table.is_some())
            .count();
        let total_expert_count = self
            .tables_by_layer
            .iter()
            .filter_map(|table| table.as_ref())
            .map(|table| table.occupied_slot_count)
            .sum::<usize>();
        ExpertWeightMemoryCacheStatistics {
            entry_count: total_expert_count,
            resident_payload_byte_count: self.resident_payload_bytes,
            maximum_resident_payload_byte_count: self.effective_maximum_resident_payload_bytes(),
            eviction_count: self.eviction_count,
            disk_page_load_count: self.disk_page_load_count,
            disk_batch_load_count: self.disk_batch_load_count,
            complete_layer_count: 0,
            complete_layer_payload_byte_count: 0,
            partial_layer_count: occupied_layer_count,
            partial_layer_payload_byte_count: self.resident_payload_bytes,
            mandatory_read_promotion_count: 0,
            complete_layer_eviction_count: 0,
            partial_layer_eviction_count: self.eviction_count,
        }
    }

    fn select_slot_for_insert(
        table: &ExpertSlotTable,
        protected_expert_ids: &[usize],
    ) -> Option<usize> {
        // Prefer a free slot; otherwise evict the least-read unprotected expert.
        if let Some(slot) =
            (0..table.slot_count()).find(|slot| table.expert_id_by_slot[*slot].is_none())
        {
            return Some(slot);
        }
        (0..table.slot_count())
            .filter(|slot| {
                table.expert_id_by_slot[*slot]
                    .is_none_or(|expert_id| protected_expert_ids.binary_search(&expert_id).is_err())
            })
            .min_by_key(|slot| {
                table
                    .read_count_by_slot
                    .get(*slot)
                    .copied()
                    .unwrap_or(u64::MAX)
            })
    }

    fn can_admit(&self, new_payload_bytes: u64) -> bool {
        self.resident_payload_bytes
            .saturating_add(new_payload_bytes)
            <= self.effective_maximum_resident_payload_bytes()
    }

    fn evict_least_used_table(&mut self) -> bool {
        let mut fewest: Option<(usize, usize)> = None;
        for (layer_index, table) in self.tables_by_layer.iter().enumerate() {
            if let Some(table) = table.as_ref() {
                let is_smaller = fewest
                    .is_none_or(|(_, smallest_count)| table.occupied_slot_count < smallest_count);
                if is_smaller {
                    fewest = Some((layer_index, table.occupied_slot_count));
                }
            }
        }
        let Some((layer_index, _)) = fewest else {
            return false;
        };
        self.remove_layer(layer_index)
    }

    fn reclaim_to_effective_ceiling(&mut self) -> RetainedExpertReclamation {
        let mut released_bytes = 0_u64;
        let mut released_count = 0_usize;
        tracing::debug!(
            resident_payload_bytes = self.resident_payload_bytes,
            effective_maximum = self.effective_maximum_resident_payload_bytes(),
            table_count = self.tables_by_layer.iter().filter(|t| t.is_some()).count(),
            "reclaim_to_effective_ceiling starting"
        );
        while self.resident_payload_bytes > self.effective_maximum_resident_payload_bytes() {
            let before = self.resident_payload_bytes;
            if !self.evict_least_used_table() {
                break;
            }
            released_bytes =
                released_bytes.saturating_add(before.saturating_sub(self.resident_payload_bytes));
            released_count += 1;
        }
        if released_count > 0 {
            tracing::info!(
                released_count,
                released_bytes,
                resident_payload_bytes = self.resident_payload_bytes,
                table_count = self.tables_by_layer.iter().filter(|t| t.is_some()).count(),
                "reclaim_to_effective_ceiling evicted tables"
            );
        }
        RetainedExpertReclamation {
            released_partial_layer_count: released_count,
            released_partial_payload_bytes: released_bytes,
            released_complete_layer_count: 0,
            released_complete_payload_bytes: 0,
        }
    }

    fn effective_maximum_resident_payload_bytes(&self) -> u64 {
        self.request_pressure_maximum_resident_payload_bytes
            .unwrap_or(self.normal_maximum_resident_payload_bytes)
    }
}
