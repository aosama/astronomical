//! Per-layer complete expert pages for SSD-streamed MoE.
//!
//! Prefill streams every expert of a decoder index and seats that complete page
//! when leftover RAM allows. A seated complete layer is a cache hit for every
//! later route. Layers that do not fit stay operation-local so sequential
//! decoder order does not thrash pinned complete layers.

use std::collections::{HashMap, HashSet};

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::expert_paging::{
    ExpertWeightPage, QuantizedExpertPageManifest, RetainedExpertReclamation,
};
use crate::qwen3_5_moe::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;

use slot_writes::RetainedReferenceOk;

mod demand;
mod flush;
mod insert;
mod previous_token_prefetch;
mod reclamation;
mod slot_writes;

#[cfg(all(test, feature = "direct-mlx"))]
mod tests;

/// Warm-table coverage of one decode token's routed experts.
#[derive(Debug, Eq, PartialEq)]
pub struct RoutedExpertCoverage {
    pub retained_expert_ids: Vec<usize>,
    pub missing_expert_ids: Vec<usize>,
}

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
    /// Issue #537: leftover-only insert. Never evicts an occupied slot.
    no_evict: bool,
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
    /// Previous-token experts parked in leftover slots and not yet demanded.
    prefetched_unconsumed_experts: HashSet<(usize, usize)>,
    prefetch_issue_count: u64,
    prefetch_capacity_drop_count: u64,
    prefetch_payload_bytes: u64,
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
            prefetched_unconsumed_experts: HashSet::new(),
            prefetch_issue_count: 0,
            prefetch_capacity_drop_count: 0,
            prefetch_payload_bytes: 0,
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
            no_evict: false,
        });
        Ok(())
    }

    /// Queues already-streamed experts for a leftover-only insert after eval.
    pub fn queue_pending_prefetch_insert(
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
            no_evict: true,
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

    /// Takes ownership of every complete layer for resident adoption, then
    /// releases whatever remains.
    ///
    /// Issue #501: the complete-resident promotion can build the resident owner
    /// from these arrays instead of re-reading the identical payload from
    /// storage. Complete layers are returned in the layer-index order the
    /// resident owner consumes; a partial page is not adoptable and is released
    /// with the rest of the cache. Accounting is transferred with ownership, so
    /// the cache stops charging adopted payload before the promotion's fit
    /// projection re-adds it as resident payload.
    pub fn take_complete_layers_for_resident_adoption(
        &mut self,
        expert_capacity: usize,
    ) -> Vec<(usize, Qwen3_5PagedExpertWeights)> {
        let mut adopted_complete_layers = Vec::new();
        for layer_index in 0..self.tables_by_layer.len() {
            if let Some(weights) = self.take_complete_layer(layer_index, expert_capacity) {
                adopted_complete_layers.push((layer_index, weights));
            }
        }
        self.release_all();
        adopted_complete_layers
    }

    /// Probes warm-table coverage for one decode token's routed experts.
    ///
    /// Issue #373: the all-or-nothing rule serves a token from retained RAM only
    /// when every routed expert is warm. Measuring how often tokens arrive
    /// partially covered is the prerequisite for deciding whether mixed serving
    /// is worth its added forward path. This probe reads no arrays and builds no
    /// page, so classification stays free for tokens that cannot be served.
    #[must_use]
    pub fn routed_expert_coverage(
        &self,
        layer_index: usize,
        routed_expert_ids: &[usize],
    ) -> RoutedExpertCoverage {
        let Some(Some(table)) = self.tables_by_layer.get(layer_index) else {
            return RoutedExpertCoverage {
                retained_expert_ids: Vec::new(),
                missing_expert_ids: routed_expert_ids.to_vec(),
            };
        };
        let mut retained_expert_ids = Vec::new();
        let mut missing_expert_ids = Vec::new();
        for routed_expert_id in routed_expert_ids {
            if table.slot_by_expert_id.contains_key(routed_expert_id) {
                retained_expert_ids.push(*routed_expert_id);
            } else {
                missing_expert_ids.push(*routed_expert_id);
            }
        }
        RoutedExpertCoverage {
            retained_expert_ids,
            missing_expert_ids,
        }
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

    /// Takes ownership of one complete layer's weights for resident adoption.
    ///
    /// Issue #501: the complete-resident promotion can build the resident owner
    /// from these arrays instead of re-reading the identical payload from
    /// storage. A complete layer's table adopted its streamed page as-is, so the
    /// returned weights are compact and expert-index ordered — the shape the
    /// resident owner wants. Accounting is transferred: the caller now owns the
    /// payload, so the cache stops charging it. Returns `None` when the layer is
    /// absent or only partially retained, because a partial page cannot become
    /// a complete resident layer.
    pub fn take_complete_layer(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
    ) -> Option<Qwen3_5PagedExpertWeights> {
        if !self.has_complete_layer(layer_index, expert_capacity) {
            return None;
        }
        let table_slot = self.tables_by_layer.get_mut(layer_index)?;
        let removed = table_slot.take()?;
        self.resident_payload_bytes = self
            .resident_payload_bytes
            .saturating_sub(removed.full_padded_payload_bytes);
        Some(removed.weights)
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
