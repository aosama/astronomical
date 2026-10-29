//! Per-layer expert slot tables for SSD-streamed MoE.
//!
//! A funded multi-token prefill route union can seed a table, and a later read
//! can fill its remaining rows into a complete layer. Decode grows warm tables
//! from routed experts. Cache-budget refusals leave streamed pages
//! operation-local rather than displacing sibling layers.

use std::collections::{HashMap, HashSet};

use astronomical_ipc_protocol::WorkerDecodeExpertCachePolicy;
use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::expert_paging::{
    ExpertWeightPage, QuantizedExpertPageManifest, RetainedExpertReclamation,
};
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;

use slot_writes::RetainedReferenceOk;

mod decode_cache_policy;
mod demand;
mod flush;
mod insert;
mod insert_planning;
mod previous_token_prefetch;
mod reclamation;
mod slot_writes;

#[cfg(all(test, feature = "direct-mlx"))]
mod memory_behavior_tests;

#[cfg(all(test, feature = "direct-mlx"))]
mod prefill_seed_tests;

#[cfg(all(test, feature = "direct-mlx"))]
mod warm_expert_slot_table_cache_tests;

#[cfg(all(test, feature = "direct-mlx"))]
mod table_query_tests;

#[cfg(all(test, feature = "direct-mlx"))]
mod slot_writes_tests;

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
    decode_policy_by_slot: Vec<decode_cache_policy::DecodeExpertSlotState>,
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
    is_decode_demand: bool,
    /// Issue #537: leftover-only insert. Never evicts an occupied slot.
    no_evict: bool,
}

/// RAM-resident expert slot tables keyed by decoder index.
#[derive(Debug)]
pub struct WarmExpertSlotTableCache {
    tables_by_layer: Vec<Option<ExpertSlotTable>>,
    pending_inserts: Vec<PendingSlotInsert>,
    expert_demand_counts_by_layer: Vec<Vec<u64>>,
    /// Persisted demand counts seeded at construction (colibri's `.coli_usage`
    /// analog): historically hot experts skip probation on first admission and
    /// the demand ranking starts warm instead of at the first-touch floor.
    historical_demand_counts_by_layer: Vec<Vec<u64>>,
    demand_assignment_weight: u64,
    resident_payload_bytes: u64,
    normal_maximum_resident_payload_bytes: u64,
    request_pressure_maximum_resident_payload_bytes: Option<u64>,
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    decode_forward_active: bool,
    decode_step_epoch: u64,
    slot_replacement_count: u64,
    decode_slot_eviction_count: u64,
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

impl WarmExpertSlotTableCache {
    #[cfg(test)]
    #[must_use]
    pub fn new(layer_count: usize) -> Self {
        Self::new_with_decode_cache_policy(layer_count, WorkerDecodeExpertCachePolicy::Baseline)
    }

    pub(crate) fn new_with_decode_cache_policy(
        layer_count: usize,
        decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    ) -> Self {
        Self {
            tables_by_layer: (0..layer_count).map(|_| None).collect(),
            pending_inserts: Vec::new(),
            expert_demand_counts_by_layer: (0..layer_count).map(|_| Vec::new()).collect(),
            historical_demand_counts_by_layer: Vec::new(),
            demand_assignment_weight: 1,
            resident_payload_bytes: 0,
            normal_maximum_resident_payload_bytes: 0,
            request_pressure_maximum_resident_payload_bytes: None,
            decode_expert_cache_policy,
            decode_forward_active: false,
            decode_step_epoch: 0,
            slot_replacement_count: 0,
            decode_slot_eviction_count: 0,
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

    pub(crate) fn begin_decode_forward(&mut self) {
        self.decode_forward_active = true;
        if self.decode_expert_cache_policy == WorkerDecodeExpertCachePolicy::Candidate {
            self.decode_step_epoch = self.decode_step_epoch.saturating_add(1);
            for table in self.tables_by_layer.iter_mut().flatten() {
                for decode_policy_state in &mut table.decode_policy_by_slot {
                    decode_policy_state.expire_probationary_admission(self.decode_step_epoch);
                }
            }
        }
    }

    pub(crate) fn end_decode_forward(&mut self) {
        self.decode_forward_active = false;
    }

    /// Seeds demand ranking and candidate admission from a persisted usage
    /// history. Historically hot experts skip probation at first admission;
    /// the seeded demand counts also rank residency ahead of first-touch noise
    /// for the least-frequently-used order.
    pub(crate) fn seed_expert_demand_history(
        &mut self,
        historical_demand_counts_by_layer: Vec<Vec<u64>>,
    ) {
        if historical_demand_counts_by_layer.len() != self.tables_by_layer.len() {
            tracing::warn!(
                seeded_layer_count = historical_demand_counts_by_layer.len(),
                expected_layer_count = self.tables_by_layer.len(),
                "expert usage history shape mismatch; starting cold"
            );
            return;
        }
        let seeded_expert_count: u64 = historical_demand_counts_by_layer
            .iter()
            .flat_map(|layer_counts| layer_counts.iter())
            .filter(|demand_count| **demand_count > 0)
            .count()
            .try_into()
            .unwrap_or(u64::MAX);
        for (layer_index, historical_layer_counts) in
            historical_demand_counts_by_layer.iter().enumerate()
        {
            let live_layer_counts = &mut self.expert_demand_counts_by_layer[layer_index];
            if live_layer_counts.len() < historical_layer_counts.len() {
                live_layer_counts.resize(historical_layer_counts.len(), 0);
            }
            for (expert_id, historical_demand_count) in historical_layer_counts.iter().enumerate() {
                live_layer_counts[expert_id] =
                    live_layer_counts[expert_id].saturating_add(*historical_demand_count);
            }
        }
        self.historical_demand_counts_by_layer = historical_demand_counts_by_layer;
        tracing::info!(
            seeded_expert_count,
            "seeded expert warm tables from persisted usage history"
        );
    }

    /// Persisted demand count for one expert, zero when no history was seeded.
    pub(crate) fn historical_demand_count(&self, layer_index: usize, expert_id: usize) -> u64 {
        self.historical_demand_counts_by_layer
            .get(layer_index)
            .and_then(|layer_counts| layer_counts.get(expert_id))
            .copied()
            .unwrap_or(0)
    }

    /// Live per-layer expert demand counts for usage-history persistence.
    pub(crate) fn expert_demand_counts(&self) -> &[Vec<u64>] {
        &self.expert_demand_counts_by_layer
    }

    pub(crate) fn take_decode_slot_eviction_count(&mut self) -> u64 {
        std::mem::take(&mut self.decode_slot_eviction_count)
    }

    /// Route experts absent from this layer's table, or `None` when the layer
    /// has no table. Sorted and deduplicated for one positional-range read.
    #[must_use]
    pub fn missing_route_expert_ids(
        &self,
        layer_index: usize,
        routed_expert_ids: &[usize],
    ) -> Option<Vec<usize>> {
        let table = self.tables_by_layer.get(layer_index)?.as_ref()?;
        let mut missing_route_expert_ids = routed_expert_ids.to_vec();
        missing_route_expert_ids.sort_unstable();
        missing_route_expert_ids.dedup();
        missing_route_expert_ids
            .retain(|expert_id| !table.slot_by_expert_id.contains_key(expert_id));
        Some(missing_route_expert_ids)
    }

    /// Every expert of the layer absent from its table, or `None` when the
    /// layer has no table: the one read set that completes a partial table.
    pub fn non_resident_expert_ids(
        &self,
        layer_index: usize,
        expert_capacity: usize,
    ) -> Option<Vec<usize>> {
        let table = self.tables_by_layer.get(layer_index)?.as_ref()?;
        Some(
            (0..expert_capacity)
                .filter(|expert_id| !table.slot_by_expert_id.contains_key(expert_id))
                .collect(),
        )
    }

    /// Occupied slots of this layer's table, zero when the layer has no table.
    pub fn occupied_slot_count(&self, layer_index: usize) -> usize {
        self.tables_by_layer
            .get(layer_index)
            .and_then(|table| table.as_ref())
            .map_or(0, |table| table.occupied_slot_count)
    }

    /// Returns the warm table's preallocated weights and a slot-indexed
    /// manifest, but only when every requested expert is seated — an
    /// all-or-nothing coverage contract. A single missing expert yields
    /// `None`, leaving mixed coverage to the caller (the decode dispatch in
    /// `forward.rs` splits it through the cached-plus-streamed route).
    /// The gather path uses slot ids derived from `page_slot_by_global_expert_id`.
    #[must_use]
    pub fn full_coverage_page(
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
            is_decode_demand: self.decode_forward_active,
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
            is_decode_demand: false,
            no_evict: true,
        });
        Ok(())
    }

    /// Counts one served read for each routed expert present in the layer's
    /// table, feeding the least-frequently-used eviction order so a stable hot
    /// set outlives one-off routing noise.
    /// Records routed reads for the least-frequently-used eviction order.
    pub fn record_routed_reads(&mut self, layer_index: usize, expert_ids: &[usize]) {
        let record_candidate_decode_reuse = self.decode_forward_active
            && self.decode_expert_cache_policy == WorkerDecodeExpertCachePolicy::Candidate;
        let decode_step_epoch = self.decode_step_epoch;
        let Some(Some(table)) = self.tables_by_layer.get_mut(layer_index) else {
            return;
        };
        for expert_id in expert_ids {
            if let Some(slot) = table.slot_by_expert_id.get(expert_id)
                && let Some(read_count) = table.read_count_by_slot.get_mut(*slot)
            {
                *read_count = read_count.saturating_add(1);
                if record_candidate_decode_reuse
                    && let Some(decode_policy_state) = table.decode_policy_by_slot.get_mut(*slot)
                {
                    decode_policy_state.record_decode_reuse(decode_step_epoch);
                }
            }
        }
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
