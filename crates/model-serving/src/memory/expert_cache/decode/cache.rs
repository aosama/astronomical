//! Decode expert cache: one resident set per sparse layer plus a demand ledger.

use super::ResidentExpertWeight;
use super::demand_ledger::DecodeDemandLedger;
use super::eviction::evict_coldest_experts;
use super::resident_set::ResidentExpertSet;

/// Family-neutral decode cache. The unit of ownership is one expert.
#[derive(Debug)]
pub struct DecodeExpertCache<W: ResidentExpertWeight> {
    layers: Vec<ResidentExpertSet<W>>,
    ledger: DecodeDemandLedger,
    ceiling_bytes: u64,
    eviction_count: u64,
    disk_expert_load_count: u64,
    disk_batch_load_count: u64,
}

impl<W: ResidentExpertWeight> DecodeExpertCache<W> {
    #[must_use]
    pub fn new(sparse_layer_count: usize) -> Self {
        Self {
            layers: (0..sparse_layer_count)
                .map(|_| ResidentExpertSet::default())
                .collect(),
            ledger: DecodeDemandLedger::new(sparse_layer_count),
            ceiling_bytes: 0,
            eviction_count: 0,
            disk_expert_load_count: 0,
            disk_batch_load_count: 0,
        }
    }

    #[must_use]
    pub fn sparse_layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn missing(&self, slot: usize, routed_ids: &[usize]) -> Vec<usize> {
        self.layers
            .get(slot)
            .map(|layer| layer.missing(routed_ids))
            .unwrap_or_else(|| routed_ids.to_vec())
    }

    #[must_use]
    pub fn contains_every(&self, slot: usize, routed_ids: &[usize]) -> bool {
        self.layers
            .get(slot)
            .is_some_and(|layer| layer.contains_every(routed_ids))
    }

    pub fn record_demand(&mut self, slot: usize, routed_ids: &[usize]) {
        self.ledger.record(slot, routed_ids);
    }

    pub fn admit(&mut self, slot: usize, expert_id: usize, weight: W) {
        let Some(layer) = self.layers.get_mut(slot) else {
            return;
        };
        layer.admit(expert_id, weight);
    }

    pub fn record_disk_load(&mut self, expert_count: usize, batch_count: usize) {
        self.disk_expert_load_count = self
            .disk_expert_load_count
            .saturating_add(u64::try_from(expert_count).unwrap_or(u64::MAX));
        self.disk_batch_load_count = self
            .disk_batch_load_count
            .saturating_add(u64::try_from(batch_count).unwrap_or(u64::MAX));
    }

    #[must_use]
    pub fn total_payload_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(ResidentExpertSet::payload_bytes)
            .fold(0_u64, u64::saturating_add)
    }

    #[must_use]
    pub fn resident_expert_count(&self) -> usize {
        self.layers
            .iter()
            .map(ResidentExpertSet::expert_count)
            .sum()
    }

    #[must_use]
    pub fn resident_ids(&self, slot: usize) -> Vec<usize> {
        self.layers
            .get(slot)
            .map(ResidentExpertSet::resident_ids)
            .unwrap_or_default()
    }

    #[must_use]
    pub fn resident_weights(&self, slot: usize) -> Vec<(usize, &W)> {
        self.layers
            .get(slot)
            .map(|layer| layer.resident_weights().collect())
            .unwrap_or_default()
    }

    pub fn set_ceiling(&mut self, ceiling_bytes: u64) {
        self.ceiling_bytes = ceiling_bytes;
        self.enforce_ceiling(&[]);
    }

    #[must_use]
    pub fn ceiling_bytes(&self) -> u64 {
        self.ceiling_bytes
    }

    /// Evicts unprotected cold experts until the cache fits the ceiling.
    ///
    /// Returns the sparse-layer slots that actually lost experts so gather
    /// views for untouched layers can stay.
    pub fn enforce_ceiling(&mut self, protected: &[(usize, usize)]) -> Vec<usize> {
        let total_payload_bytes = self.total_payload_bytes();
        if total_payload_bytes <= self.ceiling_bytes {
            return Vec::new();
        }
        let eviction = evict_coldest_experts(
            &mut self.layers,
            &self.ledger,
            total_payload_bytes.saturating_sub(self.ceiling_bytes),
            protected,
        );
        self.eviction_count = self
            .eviction_count
            .saturating_add(eviction.evicted_expert_count as u64);
        eviction.vacated_slots
    }

    #[must_use]
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    #[must_use]
    pub fn disk_expert_load_count(&self) -> u64 {
        self.disk_expert_load_count
    }

    #[must_use]
    pub fn disk_batch_load_count(&self) -> u64 {
        self.disk_batch_load_count
    }
}
