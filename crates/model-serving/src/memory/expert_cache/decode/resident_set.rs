//! One sparse layer's resident experts.

use std::collections::BTreeMap;

use super::ResidentExpertWeight;

/// Presence of an expert id in this map is residency for that layer.
#[derive(Debug)]
pub(super) struct ResidentExpertSet<W: ResidentExpertWeight> {
    experts: BTreeMap<usize, W>,
}

impl<W: ResidentExpertWeight> Default for ResidentExpertSet<W> {
    fn default() -> Self {
        Self {
            experts: BTreeMap::new(),
        }
    }
}

impl<W: ResidentExpertWeight> ResidentExpertSet<W> {
    pub(super) fn contains_every(&self, routed_ids: &[usize]) -> bool {
        routed_ids
            .iter()
            .all(|expert_id| self.experts.contains_key(expert_id))
    }

    pub(super) fn missing(&self, routed_ids: &[usize]) -> Vec<usize> {
        routed_ids
            .iter()
            .copied()
            .filter(|expert_id| !self.experts.contains_key(expert_id))
            .collect()
    }

    pub(super) fn payload_bytes(&self) -> u64 {
        self.experts
            .values()
            .map(ResidentExpertWeight::payload_bytes)
            .fold(0_u64, u64::saturating_add)
    }

    pub(super) fn expert_count(&self) -> usize {
        self.experts.len()
    }

    pub(super) fn admit(&mut self, expert_id: usize, weight: W) {
        self.experts.insert(expert_id, weight);
    }

    pub(super) fn evict(&mut self, expert_id: usize) -> Option<W> {
        self.experts.remove(&expert_id)
    }

    pub(super) fn payload_bytes_for(&self, expert_id: usize) -> Option<u64> {
        self.experts
            .get(&expert_id)
            .map(ResidentExpertWeight::payload_bytes)
    }

    pub(super) fn resident_ids(&self) -> Vec<usize> {
        self.experts.keys().copied().collect()
    }

    pub(super) fn resident_weights(&self) -> impl Iterator<Item = (usize, &W)> + '_ {
        self.experts
            .iter()
            .map(|(expert_id, weight)| (*expert_id, weight))
    }
}
