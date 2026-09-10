//! Per-layer per-expert route counts. Ranks eviction. Never causes I/O.

/// Decode-owned demand evidence keyed by sparse slot then expert id.
#[derive(Debug)]
pub(super) struct DecodeDemandLedger {
    counts_by_slot: Vec<Vec<u64>>,
}

impl DecodeDemandLedger {
    pub(super) fn new(sparse_layer_count: usize) -> Self {
        Self {
            counts_by_slot: vec![Vec::new(); sparse_layer_count],
        }
    }

    pub(super) fn record(&mut self, slot: usize, routed_ids: &[usize]) {
        let Some(slot_counts) = self.counts_by_slot.get_mut(slot) else {
            return;
        };
        for expert_id in routed_ids {
            if slot_counts.len() <= *expert_id {
                slot_counts.resize(expert_id.saturating_add(1), 0);
            }
            slot_counts[*expert_id] = slot_counts[*expert_id].saturating_add(1);
        }
    }

    pub(super) fn demand(&self, slot: usize, expert_id: usize) -> u64 {
        self.counts_by_slot
            .get(slot)
            .and_then(|slot_counts| slot_counts.get(expert_id))
            .copied()
            .unwrap_or(0)
    }
}
