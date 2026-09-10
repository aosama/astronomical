//! Coldest-expert eviction for the decode cache.

use super::ResidentExpertWeight;
use super::demand_ledger::DecodeDemandLedger;
use super::resident_set::ResidentExpertSet;

/// One resident expert considered for eviction.
struct EvictionCandidate {
    slot: usize,
    expert_id: usize,
    demand: u64,
    payload_bytes: u64,
}

/// Slots that lost at least one expert during one eviction pass.
pub(super) struct EvictionOutcome {
    pub(super) evicted_expert_count: usize,
    pub(super) vacated_slots: Vec<usize>,
}

/// Removes the lowest demand-per-byte experts until `bytes_to_free` is met.
///
/// Protected `(slot, expert_id)` pairs survive this pass so the current
/// forward cannot lose experts it just routed.
pub(super) fn evict_coldest_experts<W: ResidentExpertWeight>(
    layers: &mut [ResidentExpertSet<W>],
    ledger: &DecodeDemandLedger,
    bytes_to_free: u64,
    protected: &[(usize, usize)],
) -> EvictionOutcome {
    if bytes_to_free == 0 {
        return EvictionOutcome {
            evicted_expert_count: 0,
            vacated_slots: Vec::new(),
        };
    }
    let mut candidates = Vec::new();
    for (slot, layer) in layers.iter().enumerate() {
        for expert_id in layer.resident_ids() {
            if protected.iter().any(|(protected_slot, protected_id)| {
                *protected_slot == slot && *protected_id == expert_id
            }) {
                continue;
            }
            let Some(payload_bytes) = layer.payload_bytes_for(expert_id) else {
                continue;
            };
            if payload_bytes == 0 {
                continue;
            }
            candidates.push(EvictionCandidate {
                slot,
                expert_id,
                demand: ledger.demand(slot, expert_id),
                payload_bytes,
            });
        }
    }
    candidates.sort_by(|left, right| {
        let left_score = u128::from(left.demand) * u128::from(right.payload_bytes);
        let right_score = u128::from(right.demand) * u128::from(left.payload_bytes);
        left_score
            .cmp(&right_score)
            .then_with(|| left.slot.cmp(&right.slot))
            .then_with(|| left.expert_id.cmp(&right.expert_id))
    });
    let mut freed_bytes = 0_u64;
    let mut evicted_expert_count = 0_usize;
    let mut vacated_slots = Vec::new();
    for candidate in candidates {
        if freed_bytes >= bytes_to_free {
            break;
        }
        if layers[candidate.slot].evict(candidate.expert_id).is_some() {
            freed_bytes = freed_bytes.saturating_add(candidate.payload_bytes);
            evicted_expert_count = evicted_expert_count.saturating_add(1);
            if !vacated_slots.contains(&candidate.slot) {
                vacated_slots.push(candidate.slot);
            }
        }
    }
    EvictionOutcome {
        evicted_expert_count,
        vacated_slots,
    }
}
