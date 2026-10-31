//! Cross-session demand history: seeding from the persisted file, the
//! insert-time historical hotness lookup, and the aged persistence snapshot
//! taken at every generation finalization.
//!
//! The history's job is to make a session start warm: decode's first touch
//! of an expert is always a storage miss, so without persisted heat the
//! warm tables rebuild their hot set from zero every launch. Three rules
//! keep that heat honest:
//!
//! 1. Only the decode record seeds hotness — prompt breadth must never
//!    masquerade as conversational heat (the version-1 defect).
//! 2. Both records age: a generation's persistence halves prior evidence
//!    before adding the current window, an integer half-life of one idle
//!    generation. Without aging, additive counts would pin week-old topics
//!    forever; without accumulation, the history would forget everything
//!    but the final request (the other version-1 defect: the fused counter
//!    was cleared at every request's prefill window, so the file held
//!    exactly one generation of evidence).
//! 3. The decode stream is the single source of truth for its record: it is
//!    seeded at load, accumulates across requests, is written in full at
//!    finalization, and is then halved in place — so the file at any
//!    generation boundary is exactly the live accumulation at that moment.

use crate::qwen3_5_streaming::expert_streaming::PersistedExpertUsage;

use super::WarmExpertSlotTableCache;

/// Persisted decode demand a first admission must show before it is treated
/// as historically hot and skips probation: high enough that noise from one
/// prompt does not pin experts, low enough that a hot expert from a previous
/// session is warm from its first touch.
///
/// Only the decode record qualifies. A long prompt's route union is wide and
/// shallow — many experts touched a handful of times — and treating that
/// breadth as heat let document-heavy workloads crowd the narrow, deep hot
/// set a conversation actually re-uses.
pub(super) const HISTORICAL_HOT_DEMAND_COUNT: u64 = 4;

impl WarmExpertSlotTableCache {
    /// Seeds lifetime decode ranking and insert-time hotness from a
    /// persisted usage history.
    ///
    /// The decode record joins the live decode stream (so the residency
    /// ranking starts warm instead of at the first-touch floor) and is kept
    /// as a frozen copy for insert-time hotness checks — in-session
    /// promotion must come from live reuse, not from the seed. The prefill
    /// record never seeds live counters: prefill evidence is per-request
    /// admission input, and its history only matters at persistence time,
    /// where it is aged and merged.
    pub fn seed_expert_demand_history(&mut self, persisted_usage: PersistedExpertUsage) {
        let layer_count = self.tables_by_layer.len();
        if !persisted_usage_matches_layer_shape(&persisted_usage, layer_count) {
            tracing::warn!(
                seeded_layer_count = persisted_usage.decode_demand_counts_by_layer.len(),
                expected_layer_count = layer_count,
                "expert usage history shape mismatch; starting cold"
            );
            return;
        }
        let seeded_expert_count = seeded_nonzero_expert_count(&persisted_usage);
        for (layer_index, historical_decode_layer_counts) in persisted_usage
            .decode_demand_counts_by_layer
            .iter()
            .enumerate()
        {
            merge_layer_counts(
                &mut self.decode_demand_counts_by_layer[layer_index],
                historical_decode_layer_counts,
            );
        }
        self.persisted_prefill_demand_counts_by_layer =
            persisted_usage.prefill_demand_counts_by_layer;
        self.persisted_decode_demand_counts_by_layer =
            persisted_usage.decode_demand_counts_by_layer;
        tracing::info!(
            seeded_expert_count,
            "seeded expert warm tables from persisted usage history"
        );
    }

    /// Persisted decode demand for one expert, zero when no history was
    /// seeded. Reads the frozen-at-load copy: historically hot is a property
    /// of past sessions, and in-session warmth is earned through the live
    /// reuse scores instead.
    pub fn historical_demand_count(&self, layer_index: usize, expert_id: usize) -> u64 {
        self.persisted_decode_demand_counts_by_layer
            .get(layer_index)
            .and_then(|layer_counts| layer_counts.get(expert_id))
            .copied()
            .unwrap_or(0)
    }

    /// Takes the phase-split records to persist at one generation's
    /// finalization, applying the aging rules in place:
    ///
    /// - The prefill record becomes half the previously persisted record
    ///   plus the request's window — prompt evidence fades unless re-earned.
    /// - The decode record is the live accumulation in full, and the live
    ///   stream is then halved so unused experts fade with a one-generation
    ///   half-life while re-used experts re-climb immediately.
    pub fn take_persisted_expert_usage(&mut self) -> PersistedExpertUsage {
        let prefill_demand_counts_by_layer = aged_prefill_record(
            &self.persisted_prefill_demand_counts_by_layer,
            &self.prefill_demand_counts_by_layer,
        );
        let decode_demand_counts_by_layer = self.decode_demand_counts_by_layer.clone();
        self.persisted_prefill_demand_counts_by_layer = prefill_demand_counts_by_layer.clone();
        halve_all_layer_counts(&mut self.decode_demand_counts_by_layer);
        PersistedExpertUsage {
            prefill_demand_counts_by_layer,
            decode_demand_counts_by_layer,
        }
    }
}

/// Pure: whether both phase records carry exactly the expected layer count.
fn persisted_usage_matches_layer_shape(
    persisted_usage: &PersistedExpertUsage,
    layer_count: usize,
) -> bool {
    persisted_usage.decode_demand_counts_by_layer.len() == layer_count
        && persisted_usage.prefill_demand_counts_by_layer.len() == layer_count
}

/// Pure: how many experts carry nonzero evidence across both records — the
/// seeded warmth telemetry.
fn seeded_nonzero_expert_count(persisted_usage: &PersistedExpertUsage) -> u64 {
    let mut seeded_expert_count = 0_u64;
    for layer_counts in persisted_usage
        .decode_demand_counts_by_layer
        .iter()
        .chain(persisted_usage.prefill_demand_counts_by_layer.iter())
    {
        for demand_count in layer_counts {
            if *demand_count > 0 {
                seeded_expert_count = seeded_expert_count.saturating_add(1);
            }
        }
    }
    seeded_expert_count
}

/// Pure: adds a source layer's counts into a destination layer, sizing the
/// destination to the source on first contact.
fn merge_layer_counts(destination_layer_counts: &mut Vec<u64>, source_layer_counts: &[u64]) {
    if destination_layer_counts.len() < source_layer_counts.len() {
        destination_layer_counts.resize(source_layer_counts.len(), 0);
    }
    for (expert_id, source_demand_count) in source_layer_counts.iter().enumerate() {
        let destination_demand_count = &mut destination_layer_counts[expert_id];
        *destination_demand_count = destination_demand_count.saturating_add(*source_demand_count);
    }
}

/// Pure: halves every count in place (floor division) — one aging step.
fn halve_all_layer_counts(demand_counts_by_layer: &mut [Vec<u64>]) {
    for layer_counts in demand_counts_by_layer.iter_mut() {
        for demand_count in layer_counts.iter_mut() {
            *demand_count /= 2;
        }
    }
}

/// Pure: the next persisted prefill record — half the previously persisted
/// record plus the current request's window, sized to the wider of the two.
fn aged_prefill_record(
    persisted_prefill_counts_by_layer: &[Vec<u64>],
    prefill_window_counts_by_layer: &[Vec<u64>],
) -> Vec<Vec<u64>> {
    let mut aged_counts_by_layer = Vec::with_capacity(prefill_window_counts_by_layer.len());
    for (layer_index, prefill_window_layer_counts) in
        prefill_window_counts_by_layer.iter().enumerate()
    {
        let mut aged_layer_counts = prefill_window_layer_counts.clone();
        if let Some(persisted_layer_counts) = persisted_prefill_counts_by_layer.get(layer_index) {
            for (expert_id, persisted_demand_count) in persisted_layer_counts.iter().enumerate() {
                if expert_id >= aged_layer_counts.len() {
                    aged_layer_counts.resize(expert_id + 1, 0);
                }
                aged_layer_counts[expert_id] =
                    aged_layer_counts[expert_id].saturating_add(persisted_demand_count / 2);
            }
        }
        aged_counts_by_layer.push(aged_layer_counts);
    }
    aged_counts_by_layer
}
