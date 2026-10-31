//! Phase-split demand evidence tests: recording routing into the correct
//! stream, the prefill weight scoping, and the prefill-window lifecycle.
//!
//! These tests are pure counter exercises — no graphics runtime — so they
//! run without serialization; state inspections go through
//! `take_persisted_expert_usage`, which snapshots both streams at once.

use super::*;
use crate::qwen3_5_streaming::expert_streaming::PersistedExpertUsage;

/// Builds one layer's dense count vector from sparse sightings, matching the
/// expert-capacity sizing production recording applies on first contact.
fn dense_counts(sightings_by_expert_id: &[(usize, u64)]) -> Vec<u64> {
    let mut dense_counts = vec![0_u64; EXPERT_CAPACITY];
    for (expert_id, sighting_count) in sightings_by_expert_id {
        dense_counts[*expert_id] = *sighting_count;
    }
    dense_counts
}

const EXPERT_CAPACITY: usize = 16;

#[test]
fn should_record_each_phase_into_its_own_stream() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    // Default phase is Prefill: every generation opens by processing its
    // prompt before any generated token exists.
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[7]);
    cache.begin_decode_forward();
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[7, 9]);

    let persisted_usage = cache.take_persisted_expert_usage();
    assert_eq!(
        persisted_usage.prefill_demand_counts_by_layer,
        vec![dense_counts(&[(7, 1)])],
        "expert 7's prefill sighting must land only in the prefill record"
    );
    assert_eq!(
        persisted_usage.decode_demand_counts_by_layer,
        vec![dense_counts(&[(7, 1), (9, 1)])],
        "expert 7 and 9's decode sightings must land only in the decode record"
    );
}

#[test]
fn should_apply_the_assignment_weight_to_prefill_evidence_only() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    // The final-chunk boost multiplier scales prefill evidence...
    cache.set_prefill_demand_assignment_weight(3);
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[5]);
    // ...but decode evidence records at plain weight: the phase switch
    // restores plain weight itself.
    cache.begin_decode_forward();
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[5, 6]);

    let persisted_usage = cache.take_persisted_expert_usage();
    assert_eq!(
        persisted_usage.prefill_demand_counts_by_layer,
        vec![dense_counts(&[(5, 3)])],
        "prefill evidence carries the chunk weight"
    );
    assert_eq!(
        persisted_usage.decode_demand_counts_by_layer,
        vec![dense_counts(&[(5, 1), (6, 1)])],
        "decode evidence ignores the boost and stays comparable across tokens"
    );
}

#[test]
fn should_clear_only_the_prefill_window_between_requests() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[2, 2]);
    cache.begin_decode_forward();
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[1, 1, 1]);

    // The next request opens its prefill window: prompt evidence is spent,
    // while the lifetime decode stream must survive (issue #955 — this is
    // the exact clear that, in the fused-counter design, also destroyed the
    // decode evidence and left the persisted file one request deep).
    cache.begin_prefill_demand_window();

    let persisted_usage = cache.take_persisted_expert_usage();
    assert_eq!(
        persisted_usage.prefill_demand_counts_by_layer,
        vec![dense_counts(&[])]
    );
    assert_eq!(
        persisted_usage.decode_demand_counts_by_layer,
        vec![dense_counts(&[(1, 3)])],
        "lifetime decode evidence survives the window clear"
    );
}

#[test]
fn should_floor_a_configured_zero_weight_at_one_sighting() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.set_prefill_demand_assignment_weight(0);
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[1]);

    let persisted_usage = cache.take_persisted_expert_usage();
    assert_eq!(
        persisted_usage.prefill_demand_counts_by_layer,
        vec![dense_counts(&[(1, 1)])],
        "a zero weight must not silently discard real routing evidence"
    );
}

#[test]
fn should_ignore_recording_beyond_the_model_layer_shape() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.record_expert_demand(5, EXPERT_CAPACITY, &[1]);
    cache.begin_decode_forward();
    cache.record_expert_demand(5, EXPERT_CAPACITY, &[1]);

    let persisted_usage = cache.take_persisted_expert_usage();
    assert_eq!(
        persisted_usage.prefill_demand_counts_by_layer,
        vec![Vec::<u64>::new()],
        "an out-of-shape layer index must be dropped, not panicky"
    );
    assert_eq!(
        persisted_usage.decode_demand_counts_by_layer,
        vec![Vec::<u64>::new()],
        "an out-of-shape layer index must be dropped, not panicky"
    );
}

#[test]
fn should_report_historical_hotness_from_the_decode_record_only() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    // A prompt-heavy past session left expert 1 with broad prefill evidence
    // while expert 2 carried genuine conversational depth.
    cache.seed_expert_demand_history(PersistedExpertUsage {
        prefill_demand_counts_by_layer: vec![vec![0, 400]],
        decode_demand_counts_by_layer: vec![vec![0, 0, 4]],
    });

    assert_eq!(
        cache.historical_demand_count(0, 1),
        0,
        "prefill breadth must never masquerade as conversational heat"
    );
    assert_eq!(
        cache.historical_demand_count(0, 2),
        4,
        "decode depth is the only seed of historical hotness"
    );
}
