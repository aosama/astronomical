//! Cross-session demand history tests: seeding, aging, accumulation across
//! generations, and the full file round-trip journey across sessions.
//!
//! Pure counter and file exercises — no graphics runtime, no serialization
//! needed. These tests pin the three rules of the history contract
//! (`demand_history`): decode-only hotness, aging with a one-generation
//! half-life, and accumulation across usage so a session starts warm.

use super::*;
use crate::qwen3_5_streaming::expert_streaming::{
    PersistedExpertUsage, expert_usage_file_path, read_expert_usage, write_expert_usage,
};

const EXPERT_CAPACITY: usize = 16;

/// Builds one layer's dense count vector from sparse sightings, matching the
/// expert-capacity sizing production recording applies on first contact.
fn dense_counts(sightings_by_expert_id: &[(usize, u64)]) -> Vec<u64> {
    let mut dense_counts = vec![0_u64; EXPERT_CAPACITY];
    for (expert_id, sighting_count) in sightings_by_expert_id {
        dense_counts[*expert_id] = *sighting_count;
    }
    dense_counts
}

#[test]
fn should_carry_seeded_decode_heat_forward_through_persist_snapshots() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.seed_expert_demand_history(PersistedExpertUsage {
        prefill_demand_counts_by_layer: vec![vec![0, 900]],
        decode_demand_counts_by_layer: vec![vec![0, 0, 4]],
    });
    cache.begin_decode_forward();
    // One in-session sighting joins the seeded warmth.
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[2]);

    let persisted_usage = cache.take_persisted_expert_usage();
    assert_eq!(
        persisted_usage.decode_demand_counts_by_layer,
        vec![dense_counts(&[(2, 5)])],
        "seeded decode heat plus in-session sightings must persist together — \
         the fused design wiped the seed at the first request's window clear, \
         leaving the file exactly one request deep"
    );
    assert_eq!(
        persisted_usage.prefill_demand_counts_by_layer,
        vec![vec![0, 450]],
        "seeded prefill breadth persists at half in the aged record even when \
         the current window is empty, but it never seeds the live window"
    );
}

#[test]
fn should_age_both_records_at_each_generation_finalization() {
    let mut cache = WarmExpertSlotTableCache::new(1);

    // Generation one: a prompt window of 2, one decode sighting.
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[0, 0]);
    cache.begin_decode_forward();
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[0]);
    let first_generation = cache.take_persisted_expert_usage();
    assert_eq!(
        first_generation.prefill_demand_counts_by_layer,
        vec![dense_counts(&[(0, 2)])],
        "a cold history persists the first window at full weight"
    );
    assert_eq!(
        first_generation.decode_demand_counts_by_layer,
        vec![dense_counts(&[(0, 1)])],
        "a cold history persists the first generation's decode sightings in full"
    );

    // Generation two re-earns the same demand: prior evidence halves, the
    // fresh window joins at full weight.
    cache.begin_prefill_demand_window();
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[0, 0]);
    cache.begin_decode_forward();
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[0]);
    let second_generation = cache.take_persisted_expert_usage();
    assert_eq!(
        second_generation.prefill_demand_counts_by_layer,
        vec![dense_counts(&[(0, 3)])],
        "prefill aging halves the prior record before adding the window: 2/2 + 2"
    );
    assert_eq!(
        second_generation.decode_demand_counts_by_layer,
        vec![dense_counts(&[(0, 1)])],
        "decode halving after generation one left 0; the fresh sighting re-climbs to 1"
    );
}

#[test]
fn should_fade_idle_experts_with_a_one_generation_half_life() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.seed_expert_demand_history(PersistedExpertUsage {
        prefill_demand_counts_by_layer: vec![Vec::new()],
        decode_demand_counts_by_layer: vec![vec![16]],
    });

    // A session that loads but never routes this expert again: each
    // generation boundary halves the untouched accumulation.
    let first_snapshot = cache.take_persisted_expert_usage();
    let second_snapshot = cache.take_persisted_expert_usage();
    let third_snapshot = cache.take_persisted_expert_usage();
    assert_eq!(
        first_snapshot.decode_demand_counts_by_layer,
        vec![vec![16]],
        "the first persistence writes the full seeded accumulation"
    );
    assert_eq!(
        second_snapshot.decode_demand_counts_by_layer,
        vec![vec![8]],
        "one idle generation halves the record"
    );
    assert_eq!(
        third_snapshot.decode_demand_counts_by_layer,
        vec![vec![4]],
        "a second idle generation halves it again"
    );
}

#[test]
fn should_reject_a_shape_mismatched_seed_and_stay_cold() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.seed_expert_demand_history(PersistedExpertUsage {
        prefill_demand_counts_by_layer: vec![vec![9], vec![9]],
        decode_demand_counts_by_layer: vec![vec![9], vec![9]],
    });

    assert_eq!(
        cache.historical_demand_count(0, 0),
        0,
        "a history written for a differently shaped model must not seed hotness"
    );
    let persisted_usage = cache.take_persisted_expert_usage();
    assert_eq!(
        persisted_usage.decode_demand_counts_by_layer,
        vec![Vec::<u64>::new()],
        "the rejected seed must not leak into the live accumulation"
    );
}

#[test]
fn should_merge_persisted_prefill_records_of_differing_expert_widths() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.seed_expert_demand_history(PersistedExpertUsage {
        prefill_demand_counts_by_layer: vec![vec![0, 0, 8]],
        decode_demand_counts_by_layer: vec![Vec::new()],
    });
    // The next request's window only touched expert 0; the aged prior record
    // still reaches expert 2 and must widen the merged record.
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[0]);

    let persisted_usage = cache.take_persisted_expert_usage();
    assert_eq!(
        persisted_usage.prefill_demand_counts_by_layer,
        vec![dense_counts(&[(0, 1), (2, 4)])],
        "the merged record widens to the aged prior evidence: window 1 plus 8/2"
    );
}

#[test]
fn should_accumulate_usage_across_sessions_through_the_persisted_file() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());

    // Session one: four decode sightings of expert 2 — the historically hot
    // threshold — and prompt breadth on expert 1 that must stay prefill-only.
    let mut first_session_cache = WarmExpertSlotTableCache::new(1);
    first_session_cache.record_expert_demand(0, EXPERT_CAPACITY, &[1]);
    first_session_cache.begin_decode_forward();
    for _ in 0..4 {
        first_session_cache.record_expert_demand(0, EXPERT_CAPACITY, &[2]);
    }
    let first_session_usage = first_session_cache.take_persisted_expert_usage();
    assert!(write_expert_usage(
        &usage_file_path,
        Some("test-model"),
        &first_session_usage
    ));

    // Session two: the file seeds warmth, the conversation continues, and
    // the second persistence accumulates on top.
    let second_session_seed =
        read_expert_usage(&usage_file_path, 1).expect("the freshly written history should reload");
    let mut second_session_cache = WarmExpertSlotTableCache::new(1);
    second_session_cache.seed_expert_demand_history(second_session_seed);
    assert_eq!(
        second_session_cache.historical_demand_count(0, 2),
        4,
        "last session's conversational depth reloads as historical hotness"
    );
    second_session_cache.begin_decode_forward();
    second_session_cache.record_expert_demand(0, EXPERT_CAPACITY, &[2]);
    // The live stream started at the seeded 4, so the sighting makes 5.
    let second_session_usage = second_session_cache.take_persisted_expert_usage();
    assert_eq!(
        second_session_usage.decode_demand_counts_by_layer,
        vec![dense_counts(&[(2, 5)])],
        "usage accumulates across sessions: seed 4 plus one more sighting"
    );
    assert!(write_expert_usage(
        &usage_file_path,
        Some("test-model"),
        &second_session_usage
    ));

    // Session three: the accumulation survived the second file write. The
    // file densifies each layer to its largest persisted expert id, so the
    // restored vector is 3 wide regardless of the 16-expert capacity the
    // live streams were sized to.
    let third_session_seed =
        read_expert_usage(&usage_file_path, 1).expect("the second history should reload");
    assert_eq!(
        third_session_seed.decode_demand_counts_by_layer,
        vec![vec![0, 0, 5]],
        "heat keeps accumulating across sessions instead of resetting to the last request"
    );
    assert_eq!(
        third_session_seed.prefill_demand_counts_by_layer,
        vec![Vec::<u64>::new()],
        "prefill evidence ages independently: one sighting from two \
         generations ago floors to zero under integer halving, and an \
         all-zero record persists as the empty sparse map it round-trips as"
    );
}
