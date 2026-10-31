//! Persistence tests for the phase-split expert usage history file: version
//! two round-trips, version one and corrupt inputs are rejected as cold
//! starts, and each phase is capped independently.

use super::*;

use std::fs;

fn sample_persisted_usage() -> PersistedExpertUsage {
    PersistedExpertUsage {
        // A prompt-heavy session left broad prefill evidence on layers 0-1.
        prefill_demand_counts_by_layer: vec![vec![0, 4, 0, 9], vec![2, 0, 0, 0]],
        // The conversation itself kept re-using different experts.
        decode_demand_counts_by_layer: vec![vec![7, 0, 0, 1], vec![0, 0, 5, 0]],
    }
}

#[test]
fn should_round_trip_both_phase_records_disjointly() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());
    let persisted_usage = sample_persisted_usage();

    assert!(write_expert_usage(
        &usage_file_path,
        Some("test-model"),
        &persisted_usage
    ));
    let restored_usage = read_expert_usage(&usage_file_path, 2)
        .expect("a freshly written history should round-trip");

    assert_eq!(restored_usage, persisted_usage);
    assert_ne!(
        restored_usage.prefill_demand_counts_by_layer, restored_usage.decode_demand_counts_by_layer,
        "the two phase records must stay disjoint through the file format"
    );
    assert!(!usage_file_path.with_extension("json.partial").exists());
}

#[test]
fn should_reject_a_version_one_history_file() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());
    // A version-1 file as the previous format wrote it: one fused counter set
    // that cannot be attributed back to a phase.
    let version_one_contents = concat!(
        r#"{"format_version":1,"model_id":null,"layer_count":1,"layers":[{"layer_index":0,"#,
        r#""demand_counts_by_expert_id":{"0":9,"1":7}}]}"#
    );
    fs::write(&usage_file_path, version_one_contents)
        .expect("the version-one file should be written");

    assert!(
        read_expert_usage(&usage_file_path, 1).is_none(),
        "fused version-one counts must not seed phase-split records"
    );
}

#[test]
fn should_reject_a_history_written_for_a_different_layer_count() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());
    write_expert_usage(
        &usage_file_path,
        Some("test-model"),
        &sample_persisted_usage(),
    );

    assert!(
        read_expert_usage(&usage_file_path, 3).is_none(),
        "a shape-mismatched history must not seed the cache"
    );
}

#[test]
fn should_reject_a_corrupt_history_file() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());
    fs::write(&usage_file_path, b"{ not json").expect("the corrupt file should be written");

    assert!(read_expert_usage(&usage_file_path, 1).is_none());
}

#[test]
fn should_treat_a_missing_history_as_a_cold_start() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());

    assert!(read_expert_usage(&usage_file_path, 2).is_none());
}

#[test]
fn should_treat_a_zero_layer_model_as_a_cold_start() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());
    write_expert_usage(
        &usage_file_path,
        Some("test-model"),
        &sample_persisted_usage(),
    );

    assert!(
        read_expert_usage(&usage_file_path, 0).is_none(),
        "a zero-layer read is a shape mismatch, not a warm start"
    );
}

#[test]
fn should_cap_each_phase_independently_per_layer() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());
    // One layer where the prefill union explodes past the cap while decode
    // depth stays far below it: an extreme prompt-heavy boundary.
    let oversize_prefill_layer: Vec<u64> = (0..6_000).map(|_| 3).collect();
    let modest_decode_layer: Vec<u64> = vec![0, 42];
    let persisted_usage = PersistedExpertUsage {
        prefill_demand_counts_by_layer: vec![oversize_prefill_layer],
        decode_demand_counts_by_layer: vec![modest_decode_layer],
    };

    assert!(write_expert_usage(
        &usage_file_path,
        Some("test-model"),
        &persisted_usage
    ));
    let restored_usage =
        read_expert_usage(&usage_file_path, 1).expect("the capped history should round-trip");

    let restored_prefill_nonzero_count = restored_usage.prefill_demand_counts_by_layer[0]
        .iter()
        .filter(|demand_count| **demand_count > 0)
        .count();
    assert_eq!(
        restored_prefill_nonzero_count, 4_096,
        "the prefill record is capped at the per-layer persistence limit"
    );
    assert_eq!(
        restored_usage.decode_demand_counts_by_layer[0],
        vec![0, 42],
        "the decode record keeps every expert despite the prefill record capping"
    );
}
