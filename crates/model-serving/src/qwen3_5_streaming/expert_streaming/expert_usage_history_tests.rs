use super::*;

use std::fs;

fn should_round_trip_sparse_demand_counts_impl() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());
    let demand_counts_by_layer = vec![vec![0, 4, 0, 9], vec![2, 0, 0, 0]];

    assert!(write_expert_usage(
        &usage_file_path,
        Some("test-model"),
        &demand_counts_by_layer,
    ));
    let restored_counts = read_expert_usage(&usage_file_path, 2)
        .expect("a freshly written history should round-trip");

    assert_eq!(restored_counts, demand_counts_by_layer);
    assert!(!usage_file_path.with_extension("json.partial").exists());
}

fn should_reject_a_history_written_for_a_different_layer_count_impl() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());
    write_expert_usage(&usage_file_path, Some("test-model"), &[vec![1, 2]]);

    assert!(
        read_expert_usage(&usage_file_path, 3).is_none(),
        "a shape-mismatched history must not seed the cache"
    );
}

fn should_reject_a_corrupt_history_file_impl() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());
    fs::write(&usage_file_path, b"{ not json").expect("the corrupt file should be written");

    assert!(read_expert_usage(&usage_file_path, 1).is_none());
}

fn should_treat_a_missing_history_as_a_cold_start_impl() {
    let usage_directory = tempfile::tempdir().expect("the usage directory should be created");
    let usage_file_path = expert_usage_file_path(usage_directory.path());

    assert!(read_expert_usage(&usage_file_path, 2).is_none());
}

#[test]
fn should_round_trip_sparse_demand_counts() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    should_round_trip_sparse_demand_counts_impl();
}

#[test]
fn should_reject_a_history_written_for_a_different_layer_count() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    should_reject_a_history_written_for_a_different_layer_count_impl();
}

#[test]
fn should_reject_a_corrupt_history_file() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    should_reject_a_corrupt_history_file_impl();
}

#[test]
fn should_treat_a_missing_history_as_a_cold_start() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    should_treat_a_missing_history_as_a_cold_start_impl();
}
