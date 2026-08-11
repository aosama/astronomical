use std::fs;

use astronomical_runtime_integration::{MlxNativeExpertCache, MlxNativeExpertLayerDescriptor};

use super::native_expert_cache::{
    EXPERT_CAPACITY, INPUT_DIMENSION, build_stacked_reference, build_two_expert_source_fixture,
};
use crate::common::runtime_test_support::runtime;

#[test]
fn should_reconcile_elided_complete_layer_routes_before_exact_global_eviction() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the complete-layer fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the complete-layer source fixture should be writable");
    let bytes_per_expert = source_file_bytes.len() as u64 / EXPERT_CAPACITY as u64;
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[
            MlxNativeExpertLayerDescriptor::new(2, source_descriptors.clone()),
            MlxNativeExpertLayerDescriptor::new(2, source_descriptors),
        ],
        bytes_per_expert * 3,
    )
    .expect("the two-layer native cache should be valid");
    let both_experts = runtime
        .array_from_i32(&[0, 1], &[1, 2])
        .expect("the complete layer route should be valid");
    native_expert_cache
        .prepare_layer(&runtime, 0, &both_experts, true)
        .expect("the first layer should become complete through router evidence");
    let first_expert = runtime
        .array_from_i32(&[0], &[1])
        .expect("the first expert route should be valid");
    native_expert_cache
        .prepare_layer(&runtime, 1, &first_expert, true)
        .expect("the second layer should retain its first expert");

    let (_, elided_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &first_expert, true)
        .expect("the complete layer should defer route reconciliation");
    assert_eq!(
        elided_report.complete_layer_route_synchronization_elision_count(),
        1
    );
    let second_expert = runtime
        .array_from_i32(&[1], &[1])
        .expect("the second expert route should be valid");
    native_expert_cache
        .prepare_layer(&runtime, 1, &second_expert, true)
        .expect("the next miss should reconcile deferred recency before eviction");
    assert_eq!(native_expert_cache.statistics().eviction_count(), 1);

    let (_, recently_used_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &first_expert, true)
        .expect("the deferred recently used expert should survive eviction");
    assert_eq!(recently_used_report.cache_hit_count(), 1);
    let (_, oldest_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &second_expert, true)
        .expect("the deferred layer's oldest expert should reload");
    assert_eq!(oldest_report.cache_miss_count(), 1);
}

#[test]
fn should_match_stacked_gather_qmm_for_sorted_multi_token_prefill() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the sorted prefill fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the sorted prefill source fixture should be writable");
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(2, source_descriptors)],
        source_file_bytes.len() as u64,
    )
    .expect("the sorted prefill native cache should be valid");
    let selected_expert_values = [vec![0; 32], vec![1; 32]].concat();
    let selected_expert_indices = runtime
        .array_from_i32(&selected_expert_values, &[64])
        .expect("the sorted prefill route should be valid");
    let (native_snapshot, _) = native_expert_cache
        .prepare_layer(&runtime, 0, &selected_expert_indices, true)
        .expect("the sorted prefill route should load both experts");
    let activations = runtime
        .array_from_f32(
            &vec![1.0; 64 * INPUT_DIMENSION as usize],
            &[64, 1, INPUT_DIMENSION],
        )
        .expect("the sorted prefill activations should be valid");
    let native_output = native_snapshot
        .gather_matmul(
            &runtime,
            astronomical_runtime_integration::MlxNativeExpertProjection::Up,
            &activations,
            &selected_expert_indices,
            true,
            true,
        )
        .expect("the sorted native projection should build");
    let stacked_output = build_stacked_reference(&runtime, &activations, &selected_expert_indices);
    assert_eq!(
        native_output
            .to_vec_f32()
            .expect("the sorted native output should evaluate"),
        stacked_output
            .to_vec_f32()
            .expect("the sorted stacked output should evaluate")
    );
}
