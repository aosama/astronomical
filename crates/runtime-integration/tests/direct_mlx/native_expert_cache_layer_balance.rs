use std::fs;

use astronomical_runtime_integration::{MlxNativeExpertCache, MlxNativeExpertLayerDescriptor};

use super::native_expert_cache_fixture::{EXPERT_CAPACITY, build_two_expert_source_fixture};
use crate::common::runtime_test_support::runtime;

// These tests distinguish the one hard global byte ceiling from per-layer
// fairness. A layer may borrow unused bytes, but it may not destroy another
// populated layer's protected decode working set merely to retain a broad route.

#[test]
fn should_preserve_each_layers_decode_working_set_when_a_broad_route_exceeds_its_proportional_share()
 {
    let runtime = runtime();
    let fixture_directory =
        tempfile::tempdir().expect("the layer-balanced cache fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the layer-balanced source fixture should be writable");
    let bytes_per_expert = source_file_bytes.len() as u64 / EXPERT_CAPACITY as u64;
    let layer_descriptor = MlxNativeExpertLayerDescriptor::new(EXPERT_CAPACITY, source_descriptors);
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[layer_descriptor.clone(), layer_descriptor],
        bytes_per_expert * 2,
    )
    .expect("the two-layer cache should retain one expert per layer");
    let first_expert_indices = runtime
        .array_from_i32(&[0], &[1])
        .expect("the decode-sized route should be valid");

    for layer_index in 0..2 {
        native_expert_cache
            .prepare_layer(&runtime, layer_index, &first_expert_indices, true)
            .expect("each layer should establish its decode working set");
    }
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 2);

    let broad_first_layer_route = runtime
        .array_from_i32(&[0, 1], &[1, 2])
        .expect("the broad first-layer route should be valid");
    let (_broad_snapshot, broad_route_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &broad_first_layer_route, true)
        .expect("the broad route should execute without displacing another layer");
    assert_eq!(broad_route_report.cache_hit_count(), 1);
    assert_eq!(broad_route_report.cache_miss_count(), 1);
    assert_eq!(broad_route_report.disk_page_load_count(), 1);
    assert_eq!(
        native_expert_cache.statistics().resident_expert_count(),
        2,
        "a broad route may use an ephemeral page but must not consume another layer's proportional retention share"
    );

    let (_second_layer_snapshot, second_layer_report) = native_expert_cache
        .prepare_layer(&runtime, 1, &first_expert_indices, true)
        .expect("the second layer decode route should remain warm");
    assert_eq!(second_layer_report.cache_hit_count(), 1);
    assert_eq!(second_layer_report.cache_miss_count(), 0);
    assert_eq!(second_layer_report.disk_page_load_count(), 0);
}

#[test]
fn should_not_rebalance_layers_when_the_global_ceiling_still_fits() {
    let runtime = runtime();
    let fixture_directory =
        tempfile::tempdir().expect("the layer-balanced cache fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the layer-balanced source fixture should be writable");
    let bytes_per_expert = source_file_bytes.len() as u64 / EXPERT_CAPACITY as u64;
    let layer_descriptor = MlxNativeExpertLayerDescriptor::new(EXPERT_CAPACITY, source_descriptors);
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[layer_descriptor.clone(), layer_descriptor],
        bytes_per_expert * 4,
    )
    .expect("the two-layer cache should accept the initial ceiling");
    let both_first_layer_experts = runtime
        .array_from_i32(&[0, 1], &[1, 2])
        .expect("the complete first-layer route should be valid");

    native_expert_cache
        .prepare_layer(&runtime, 0, &both_first_layer_experts, true)
        .expect("both first-layer experts should enter retention");
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 2);

    native_expert_cache
        .update_maximum_resident_payload_byte_count(bytes_per_expert * 2)
        .expect("the lower global ceiling should fit the existing payload");

    let statistics_after_global_ceiling_update = native_expert_cache.statistics();
    assert_eq!(
        statistics_after_global_ceiling_update.resident_expert_count(),
        2,
        "global enforcement must not evict a page when total resident bytes already fit",
    );
    assert_eq!(
        statistics_after_global_ceiling_update.resident_payload_byte_count(),
        bytes_per_expert * 2,
    );
}

#[test]
fn should_borrow_unused_layer_capacity_for_a_hot_route() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the layer-borrowing fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the layer-borrowing source fixture should be writable");
    let bytes_per_expert = source_file_bytes.len() as u64 / EXPERT_CAPACITY as u64;
    let layer_descriptor = MlxNativeExpertLayerDescriptor::new(EXPERT_CAPACITY, source_descriptors);
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[layer_descriptor.clone(), layer_descriptor],
        bytes_per_expert * 2,
    )
    .expect("the cache should provide two globally available expert slots");
    let both_first_layer_experts = runtime
        .array_from_i32(&[0, 1], &[1, 2])
        .expect("the broad first-layer route should be valid");

    let (_snapshot, cold_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &both_first_layer_experts, true)
        .expect("the hot layer should borrow the unused second-layer capacity");
    assert_eq!(cold_report.disk_page_load_count(), 2);
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 2);

    let (_warm_snapshot, warm_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &both_first_layer_experts, true)
        .expect("the borrowed hot route should remain warm");
    assert_eq!(warm_report.disk_page_load_count(), 0);
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 2);
}

#[test]
fn should_enforce_a_lower_cache_ceiling_while_retention_growth_is_frozen() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the frozen-retention fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the frozen-retention source fixture should be writable");
    let bytes_per_expert = source_file_bytes.len() as u64 / EXPERT_CAPACITY as u64;
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            source_descriptors,
        )],
        bytes_per_expert * 2,
    )
    .expect("the native cache should retain both experts initially");
    let both_expert_indices = runtime
        .array_from_i32(&[0, 1], &[1, 2])
        .expect("the complete expert route should be valid");
    native_expert_cache
        .prepare_layer(&runtime, 0, &both_expert_indices, true)
        .expect("both experts should enter retention before pressure");
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 2);
    assert!(native_expert_cache.freeze_retention_growth());

    native_expert_cache
        .update_maximum_resident_payload_byte_count(bytes_per_expert)
        .expect("a lower route-specific ceiling should be enforceable during pressure");

    let constrained_cache_statistics = native_expert_cache.statistics();
    assert_eq!(
        constrained_cache_statistics.maximum_resident_payload_byte_count(),
        bytes_per_expert
    );
    assert_eq!(
        constrained_cache_statistics.resident_payload_byte_count(),
        bytes_per_expert,
        "retained expert payload must never remain above a lower live ceiling"
    );
    assert_eq!(constrained_cache_statistics.resident_expert_count(), 1);
    assert!(native_expert_cache.resume_retention_growth());
    assert_eq!(
        native_expert_cache
            .statistics()
            .maximum_resident_payload_byte_count(),
        bytes_per_expert,
        "resume must preserve the newest configured ceiling"
    );
}

#[test]
fn should_defer_a_higher_cache_ceiling_until_frozen_retention_growth_resumes() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the frozen-retention fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the frozen-retention source fixture should be writable");
    let bytes_per_expert = source_file_bytes.len() as u64 / EXPERT_CAPACITY as u64;
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            source_descriptors,
        )],
        bytes_per_expert,
    )
    .expect("the native cache should start with one expert of retention");
    let first_expert_indices = runtime
        .array_from_i32(&[0], &[1])
        .expect("the first expert route should be valid");
    native_expert_cache
        .prepare_layer(&runtime, 0, &first_expert_indices, true)
        .expect("one expert should enter retention before pressure");
    assert!(native_expert_cache.freeze_retention_growth());

    native_expert_cache
        .update_maximum_resident_payload_byte_count(bytes_per_expert * 2)
        .expect("a higher configured ceiling should be remembered during pressure");
    assert_eq!(
        native_expert_cache
            .statistics()
            .maximum_resident_payload_byte_count(),
        bytes_per_expert,
        "frozen retention must not grow before explicit resume"
    );

    assert!(native_expert_cache.resume_retention_growth());
    assert_eq!(
        native_expert_cache
            .statistics()
            .maximum_resident_payload_byte_count(),
        bytes_per_expert * 2
    );
}
