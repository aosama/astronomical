use std::fs;

use astronomical_runtime_integration::{
    MlxNativeExpertCache, MlxNativeExpertLayerDescriptor, MlxNativeExpertProjection,
};

use super::native_expert_cache_fixture::{
    EXPERT_CAPACITY, INPUT_DIMENSION, build_stacked_reference, build_two_expert_source_fixture,
};
use crate::common::runtime_test_support::runtime;

#[test]
fn should_stream_router_requested_source_ranges_directly_into_reusable_native_slots() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the native cache fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the expert source fixture should be writable");

    let layer_descriptor = MlxNativeExpertLayerDescriptor::new(EXPERT_CAPACITY, source_descriptors);
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[layer_descriptor],
        source_file_bytes.len() as u64,
    )
    .expect("the native expert cache should validate its source inventory");
    let selected_expert_indices = runtime
        .array_from_i32(&[1, 0], &[1, 2])
        .expect("the selected expert indices should be valid");

    let (cold_snapshot, cold_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &selected_expert_indices, true)
        .expect("the cold route should stream both requested experts");
    assert_eq!(cold_report.cache_miss_count(), 2);
    assert_eq!(cold_report.cache_hit_count(), 0);
    assert_eq!(
        cold_report.successful_source_read_byte_count(),
        source_file_bytes.len() as u64
    );
    assert_eq!(cold_report.payload_copy_byte_count(), 0);
    assert!(cold_report.successful_source_read_elapsed_nanoseconds() > 0);
    assert_eq!(cold_report.route_dependency_synchronization_count(), 1);
    assert!(cold_report.route_dependency_synchronization_elapsed_nanoseconds() > 0);
    assert!(cold_report.maximum_route_dependency_synchronization_elapsed_nanoseconds() > 0);
    assert!(
        cold_report.maximum_route_dependency_synchronization_elapsed_nanoseconds()
            <= cold_report.route_dependency_synchronization_elapsed_nanoseconds()
    );

    let activations = runtime
        .array_from_f32(&[1.0; INPUT_DIMENSION as usize], &[1, INPUT_DIMENSION])
        .expect("the activations should be valid");
    let native_output = cold_snapshot
        .gather_matmul(
            &runtime,
            MlxNativeExpertProjection::Up,
            &activations,
            &selected_expert_indices,
            true,
            false,
        )
        .expect("the native paged projection should build");
    let stacked_output = build_stacked_reference(&runtime, &activations, &selected_expert_indices);
    assert_eq!(
        native_output
            .to_vec_f32()
            .expect("the native output should evaluate"),
        stacked_output
            .to_vec_f32()
            .expect("the stacked output should evaluate")
    );

    let (_warm_snapshot, warm_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &selected_expert_indices, true)
        .expect("the warm route should reuse the native slots");
    assert_eq!(warm_report.cache_hit_count(), 0);
    assert_eq!(warm_report.cache_miss_count(), 0);
    assert_eq!(warm_report.successful_source_read_byte_count(), 0);
    assert_eq!(warm_report.successful_source_read_elapsed_nanoseconds(), 0);
    assert_eq!(warm_report.payload_copy_byte_count(), 0);
    assert_eq!(warm_report.page_table_publication_count(), 0);
    assert_eq!(warm_report.route_dependency_synchronization_count(), 0);
    assert_eq!(
        warm_report.route_dependency_synchronization_elapsed_nanoseconds(),
        0
    );
    assert_eq!(
        warm_report.complete_layer_route_synchronization_elision_count(),
        1
    );
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 2);
}

#[test]
fn should_retain_an_old_generation_until_gpu_completion_after_exact_native_eviction() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the native cache fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the expert source fixture should be writable");
    let bytes_per_expert = source_file_bytes.len() as u64 / EXPERT_CAPACITY as u64;
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            source_descriptors,
        )],
        bytes_per_expert,
    )
    .expect("the one-slot native cache should be valid");
    let activations = runtime
        .array_from_f32(&[1.0; INPUT_DIMENSION as usize], &[1, INPUT_DIMENSION])
        .expect("the activations should be valid");
    let first_expert_indices = runtime
        .array_from_i32(&[0], &[1])
        .expect("the first expert route should be valid");
    let (first_snapshot, _) = native_expert_cache
        .prepare_layer(&runtime, 0, &first_expert_indices, true)
        .expect("the first expert should enter the native cache");
    let first_generation_output = first_snapshot
        .gather_matmul(
            &runtime,
            MlxNativeExpertProjection::Gate,
            &activations,
            &first_expert_indices,
            true,
            false,
        )
        .expect("the first generation graph should remain lazy");

    let second_expert_indices = runtime
        .array_from_i32(&[1], &[1])
        .expect("the second expert route should be valid");
    let (second_snapshot, second_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &second_expert_indices, true)
        .expect("the second expert should replace the first cache entry");
    assert_eq!(second_report.cache_miss_count(), 1);
    assert_eq!(native_expert_cache.statistics().eviction_count(), 1);
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 1);
    assert_eq!(
        native_expert_cache
            .statistics()
            .resident_payload_byte_count(),
        bytes_per_expert
    );

    assert!(
        first_generation_output
            .to_vec_f32()
            .expect("the evicted generation should remain executable")
            .iter()
            .all(|output_value| *output_value == 64.0)
    );
    let second_generation_output = second_snapshot
        .gather_matmul(
            &runtime,
            MlxNativeExpertProjection::Gate,
            &activations,
            &second_expert_indices,
            true,
            false,
        )
        .expect("the replacement generation should build");
    assert!(
        second_generation_output
            .to_vec_f32()
            .expect("the replacement generation should evaluate")
            .iter()
            .all(|output_value| *output_value == 192.0)
    );
}

#[test]
fn should_execute_an_oversized_route_ephemerally_and_reclaim_native_retention() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the native cache fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the expert source fixture should be writable");
    let bytes_per_expert = source_file_bytes.len() as u64 / EXPERT_CAPACITY as u64;
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            source_descriptors,
        )],
        bytes_per_expert,
    )
    .expect("the one-slot native cache should be valid");
    let both_expert_indices = runtime
        .array_from_i32(&[0, 1], &[1, 2])
        .expect("the oversized route should be valid");

    let (ephemeral_snapshot, ephemeral_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &both_expert_indices, true)
        .expect("the oversized route should use an ephemeral native snapshot");
    assert_eq!(ephemeral_report.cache_miss_count(), 2);
    assert_eq!(ephemeral_report.disk_page_load_count(), 2);
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 0);
    let activations = runtime
        .array_from_f32(&[1.0; INPUT_DIMENSION as usize], &[1, INPUT_DIMENSION])
        .expect("the activations should be valid");
    let ephemeral_output = ephemeral_snapshot
        .gather_matmul(
            &runtime,
            MlxNativeExpertProjection::Down,
            &activations,
            &both_expert_indices,
            true,
            false,
        )
        .expect("the ephemeral route should remain executable");
    assert_eq!(
        ephemeral_output
            .to_vec_f32()
            .expect("the ephemeral output should evaluate"),
        build_stacked_reference(&runtime, &activations, &both_expert_indices)
            .to_vec_f32()
            .expect("the stacked output should evaluate")
    );

    let first_expert_indices = runtime
        .array_from_i32(&[0], &[1])
        .expect("the retained route should be valid");
    let (retained_snapshot, _) = native_expert_cache
        .prepare_layer(&runtime, 0, &first_expert_indices, true)
        .expect("one expert should enter retention");
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 1);
    assert!(native_expert_cache.freeze_retention_growth());
    assert!(
        native_expert_cache
            .reclaim_retained_payload_bytes(bytes_per_expert)
            .expect("native reclamation should succeed")
    );
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 0);
    assert!(
        retained_snapshot
            .gather_matmul(
                &runtime,
                MlxNativeExpertProjection::Gate,
                &activations,
                &first_expert_indices,
                true,
                false,
            )
            .expect("the reclaimed snapshot should remain executable")
            .to_vec_f32()
            .expect("the reclaimed generation should evaluate")
            .iter()
            .all(|output_value| *output_value == 64.0)
    );
    assert!(native_expert_cache.resume_retention_growth());
}

#[test]
fn should_reject_a_truncated_source_inventory_before_serving_a_route() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the native cache fixture should exist");
    let source_file_path = fixture_directory.path().join("truncated-expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(
        &source_file_path,
        &source_file_bytes[..source_file_bytes.len() - 1],
    )
    .expect("the truncated expert source fixture should be writable");

    let construction_error = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            source_descriptors,
        )],
        source_file_bytes.len() as u64,
    )
    .expect_err("source ranges outside the file must fail startup validation");
    assert!(
        construction_error
            .to_string()
            .contains("source range exceeds its file")
    );
}

#[test]
fn should_deduplicate_repeated_assignments_and_reject_out_of_range_experts() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the native cache fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the expert source fixture should be writable");
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            source_descriptors,
        )],
        source_file_bytes.len() as u64,
    )
    .expect("the native cache fixture should be valid");
    let repeated_expert_indices = runtime
        .array_from_i32(&[1, 1, 1], &[1, 3])
        .expect("the repeated route should be valid");
    let (_, repeated_route_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &repeated_expert_indices, false)
        .expect("repeated assignments should load one distinct expert");
    assert_eq!(repeated_route_report.cache_miss_count(), 1);
    assert_eq!(repeated_route_report.disk_page_load_count(), 1);
    assert_eq!(
        repeated_route_report.route_dependency_synchronization_count(),
        1
    );
    assert_eq!(
        repeated_route_report.route_dependency_synchronization_elapsed_nanoseconds(),
        0
    );
    assert_eq!(
        repeated_route_report.successful_source_read_elapsed_nanoseconds(),
        0
    );
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 1);

    let invalid_expert_indices = runtime
        .array_from_i32(&[EXPERT_CAPACITY as i32], &[1])
        .expect("the invalid expert route array should be constructible");
    let route_error = native_expert_cache
        .prepare_layer(&runtime, 0, &invalid_expert_indices, true)
        .expect_err("an expert ID at capacity must be rejected");
    assert!(route_error.to_string().contains("out-of-range expert ID"));
}

#[test]
fn should_discard_a_partial_candidate_after_a_short_read_and_serve_the_next_route() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the short-read fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the complete expert source fixture should be writable");
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            source_descriptors,
        )],
        source_file_bytes.len() as u64,
    )
    .expect("the complete source inventory should validate");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&source_file_path)
        .and_then(|source_file| source_file.set_len(source_file_bytes.len() as u64 - 1))
        .expect("the source should truncate after startup validation");
    let selected_expert_indices = runtime
        .array_from_i32(&[1], &[1])
        .expect("the selected expert route should be valid");
    native_expert_cache
        .prepare_layer(&runtime, 0, &selected_expert_indices, true)
        .expect_err("a short direct range read must reject the complete candidate slot");
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 0);

    fs::write(&source_file_path, &source_file_bytes)
        .expect("the complete expert source should be restorable");
    let (_snapshot, recovered_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &selected_expert_indices, true)
        .expect("the same native cache should serve a valid route after the failed candidate");
    assert_eq!(recovered_report.cache_miss_count(), 1);
    assert_eq!(recovered_report.disk_page_load_count(), 1);
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 1);
}

#[test]
fn should_discard_every_new_page_when_a_later_page_in_the_same_route_has_a_short_read() {
    let runtime = runtime();
    let fixture_directory =
        tempfile::tempdir().expect("the multi-page short-read fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the complete multi-page source fixture should be writable");
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            source_descriptors,
        )],
        source_file_bytes.len() as u64,
    )
    .expect("the complete multi-page source inventory should validate");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&source_file_path)
        .and_then(|source_file| source_file.set_len(source_file_bytes.len() as u64 - 1))
        .expect("the second expert source should truncate after startup validation");
    let both_expert_indices = runtime
        .array_from_i32(&[0, 1], &[1, 2])
        .expect("the two-expert route should be valid");

    native_expert_cache
        .prepare_layer(&runtime, 0, &both_expert_indices, true)
        .expect_err("a later short read must reject the complete route transaction");
    assert_eq!(
        native_expert_cache.statistics().resident_expert_count(),
        0,
        "no newly loaded page may enter cache ownership until every route page is complete",
    );

    fs::write(&source_file_path, &source_file_bytes)
        .expect("the complete multi-page source should be restorable");
    let (_snapshot, recovered_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &both_expert_indices, true)
        .expect("the complete route should load after its source is restored");
    assert_eq!(recovered_report.cache_miss_count(), 2);
    assert_eq!(recovered_report.disk_page_load_count(), 2);
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 2);
}
