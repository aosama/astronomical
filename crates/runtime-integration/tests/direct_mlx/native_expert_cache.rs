use std::fs;

use astronomical_runtime_integration::{
    MlxDtype, MlxNativeExpertCache, MlxNativeExpertLayerDescriptor, MlxNativeExpertParameter,
    MlxNativeExpertProjection, MlxNativeExpertTensorSourceDescriptor,
};

use crate::common::runtime_test_support::runtime;

pub(super) const EXPERT_CAPACITY: usize = 2;
pub(super) const INPUT_DIMENSION: i32 = 64;
pub(super) const OUTPUT_DIMENSION: i32 = 64;
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const QUANTIZATION_BITS: i32 = 4;
const PACKED_WORDS_PER_OUTPUT_ROW: i32 = INPUT_DIMENSION * QUANTIZATION_BITS / i32::BITS as i32;
const PACKED_WORDS_PER_EXPERT: usize = (OUTPUT_DIMENSION * PACKED_WORDS_PER_OUTPUT_ROW) as usize;
const QUANTIZATION_VALUES_PER_EXPERT: usize = OUTPUT_DIMENSION as usize;

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

pub(super) fn build_two_expert_source_fixture(
    source_file_path: &std::path::Path,
) -> (Vec<u8>, Vec<MlxNativeExpertTensorSourceDescriptor>) {
    let mut source_file_bytes = Vec::new();
    let mut source_descriptors = Vec::new();
    for projection in [
        MlxNativeExpertProjection::Gate,
        MlxNativeExpertProjection::Up,
        MlxNativeExpertProjection::Down,
    ] {
        let packed_weight_offset = source_file_bytes.len() as u64;
        append_u32_values(
            &mut source_file_bytes,
            &[
                vec![0x1111_1111; PACKED_WORDS_PER_EXPERT],
                vec![0x3333_3333; PACKED_WORDS_PER_EXPERT],
            ]
            .concat(),
        );
        source_descriptors.push(MlxNativeExpertTensorSourceDescriptor::new(
            projection,
            MlxNativeExpertParameter::PackedWeight,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
            source_file_path.to_path_buf(),
            packed_weight_offset,
            PACKED_WORDS_PER_EXPERT * u32::BITS as usize / 8,
            vec![1, OUTPUT_DIMENSION, PACKED_WORDS_PER_OUTPUT_ROW],
            MlxDtype::UInt32,
        ));

        let scales_offset = source_file_bytes.len() as u64;
        append_f32_values(
            &mut source_file_bytes,
            &vec![1.0; QUANTIZATION_VALUES_PER_EXPERT * EXPERT_CAPACITY],
        );
        source_descriptors.push(MlxNativeExpertTensorSourceDescriptor::new(
            projection,
            MlxNativeExpertParameter::Scales,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
            source_file_path.to_path_buf(),
            scales_offset,
            QUANTIZATION_VALUES_PER_EXPERT * size_of::<f32>(),
            vec![1, OUTPUT_DIMENSION, 1],
            MlxDtype::Float32,
        ));

        let biases_offset = source_file_bytes.len() as u64;
        append_f32_values(
            &mut source_file_bytes,
            &vec![0.0; QUANTIZATION_VALUES_PER_EXPERT * EXPERT_CAPACITY],
        );
        source_descriptors.push(MlxNativeExpertTensorSourceDescriptor::new(
            projection,
            MlxNativeExpertParameter::Biases,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
            source_file_path.to_path_buf(),
            biases_offset,
            QUANTIZATION_VALUES_PER_EXPERT * size_of::<f32>(),
            vec![1, OUTPUT_DIMENSION, 1],
            MlxDtype::Float32,
        ));
    }
    (source_file_bytes, source_descriptors)
}

pub(super) fn build_stacked_reference(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    activations: &astronomical_runtime_integration::MlxArray,
    selected_expert_indices: &astronomical_runtime_integration::MlxArray,
) -> astronomical_runtime_integration::MlxArray {
    let stacked_packed_weights = runtime
        .array_from_u32(
            &[
                vec![0x1111_1111; PACKED_WORDS_PER_EXPERT],
                vec![0x3333_3333; PACKED_WORDS_PER_EXPERT],
            ]
            .concat(),
            &[
                EXPERT_CAPACITY as i32,
                OUTPUT_DIMENSION,
                PACKED_WORDS_PER_OUTPUT_ROW,
            ],
        )
        .expect("the stacked packed weights should be valid");
    let stacked_scales = runtime
        .array_from_f32(
            &vec![1.0; QUANTIZATION_VALUES_PER_EXPERT * EXPERT_CAPACITY],
            &[EXPERT_CAPACITY as i32, OUTPUT_DIMENSION, 1],
        )
        .expect("the stacked scales should be valid");
    let stacked_biases = runtime
        .array_from_f32(
            &vec![0.0; QUANTIZATION_VALUES_PER_EXPERT * EXPERT_CAPACITY],
            &[EXPERT_CAPACITY as i32, OUTPUT_DIMENSION, 1],
        )
        .expect("the stacked biases should be valid");
    runtime
        .gather_quantized_matmul_affine(
            activations,
            &stacked_packed_weights,
            &stacked_scales,
            &stacked_biases,
            None,
            Some(selected_expert_indices),
            true,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
            false,
        )
        .expect("the stacked affine reference should build")
}

fn append_u32_values(destination: &mut Vec<u8>, values: &[u32]) {
    for value in values {
        destination.extend_from_slice(&value.to_ne_bytes());
    }
}

fn append_f32_values(destination: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        destination.extend_from_slice(&value.to_ne_bytes());
    }
}
