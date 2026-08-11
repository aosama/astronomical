use std::fs;

use astronomical_runtime_integration::{
    MlxDtype, MlxNativeExpertCache, MlxNativeExpertLayerDescriptor, MlxNativeExpertParameter,
    MlxNativeExpertProjection, MlxNativeExpertTensorSourceDescriptor,
};

use crate::common::runtime_test_support::runtime;

const EXPERT_CAPACITY: usize = 2;
const INPUT_DIMENSION: i32 = 64;
const OUTPUT_DIMENSION: i32 = 32;

#[test]
fn should_match_stacked_gather_mm_for_native_bfloat16_target_verification_shapes() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the native BF16 fixture should exist");
    let source_file_path = fixture_directory.path().join("native-bfloat16-experts.bin");
    let (source_file_bytes, tensor_sources, expert_weight_values) =
        build_native_bfloat16_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the native BF16 source fixture should be writable");
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            tensor_sources,
        )],
        source_file_bytes.len() as u64,
    )
    .expect("the native BF16 cache should accept three weight sources");
    let selected_expert_indices = runtime
        .array_from_i32(&[0, 1, 1, 0], &[2, 2])
        .expect("the target-verification route should be valid");
    let (native_snapshot, request_report) = native_expert_cache
        .prepare_layer(&runtime, 0, &selected_expert_indices, true)
        .expect("the native BF16 experts should load directly into paged slots");
    assert_eq!(request_report.cache_miss_count(), 2);
    assert_eq!(request_report.successful_source_read_count(), 6);
    assert_eq!(request_report.payload_copy_byte_count(), 0);

    let float32_activations = runtime
        .array_from_f32(
            &[
                vec![1.0; INPUT_DIMENSION as usize],
                vec![2.0; INPUT_DIMENSION as usize],
            ]
            .concat(),
            &[2, 1, INPUT_DIMENSION],
        )
        .expect("the target-verification activations should be valid");
    let activations = runtime
        .astype(&float32_activations, MlxDtype::BFloat16)
        .expect("the activations should preserve native BF16 precision");
    let native_output = native_snapshot
        .gather_matmul(
            &runtime,
            MlxNativeExpertProjection::Down,
            &activations,
            &selected_expert_indices,
            true,
            false,
        )
        .expect("the paged native BF16 product should build");

    let float32_stacked_weights = runtime
        .array_from_f32(
            &expert_weight_values,
            &[EXPERT_CAPACITY as i32, OUTPUT_DIMENSION, INPUT_DIMENSION],
        )
        .expect("the stacked native BF16 reference weights should be valid");
    let stacked_weights = runtime
        .astype(&float32_stacked_weights, MlxDtype::BFloat16)
        .expect("the stacked weights should preserve native BF16 precision");
    let transposed_stacked_weights = runtime
        .transpose_axes(&stacked_weights, &[0, 2, 1])
        .expect("the stacked weights should transpose for gather_mm");
    let stacked_output = runtime
        .gather_dense_matmul(
            &activations,
            &transposed_stacked_weights,
            None,
            Some(&selected_expert_indices),
            false,
        )
        .expect("the stacked native BF16 gather_mm reference should build");
    let native_output_float32 = runtime
        .astype(&native_output, MlxDtype::Float32)
        .expect("the paged native BF16 output should cast for host comparison");
    let stacked_output_float32 = runtime
        .astype(&stacked_output, MlxDtype::Float32)
        .expect("the stacked native BF16 output should cast for host comparison");
    assert_eq!(
        native_output_float32
            .to_vec_f32()
            .expect("the paged native BF16 output should evaluate"),
        stacked_output_float32
            .to_vec_f32()
            .expect("the stacked native BF16 output should evaluate")
    );
}

fn build_native_bfloat16_source_fixture(
    source_file_path: &std::path::Path,
) -> (
    Vec<u8>,
    Vec<MlxNativeExpertTensorSourceDescriptor>,
    Vec<f32>,
) {
    let values_per_expert = (INPUT_DIMENSION * OUTPUT_DIMENSION) as usize;
    let expert_weight_values =
        [vec![1.0; values_per_expert], vec![2.0; values_per_expert]].concat();
    let mut source_file_bytes = Vec::new();
    let mut tensor_sources = Vec::new();
    for projection in [
        MlxNativeExpertProjection::Gate,
        MlxNativeExpertProjection::Up,
        MlxNativeExpertProjection::Down,
    ] {
        let tensor_payload_offset = source_file_bytes.len() as u64;
        for value in &expert_weight_values {
            source_file_bytes.extend_from_slice(&bfloat16_bits(*value).to_ne_bytes());
        }
        tensor_sources.push(MlxNativeExpertTensorSourceDescriptor::new(
            projection,
            MlxNativeExpertParameter::PackedWeight,
            0,
            0,
            source_file_path.to_path_buf(),
            tensor_payload_offset,
            values_per_expert * size_of::<u16>(),
            vec![1, OUTPUT_DIMENSION, INPUT_DIMENSION],
            MlxDtype::BFloat16,
        ));
    }
    (source_file_bytes, tensor_sources, expert_weight_values)
}

fn bfloat16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}
