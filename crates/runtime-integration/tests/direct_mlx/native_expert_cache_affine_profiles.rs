use std::fs;

use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxNativeExpertCache, MlxNativeExpertLayerDescriptor,
    MlxNativeExpertParameter, MlxNativeExpertProjection, MlxNativeExpertTensorSourceDescriptor,
    MlxRuntime,
};

use crate::common::runtime_test_support::runtime;

const INPUT_DIMENSION: i32 = 128;
const OUTPUT_DIMENSION: i32 = 32;

#[test]
fn should_match_stacked_gather_qmm_for_every_supported_native_affine_profile() {
    let runtime = runtime();
    for quantization_bits in [2, 3, 4, 5, 6, 8] {
        for quantization_group_size in [32, 64, 128] {
            eprintln!(
                "[native-expert-affine-profile] status=progress bits={quantization_bits} group_size={quantization_group_size}"
            );
            assert_native_profile_matches_stacked_reference(
                &runtime,
                quantization_bits,
                quantization_group_size,
            );
        }
    }
}

#[test]
fn should_support_mixed_affine_profiles_in_one_native_layer() {
    let runtime = runtime();
    let fixture_directory =
        tempfile::tempdir().expect("the mixed affine fixture directory should exist");
    let source_file_path = fixture_directory.path().join("mixed-expert-source.bin");
    let projection_profiles = [
        (MlxNativeExpertProjection::Gate, 2, 32),
        (MlxNativeExpertProjection::Up, 6, 64),
        (MlxNativeExpertProjection::Down, 8, 128),
    ];
    let mut source_file_bytes = Vec::new();
    let mut tensor_sources = Vec::new();
    for (projection, quantization_bits, quantization_group_size) in projection_profiles {
        let packed_words_per_output_row = INPUT_DIMENSION * quantization_bits / i32::BITS as i32;
        let quantization_group_count = INPUT_DIMENSION / quantization_group_size;
        let packed_weight_values =
            vec![0x1111_1111; (OUTPUT_DIMENSION * packed_words_per_output_row) as usize];
        let quantization_scale_values =
            vec![1.0; (OUTPUT_DIMENSION * quantization_group_count) as usize];
        let quantization_bias_values =
            vec![0.0; (OUTPUT_DIMENSION * quantization_group_count) as usize];
        append_source(
            &mut source_file_bytes,
            &mut tensor_sources,
            &source_file_path,
            projection,
            MlxNativeExpertParameter::PackedWeight,
            quantization_group_size,
            quantization_bits,
            &u32_bytes(&packed_weight_values),
            vec![1, OUTPUT_DIMENSION, packed_words_per_output_row],
            MlxDtype::UInt32,
        );
        append_source(
            &mut source_file_bytes,
            &mut tensor_sources,
            &source_file_path,
            projection,
            MlxNativeExpertParameter::Scales,
            quantization_group_size,
            quantization_bits,
            &f32_bytes(&quantization_scale_values),
            vec![1, OUTPUT_DIMENSION, quantization_group_count],
            MlxDtype::Float32,
        );
        append_source(
            &mut source_file_bytes,
            &mut tensor_sources,
            &source_file_path,
            projection,
            MlxNativeExpertParameter::Biases,
            quantization_group_size,
            quantization_bits,
            &f32_bytes(&quantization_bias_values),
            vec![1, OUTPUT_DIMENSION, quantization_group_count],
            MlxDtype::Float32,
        );
    }
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the mixed affine source fixture should be writable");
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(1, tensor_sources)],
        source_file_bytes.len() as u64,
    )
    .expect("one layer should accept independent projection quantization profiles");
    let selected_expert_indices = runtime
        .array_from_i32(&[0], &[1])
        .expect("the mixed-profile route should be valid");
    let (native_snapshot, _) = native_expert_cache
        .prepare_layer(&runtime, 0, &selected_expert_indices, true)
        .expect("the mixed-profile expert should load");
    let activations = runtime
        .array_from_f32(&[1.0; INPUT_DIMENSION as usize], &[1, INPUT_DIMENSION])
        .expect("the mixed-profile activations should be valid");
    for (projection, quantization_bits, quantization_group_size) in projection_profiles {
        let packed_words_per_output_row = INPUT_DIMENSION * quantization_bits / i32::BITS as i32;
        let quantization_group_count = INPUT_DIMENSION / quantization_group_size;
        let packed_weight_values =
            vec![0x1111_1111; (OUTPUT_DIMENSION * packed_words_per_output_row) as usize];
        let quantization_scale_values =
            vec![1.0; (OUTPUT_DIMENSION * quantization_group_count) as usize];
        let quantization_bias_values =
            vec![0.0; (OUTPUT_DIMENSION * quantization_group_count) as usize];
        let native_output = native_snapshot
            .gather_matmul(
                &runtime,
                projection,
                &activations,
                &selected_expert_indices,
                true,
                false,
            )
            .expect("each mixed-profile native projection should build");
        let stacked_output = stacked_reference(
            &runtime,
            &activations,
            &selected_expert_indices,
            quantization_bits,
            quantization_group_size,
            packed_words_per_output_row,
            quantization_group_count,
            &packed_weight_values,
            &quantization_scale_values,
            &quantization_bias_values,
        );
        assert_eq!(
            native_output
                .to_vec_f32()
                .expect("the mixed-profile native output should evaluate"),
            stacked_output
                .to_vec_f32()
                .expect("the mixed-profile stacked output should evaluate")
        );
    }
}

fn assert_native_profile_matches_stacked_reference(
    runtime: &MlxRuntime,
    quantization_bits: i32,
    quantization_group_size: i32,
) {
    let fixture_directory = tempfile::tempdir().expect("the affine fixture directory should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let packed_words_per_output_row = INPUT_DIMENSION * quantization_bits / i32::BITS as i32;
    let packed_word_count = (OUTPUT_DIMENSION * packed_words_per_output_row) as usize;
    let quantization_group_count = INPUT_DIMENSION / quantization_group_size;
    let quantization_value_count = (OUTPUT_DIMENSION * quantization_group_count) as usize;
    let packed_weight_values = vec![0x1111_1111; packed_word_count];
    let quantization_scale_values = vec![1.0; quantization_value_count];
    let quantization_bias_values = vec![0.0; quantization_value_count];
    let (source_file_bytes, tensor_sources) = build_source_fixture(
        &source_file_path,
        quantization_group_size,
        quantization_bits,
        packed_words_per_output_row,
        quantization_group_count,
        &packed_weight_values,
        &quantization_scale_values,
        &quantization_bias_values,
    );
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the affine source fixture should be writable");
    let native_expert_cache = MlxNativeExpertCache::new(
        runtime,
        &[MlxNativeExpertLayerDescriptor::new(1, tensor_sources)],
        source_file_bytes.len() as u64,
    )
    .expect("the native affine profile should be accepted");
    let selected_expert_indices = runtime
        .array_from_i32(&[0], &[1])
        .expect("the selected expert index should be valid");
    let (native_snapshot, request_report) = native_expert_cache
        .prepare_layer(runtime, 0, &selected_expert_indices, true)
        .expect("the native affine expert should load");
    assert_eq!(request_report.cache_miss_count(), 1);
    assert_eq!(request_report.payload_copy_byte_count(), 0);
    let activations = runtime
        .array_from_f32(&[1.0; INPUT_DIMENSION as usize], &[1, INPUT_DIMENSION])
        .expect("the affine activations should be valid");
    let native_output = native_snapshot
        .gather_matmul(
            runtime,
            MlxNativeExpertProjection::Up,
            &activations,
            &selected_expert_indices,
            true,
            false,
        )
        .expect("the native affine gathered product should build");
    let stacked_output = stacked_reference(
        runtime,
        &activations,
        &selected_expert_indices,
        quantization_bits,
        quantization_group_size,
        packed_words_per_output_row,
        quantization_group_count,
        &packed_weight_values,
        &quantization_scale_values,
        &quantization_bias_values,
    );
    assert_eq!(
        native_output
            .to_vec_f32()
            .expect("the native affine output should evaluate"),
        stacked_output
            .to_vec_f32()
            .expect("the stacked affine output should evaluate")
    );
}

fn build_source_fixture(
    source_file_path: &std::path::Path,
    quantization_group_size: i32,
    quantization_bits: i32,
    packed_words_per_output_row: i32,
    quantization_group_count: i32,
    packed_weight_values: &[u32],
    quantization_scale_values: &[f32],
    quantization_bias_values: &[f32],
) -> (Vec<u8>, Vec<MlxNativeExpertTensorSourceDescriptor>) {
    let mut source_file_bytes = Vec::new();
    let mut tensor_sources = Vec::new();
    for projection in [
        MlxNativeExpertProjection::Gate,
        MlxNativeExpertProjection::Up,
        MlxNativeExpertProjection::Down,
    ] {
        append_source(
            &mut source_file_bytes,
            &mut tensor_sources,
            source_file_path,
            projection,
            MlxNativeExpertParameter::PackedWeight,
            quantization_group_size,
            quantization_bits,
            &u32_bytes(packed_weight_values),
            vec![1, OUTPUT_DIMENSION, packed_words_per_output_row],
            MlxDtype::UInt32,
        );
        append_source(
            &mut source_file_bytes,
            &mut tensor_sources,
            source_file_path,
            projection,
            MlxNativeExpertParameter::Scales,
            quantization_group_size,
            quantization_bits,
            &f32_bytes(quantization_scale_values),
            vec![1, OUTPUT_DIMENSION, quantization_group_count],
            MlxDtype::Float32,
        );
        append_source(
            &mut source_file_bytes,
            &mut tensor_sources,
            source_file_path,
            projection,
            MlxNativeExpertParameter::Biases,
            quantization_group_size,
            quantization_bits,
            &f32_bytes(quantization_bias_values),
            vec![1, OUTPUT_DIMENSION, quantization_group_count],
            MlxDtype::Float32,
        );
    }
    (source_file_bytes, tensor_sources)
}

#[allow(clippy::too_many_arguments)]
fn append_source(
    source_file_bytes: &mut Vec<u8>,
    tensor_sources: &mut Vec<MlxNativeExpertTensorSourceDescriptor>,
    source_file_path: &std::path::Path,
    projection: MlxNativeExpertProjection,
    parameter: MlxNativeExpertParameter,
    quantization_group_size: i32,
    quantization_bits: i32,
    parameter_bytes: &[u8],
    expert_shape: Vec<i32>,
    dtype: MlxDtype,
) {
    let tensor_payload_offset = source_file_bytes.len() as u64;
    source_file_bytes.extend_from_slice(parameter_bytes);
    tensor_sources.push(MlxNativeExpertTensorSourceDescriptor::new(
        projection,
        parameter,
        quantization_group_size,
        quantization_bits,
        source_file_path.to_path_buf(),
        tensor_payload_offset,
        parameter_bytes.len(),
        expert_shape,
        dtype,
    ));
}

#[allow(clippy::too_many_arguments)]
fn stacked_reference(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    selected_expert_indices: &MlxArray,
    quantization_bits: i32,
    quantization_group_size: i32,
    packed_words_per_output_row: i32,
    quantization_group_count: i32,
    packed_weight_values: &[u32],
    quantization_scale_values: &[f32],
    quantization_bias_values: &[f32],
) -> MlxArray {
    let packed_weights = runtime
        .array_from_u32(
            packed_weight_values,
            &[1, OUTPUT_DIMENSION, packed_words_per_output_row],
        )
        .expect("the stacked packed weights should be valid");
    let scales = runtime
        .array_from_f32(
            quantization_scale_values,
            &[1, OUTPUT_DIMENSION, quantization_group_count],
        )
        .expect("the stacked scales should be valid");
    let biases = runtime
        .array_from_f32(
            quantization_bias_values,
            &[1, OUTPUT_DIMENSION, quantization_group_count],
        )
        .expect("the stacked biases should be valid");
    runtime
        .gather_quantized_matmul_affine(
            activations,
            &packed_weights,
            &scales,
            &biases,
            None,
            Some(selected_expert_indices),
            true,
            quantization_group_size,
            quantization_bits,
            false,
        )
        .expect("the stacked affine reference should build")
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}
